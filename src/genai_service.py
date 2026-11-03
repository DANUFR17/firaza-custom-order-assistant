"""
genai_service.py - Fitur Generative AI (API proprietary Gemini) untuk Firaza.

Dua fitur (fitur KEDUA setelah klasifikasi oleh model sendiri, sesuai aturan Main Quest):
  1. Structured order summary (JSON)  -> OrderSummary
  2. Draft kartu ucapan               -> teks

Mitigasi risiko (dari project plan):
  - Output Gemini dibatasi skema JSON (response_schema) dan divalidasi Pydantic.
  - Nama dan jumlah TIDAK dipercaya begitu saja: dicek terhadap teks asli customer ("grounding check");
    bila ada item terstruktur di database, nilainya yang dipakai.
  - Bila Gemini tidak tersedia / error / output tidak valid, dipakai fallback berbasis aturan
    (sumber = "fallback") sehingga API tetap berfungsi saat demo.
  - Hasil di-cache per proses agar tidak memanggil API berulang untuk input yang sama.

Konfigurasi (.env, JANGAN di-commit):
  GEMINI_API_KEY=...        (atau GOOGLE_API_KEY)
  GEMINI_MODEL=...          (opsional; cek daftar model terbaru di Google AI Studio)

Cek koneksi dan model:
  python src/genai_service.py --check
"""
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv opsional
    pass

from schemas import GeminiSummary

DEFAULT_MODEL = "gemini-3.1-flash-lite"   # ganti lewat GEMINI_MODEL bila Google memperbarui daftar model

# ---------------------------------------------------------------- PROMPT (didokumentasikan di docs/prompts.md)
SUMMARY_SYSTEM_PROMPT = """Kamu adalah asisten admin toko Firaza (hampers, handuk, sajadah, dan kado personalisasi).
Tugasmu mengubah pesan customer menjadi ringkasan order terstruktur dalam Bahasa Indonesia.

Aturan ketat:
- Gunakan HANYA informasi yang ada di pesan customer atau data item yang diberikan. Jangan mengarang.
- Nama/teks custom harus disalin persis seperti ditulis customer (jangan diperbaiki atau diterjemahkan).
- Jumlah hanya diisi jika disebut jelas; jika tidak, isi null.
- Jika informasi penting belum jelas (jumlah, nama/teks custom, warna, font, tenggat), tulis di perlu_klarifikasi.
- Ringkasan maksimal 2 kalimat, netral, tanpa emoji.
- Jangan menentukan jenis order; itu sudah ditentukan sistem."""

SUMMARY_USER_TEMPLATE = """Jenis order (dari sistem): {order_type}

Pesan customer:
\"\"\"{raw_message}\"\"\"

Data item yang sudah dicatat admin (boleh kosong):
{items_json}

Buat ringkasan order terstruktur."""

GREETING_SYSTEM_PROMPT = """Kamu menulis draft kartu ucapan singkat dalam Bahasa Indonesia untuk hampers/kado dari toko Firaza.

Aturan:
- Maksimal 60 kata, 2-4 kalimat, tanpa emoji dan tanpa hashtag.
- Sesuaikan dengan acara dan nada yang diminta.
- Jika nama penerima diberikan, sebut persis seperti ditulis. Jika tidak ada, jangan mengarang nama.
- Jangan menyebut harga, promo, atau fakta lain yang tidak diberikan.
- Keluarkan HANYA teks kartu ucapan, tanpa pembuka atau penjelasan."""

GREETING_USER_TEMPLATE = """Acara: {acara}
Nada: {tone}
Nama penerima/pada produk: {names}
Konteks pesan customer: \"\"\"{raw_message}\"\"\"

Tulis draft kartu ucapan."""

# ---------------------------------------------------------------- util teks
COLORS = ["navy", "hitam", "maroon", "hijau botol", "hijau sage", "abu-abu", "abu", "krem", "biru dongker", "biru muda",
          "coklat", "cokelat", "ungu", "tosca", "putih", "pink", "merah", "emas", "silver"]
EVENTS = ["pernikahan", "nikahan", "wedding", "lamaran", "akad", "pengajian", "wisuda", "ulang tahun", "aqiqah",
          "akikah", "lebaran", "idul fitri", "umroh", "haji", "tasyakuran", "syukuran", "khitanan", "reuni",
          "perpisahan", "acara kantor", "gathering", "arisan", "souvenir"]
PRODUCTS = [("sajadah", "sajadah"), ("sjd", "sajadah"), ("handuk", "handuk"), ("hnduk", "handuk"),
            ("hampers", "hampers"), ("parcel", "hampers"), ("seserahan", "hampers")]
UNITS = r"(?:pcs|pc|biji|buah|box|paket|set|lembar)"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w&]+", " ", (s or "").lower())).strip()


def _numbers(s: str) -> set:
    return {int(x) for x in re.findall(r"\d+", s or "")}


def extract_basic(raw: str) -> Dict[str, Any]:
    """Ekstraksi sederhana berbasis aturan (dipakai fallback dan sebagai pembanding)."""
    low = (raw or "").lower()
    out: Dict[str, Any] = {"produk": None, "jumlah": None, "warna": None, "teks_custom": None, "acara": None}
    for key, name in PRODUCTS:
        if re.search(rf"\b{key}\b", low):
            out["produk"] = name
            break
    m = re.search(rf"\b(\d{{1,5}})\s*{UNITS}\b", low) or re.search(rf"{UNITS}\s*(\d{{1,5}})\b", low)
    if m:
        out["jumlah"] = int(m.group(1))
    for c in sorted(COLORS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(c)}\b", low):
            out["warna"] = c
            break
    for e in EVENTS:
        if re.search(rf"\b{e}\b", low):
            out["acara"] = e
            break
    q = re.search(r"['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]{1,60})['\"\u2018\u2019\u201c\u201d]", raw or "")
    if q:
        out["teks_custom"] = q.group(1).strip()
    else:
        n = re.search(r"(?:nama|atas nama|bertuliskan|tulisan|inisial)\s+([A-Z][\w.]*(?:\s*(?:&|dan|\+)\s*[A-Z][\w.]*|\s+[A-Z][\w.]*)*)", raw or "")
        if n:
            out["teks_custom"] = n.group(1).strip()
    return out


def fallback_summary(raw: str, items: List[Dict[str, Any]], order_type: Optional[str], reason: str = "") -> Dict[str, Any]:
    ex = extract_basic(raw)
    summary_items: List[Dict[str, Any]] = []
    if items:
        for it in items:
            summary_items.append({"produk": it.get("produk") or ex["produk"] or "tidak disebutkan", "jumlah": it.get("jumlah"),
                                  "warna": it.get("warna"), "teks_custom": it.get("teks_custom"),
                                  "font": it.get("font"), "layout": it.get("layout")})
    elif ex["produk"] or ex["jumlah"] or ex["teks_custom"]:
        summary_items.append({"produk": ex["produk"] or "tidak disebutkan", "jumlah": ex["jumlah"], "warna": ex["warna"],
                              "teks_custom": ex["teks_custom"], "font": None, "layout": None})
    clar: List[str] = []
    is_custom = (order_type or "").startswith("Custom")
    first = summary_items[0] if summary_items else {}
    if not first.get("jumlah"):
        clar.append("Jumlah belum disebutkan")
    if is_custom and not first.get("teks_custom"):
        clar.append("Nama/teks custom belum jelas")
    if is_custom and not first.get("font"):
        clar.append("Font belum dipilih")
    parts = []
    if summary_items:
        it = summary_items[0]
        desc = " ".join(str(x) for x in [it.get("jumlah") and f"{it['jumlah']} pcs", it["produk"], it.get("warna")] if x)
        parts.append(desc)
    acara = ex["acara"]
    sentence = f"{order_type or 'Order'}: " + (parts[0] if parts else "detail produk belum jelas")
    if acara:
        sentence += f" untuk {acara}"
    if first.get("teks_custom"):
        sentence += f", teks custom \"{first['teks_custom']}\""
    return {"ringkasan": sentence + ".", "items": summary_items, "acara": acara, "tenggat": None,
            "perlu_klarifikasi": clar, "sumber": "fallback",
            "peringatan": ([f"Memakai ringkasan berbasis aturan: {reason}"] if reason else [])}


def ground_summary(data: Dict[str, Any], source_text: str, db_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cek bahwa nama/jumlah dari Gemini benar-benar ada di teks sumber; jika ada item DB, pakai nilainya."""
    warnings: List[str] = list(data.get("peringatan", []))
    src_norm = _norm(source_text + " " + " ".join(str(i.get("teks_custom") or "") for i in db_items))
    src_nums = _numbers(source_text) | {i.get("jumlah") for i in db_items if i.get("jumlah")}
    if db_items:
        data["items"] = [{"produk": i.get("produk") or "tidak disebutkan", "jumlah": i.get("jumlah"), "warna": i.get("warna"),
                          "teks_custom": i.get("teks_custom"), "font": i.get("font"), "layout": i.get("layout")}
                         for i in db_items]
    else:
        for it in data.get("items", []):
            t = it.get("teks_custom")
            if t and _norm(t) not in src_norm:
                warnings.append(f"Teks custom '{t}' tidak ditemukan di pesan customer; dikosongkan")
                it["teks_custom"] = None
            j = it.get("jumlah")
            if j and j not in src_nums:
                warnings.append(f"Jumlah {j} tidak ditemukan di pesan customer; dikosongkan")
                it["jumlah"] = None
    data["peringatan"] = warnings
    return data


GREETING_FALLBACKS = {
    "pernikahan": "Selamat menempuh hidup baru{names}. Semoga cinta kalian dipenuhi keberkahan, kebahagiaan, dan kebersamaan yang abadi.",
    "wisuda": "Selamat atas kelulusan{names}. Semoga ilmu yang didapat membawa manfaat dan membuka jalan menuju masa depan yang cerah.",
    "ulang tahun": "Selamat ulang tahun{names}. Semoga sehat, bahagia, dan dimudahkan dalam setiap langkah di tahun yang baru.",
    "lebaran": "Selamat Idul Fitri{names}. Taqabbalallahu minna wa minkum. Mohon maaf lahir dan batin.",
    "aqiqah": "Selamat atas kelahiran sang buah hati{names}. Semoga tumbuh menjadi anak yang sholeh/sholehah dan membanggakan keluarga.",
    "umum": "Terima kasih telah menjadi bagian dari momen istimewa ini{names}. Semoga hadiah kecil ini membawa kebahagiaan.",
}
EVENT_GROUP = {"pernikahan": "pernikahan", "nikahan": "pernikahan", "wedding": "pernikahan", "lamaran": "pernikahan",
               "akad": "pernikahan", "wisuda": "wisuda", "ulang tahun": "ulang tahun", "lebaran": "lebaran",
               "idul fitri": "lebaran", "aqiqah": "aqiqah", "akikah": "aqiqah"}


def fallback_greeting(acara: Optional[str], names: str) -> str:
    group = EVENT_GROUP.get((acara or "").lower(), "umum")
    return GREETING_FALLBACKS[group].format(names=f" untuk {names}" if names else "")


class GenAIService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self._client = None
        self._cache: Dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def client(self):
        if self._client is None:
            from google import genai   # impor lazy: API tetap jalan walau paket/key belum ada
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    # -------- pemanggilan Gemini (dipisah agar mudah di-mock saat pengujian)
    def _call_summary(self, prompt: str) -> str:
        from google.genai import types
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SUMMARY_SYSTEM_PROMPT, temperature=0.2,
                                               response_mime_type="application/json", response_schema=GeminiSummary))
        return resp.text

    def _call_greeting(self, prompt: str) -> str:
        from google.genai import types
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=GREETING_SYSTEM_PROMPT, temperature=0.7))
        return resp.text

    @staticmethod
    def _key(kind: str, model: str, prompt: str) -> str:
        return hashlib.sha256(f"{kind}|{model}|{prompt}".encode()).hexdigest()

    # -------- fitur 1: structured summary
    def summarize(self, raw_message: Optional[str], items: List[Dict[str, Any]], order_type: Optional[str],
                  use_cache: bool = True) -> Dict[str, Any]:
        raw = raw_message or ""
        if not self.available:
            return fallback_summary(raw, items, order_type, "GEMINI_API_KEY belum diatur")
        prompt = SUMMARY_USER_TEMPLATE.format(order_type=order_type or "belum ditentukan", raw_message=raw,
                                              items_json=json.dumps(items, ensure_ascii=False) if items else "[]")
        key = self._key("summary", self.model, prompt)
        if use_cache and key in self._cache:
            return json.loads(json.dumps(self._cache[key]))
        try:
            parsed = GeminiSummary.model_validate_json(self._call_summary(prompt))
            data = parsed.model_dump()
            data["sumber"] = "gemini"
            data["peringatan"] = []
            data = ground_summary(data, raw, items)
        except Exception as e:  # jaringan, kuota, model salah nama, JSON tidak valid, dll.
            return fallback_summary(raw, items, order_type, f"Gemini gagal ({type(e).__name__})")
        self._cache[key] = data
        return json.loads(json.dumps(data))

    # -------- fitur 2: kartu ucapan
    def greeting(self, raw_message: Optional[str], acara: Optional[str], names: str, tone: str = "hangat",
                 use_cache: bool = True) -> Dict[str, Any]:
        acara = acara or extract_basic(raw_message or "")["acara"]
        if not self.available:
            return {"kartu_ucapan": fallback_greeting(acara, names), "sumber": "fallback",
                    "peringatan": ["Memakai template: GEMINI_API_KEY belum diatur"]}
        prompt = GREETING_USER_TEMPLATE.format(acara=acara or "tidak disebutkan", tone=tone,
                                               names=names or "tidak ada", raw_message=raw_message or "")
        key = self._key(f"greeting|{tone}", self.model, prompt)
        if use_cache and key in self._cache:
            return dict(self._cache[key])
        try:
            text = (self._call_greeting(prompt) or "").strip().strip('"')
            if not text or len(text) > 700:
                raise ValueError("keluaran kosong atau terlalu panjang")
            data = {"kartu_ucapan": text, "sumber": "gemini", "peringatan": []}
        except Exception as e:
            return {"kartu_ucapan": fallback_greeting(acara, names), "sumber": "fallback",
                    "peringatan": [f"Memakai template: Gemini gagal ({type(e).__name__})"]}
        self._cache[key] = data
        return dict(data)


def _check():
    svc = GenAIService()
    print(f"Model       : {svc.model}")
    print(f"API key     : {'ada' if svc.available else 'TIDAK ADA (isi GEMINI_API_KEY di .env)'}")
    if not svc.available:
        return
    try:
        names = [m.name for m in svc.client.models.list()]
        flash = [n for n in names if "flash" in n]
        print(f"Model tersedia untuk key ini (contoh, yang mengandung 'flash'): {flash[:12]}")
    except Exception as e:
        print(f"Gagal mengambil daftar model: {type(e).__name__}: {e}")
    out = svc.summarize("Mau sajadah navy 20 pcs untuk nikahan, custom nama Budi & Sarah", [], "Custom Sajadah")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="uji key, daftar model, dan satu panggilan summary")
    if ap.parse_args().check:
        _check()
