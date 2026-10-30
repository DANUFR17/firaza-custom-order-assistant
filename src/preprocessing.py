"""
preprocessing.py - Preprocessing pesan customer Firaza (v1)

Langkah (urutan penting):
 1. Case folding + hapus emoji/URL
 2. Masking: tulisan dalam tanda kutip -> NAMA, dimensi (70x140) -> DIM,
    nominal uang (50rb, Rp200.000) -> UANG, angka lain -> NUM
 3. Normalisasi slang/singkatan (sjd -> sajadah, brp -> berapa, ga -> tidak, ...)
 4. Koreksi typo terbatas: hanya terhadap kamus kata domain (fuzzy match, cutoff tinggi),
    supaya kata umum tidak ikut "dikoreksi" secara salah
 5. Penandaan negasi: kata setelah tidak/bukan/tanpa/belum/jangan diberi awalan NEG_
    sampai batas tanda baca atau kata penghubung (maks. 3 kata)
 6. Hapus tanda baca

Catatan desain:
 - Stopword TIDAK dibuang. Kata seperti "tidak", "bukan", "tanpa" justru pembeda penting
   (mis. "tidak perlu bordir" = Ready Stock, "mau bordir" = Custom Handuk).
 - Semua langkah berbasis aturan (tanpa model pihak ketiga).
"""
import re
from difflib import get_close_matches

SLANG = {
    "sjd": "sajadah", "sjdh": "sajadah", "sajadh": "sajadah", "hnduk": "handuk", "hnd": "handuk",
    "brp": "berapa", "brpa": "berapa", "berpa": "berapa", "utk": "untuk", "yg": "yang",
    "dg": "dengan", "dgn": "dengan", "bs": "bisa", "tlg": "tolong", "krm": "kirim", "hr": "hari",
    "mksh": "makasih", "blm": "belum", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "ngga": "tidak", "gk": "tidak", "tdk": "tidak", "aja": "saja", "udah": "sudah", "udh": "sudah",
    "gimana": "bagaimana", "bj": "pcs", "pc": "pcs", "biji": "pcs", "buah": "pcs", "rdy": "ready",
    "redy": "ready", "hmpers": "hampers", "bordr": "bordir", "pgn": "mau", "mo": "mau",
    "kalo": "kalau", "jg": "juga", "pake": "pakai", "gue": "saya", "aku": "saya", "lu": "kamu",
    "thx": "makasih", "thanks": "makasih", "terimakasih": "makasih", "bgt": "banget",
    "dr": "dari", "sm": "sama", "tp": "tapi", "dlu": "dulu", "bsk": "besok", "skrg": "sekarang",
    "pesen": "pesan", "order": "pesan", "ordernya": "pesan", "diorder": "dipesan",
}

# Kata domain untuk koreksi typo (fuzzy match). Sengaja pendek dan spesifik.
DOMAIN = [
    "sajadah", "handuk", "bordir", "hampers", "parcel", "seserahan", "custom", "ready", "stok",
    "harga", "ongkir", "resi", "komplain", "reseller", "personalisasi", "packaging", "souvenir",
    "tanya", "berapa", "kartu", "ucapan", "pengiriman", "pesanan", "logo", "inisial", "printing",
    "estimasi", "minimal", "katalog", "retur", "paket", "kirim",
]

NEGATORS = {"tidak", "bukan", "tanpa", "belum", "jangan"}
SCOPE_BREAK = {"tapi", "tetapi", "namun", "dan", "atau", "karena", "sekalian", "kok"}
NEG_WINDOW = 3

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
QUOTE_RE = re.compile(r"['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]{1,40})['\"\u2018\u2019\u201c\u201d]")
DIM_RE = re.compile(r"\b\d{2,3}\s?[x\u00d7]\s?\d{2,3}\b")
MONEY_RE = re.compile(r"\b(?:rp\.?\s?\d[\d.,]*|\d[\d.,]*\s?(?:rb|ribu|k|jt|juta))\b")
NUM_RE = re.compile(r"\b\d+\b")
TOKEN_RE = re.compile(r"NAMA|DIM|UANG|NUM|[a-z]+|[.,?!;:]")
PUNCT = set(".,?!;:")


def _fix_typos(tok):
    """Koreksi typo hanya terhadap kamus domain (cutoff tinggi agar kata umum tidak rusak)."""
    if len(tok) < 5 or tok in DOMAIN:
        return tok
    m = get_close_matches(tok, DOMAIN, n=1, cutoff=0.84)
    return m[0] if m else tok


def preprocess(text, negation=True, fuzzy=True, mask=True):
    """Ubah satu pesan mentah menjadi string token yang siap masuk TF-IDF / TextVectorization."""
    t = EMOJI_RE.sub(" ", URL_RE.sub(" ", str(text))).lower()
    if mask:
        t = QUOTE_RE.sub(" NAMA ", t)
        t = DIM_RE.sub(" DIM ", t)
        t = MONEY_RE.sub(" UANG ", t)
        t = NUM_RE.sub(" NUM ", t)
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)      # kakkk -> kak
    t = re.sub(r"\bya+\b", "ya", t)               # yaa -> ya
    out = []
    for tok in TOKEN_RE.findall(t):
        if tok.islower():
            tok = SLANG.get(tok, tok)
            if fuzzy:
                tok = _fix_typos(tok)
            tok = SLANG.get(tok, tok)
        out.append(tok)
    if negation:
        res, left = [], 0
        for tok in out:
            if tok in PUNCT or tok in SCOPE_BREAK:
                left = 0
                res.append(tok)
            elif tok in NEGATORS:
                left = NEG_WINDOW
                res.append(tok)
            elif left > 0 and tok.islower():
                res.append("NEG_" + tok)
                left -= 1
            else:
                res.append(tok)
        out = res
    return " ".join(tok for tok in out if tok not in PUNCT)


if __name__ == "__main__":
    demo = [
        "Kak, yang sajadah polos buat dipakai sendiri ada warna maroon? Aku cuma perlu satu, nggak pakai nama atau tulisan.",
        "p sajadh hijau botol rdy tidak? pgn gue custom nama Siti tlg info yaa 🙏",
        "Mau sajadah navy 20 pcs untuk nikahan, custom nama 'Budi & Sarah' ukuran 70x140 budget Rp200.000",
        "hlo kak tlg rekomendasikan isi pacrel tasyakuran budget 50rb thanks",
    ]
    for d in demo:
        print(d, "\n  ->", preprocess(d), "\n")
