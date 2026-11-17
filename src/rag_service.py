"""
rag_service.py - RAG (Retrieval-Augmented Generation) untuk menjawab pertanyaan pelanggan Firaza (Side Quest).

Komponen:
  1. Basis pengetahuan  : data/knowledge_base.json (FAQ dan katalog CONTOH, sintetis)
  2. Model embedding    : bge-m3 (BAAI, Hugging Face) dijalankan lewat Ollama   [RAG_EMBED_BACKEND=ollama]
  3. Vector database    : FAISS (IndexFlatIP, cosine karena vektor dinormalisasi); numpy bila FAISS tidak terpasang
  4. Model generation   : SLM lokal Qwen2.5-3B lewat Ollama (cadangan: Gemini, lalu jawaban ekstraktif)

Pengaman: bila skor kemiripan teratas di bawah ambang (RAG_MIN_SCORE), sistem MENOLAK menjawab dan mengarahkan ke admin,
supaya tidak mengarang. Jawaban selalu disertai sumber.

Konfigurasi (env / .env):
  RAG_EMBED_BACKEND ollama|hash   (hash = embedding sederhana tanpa model, hanya untuk pengujian)
  RAG_EMBED_MODEL   bge-m3        RAG_LLM_BACKEND ollama|gemini|extractive     RAG_LLM_MODEL qwen2.5:3b
  OLLAMA_URL        http://127.0.0.1:11434     RAG_MIN_SCORE (default 0.45 untuk bge-m3, hitung ulang dengan eval_rag.py)
  RAG_KB            data/knowledge_base.json   RAG_INDEX_DIR models/rag

CLI:  python src/rag_service.py --build | --status | --ask "Bagaimana cara melacak pesanan?"
"""
import argparse
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import requests

try:
    import faiss
except Exception:          # FAISS opsional: ada cadangan numpy
    faiss = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SYSTEM_PROMPT = """Kamu adalah asisten layanan pelanggan toko Firaza (hampers, handuk, sajadah, dan kado personalisasi).
Jawab pertanyaan HANYA berdasarkan KONTEKS yang diberikan.
Aturan:
- Jika konteks tidak memuat jawabannya, katakan bahwa kamu belum punya informasinya dan sarankan menghubungi admin.
- Jangan mengarang harga, waktu pengerjaan, kebijakan, atau fakta lain yang tidak ada di konteks.
- Jawab dalam Bahasa Indonesia yang ramah, singkat (maksimal 4 kalimat), tanpa emoji.
- Sebut sumber dengan nomor konteks, misalnya [1] atau [2]."""

USER_TEMPLATE = """KONTEKS:
{context}

PERTANYAAN: {question}

JAWABAN:"""

ABSTAIN_TEXT = ("Maaf, saya belum menemukan informasi tentang itu di panduan Firaza. "
                "Silakan hubungi admin Firaza lewat WhatsApp atau Instagram resmi untuk bantuan lebih lanjut.")


class RagUnavailable(Exception):
    """Layanan pendukung (Ollama, Gemini) tidak dapat dipakai."""


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.maximum(n, 1e-12)).astype("float32")


# ---------------------------------------------------------------- embedding
STOP = {"yang", "di", "ke", "dan", "atau", "itu", "ini", "untuk", "dengan", "dari", "pada", "adalah", "saya", "anda", "sih"}


class HashEmbedder:
    """Embedding sederhana (hash kata dan pasangan kata). Hanya untuk pengujian tanpa model; kualitasnya rendah."""
    name, dim = "hash", 512

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            toks = [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]
            feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
            for f in feats:
                out[i, int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        return _normalize(np.sqrt(out))


class OllamaEmbedder:
    def __init__(self):
        self.model = _env("RAG_EMBED_MODEL", "bge-m3")
        self.url = _env("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.name = f"ollama:{self.model}"

    def embed(self, texts: List[str]) -> np.ndarray:
        try:
            r = requests.post(self.url + "/api/embed", json={"model": self.model, "input": list(texts)}, timeout=180)
        except requests.exceptions.RequestException as e:
            raise RagUnavailable(f"Ollama tidak dapat dihubungi di {self.url} ({type(e).__name__}). Pastikan Ollama berjalan.")
        if r.status_code != 200:
            raise RagUnavailable(f"Ollama menjawab {r.status_code}: {r.text[:200]} (apakah model '{self.model}' sudah di-pull?)")
        return _normalize(np.array(r.json()["embeddings"], dtype="float32"))


def make_embedder():
    return HashEmbedder() if _env("RAG_EMBED_BACKEND", "ollama") == "hash" else OllamaEmbedder()


# ---------------------------------------------------------------- vector index
class VectorIndex:
    """Indeks kemiripan cosine: FAISS IndexFlatIP bila tersedia, jika tidak numpy (hasil sama)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self._faiss = None
        if faiss is not None:
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def search(self, q: np.ndarray, k: int):
        k = min(k, len(self.vectors))
        q = np.ascontiguousarray(q.reshape(1, -1), dtype="float32")
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        s = self.vectors @ q[0]
        ids = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in ids]


# ---------------------------------------------------------------- layanan RAG
class RagService:
    def __init__(self, kb_path: Optional[str] = None, index_dir: Optional[str] = None):
        self.kb_path = kb_path or _env("RAG_KB", "data/knowledge_base.json")
        self.index_dir = index_dir or _env("RAG_INDEX_DIR", "models/rag")
        with open(self.kb_path, encoding="utf-8") as f:
            self.docs: List[Dict[str, Any]] = json.load(f)
        self.embedder = make_embedder()
        self.index: Optional[VectorIndex] = None
        self.llm_backend = _env("RAG_LLM_BACKEND", "ollama")
        self.llm_model = _env("RAG_LLM_MODEL", "qwen2.5:3b")
        self.ollama_url = _env("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.min_score = float(_env("RAG_MIN_SCORE", "0.15" if self.embedder.name == "hash" else "0.45"))

    # ---- indeks
    def _kb_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.docs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def _paths(self):
        return os.path.join(self.index_dir, "vectors.npy"), os.path.join(self.index_dir, "meta.json")

    def build_index(self) -> Dict[str, Any]:
        texts = [f"{d['judul']}. {d['isi']}" for d in self.docs]
        vecs = self.embedder.embed(texts)
        os.makedirs(self.index_dir, exist_ok=True)
        vp, mp = self._paths()
        np.save(vp, vecs)
        meta = {"embedder": self.embedder.name, "kb_hash": self._kb_hash(), "n": len(self.docs), "dim": int(vecs.shape[1]),
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self.index = VectorIndex(vecs)
        return meta

    def ensure_index(self) -> None:
        if self.index is not None:
            return
        vp, mp = self._paths()
        if os.path.exists(vp) and os.path.exists(mp):
            meta = json.load(open(mp, encoding="utf-8"))
            if meta.get("embedder") == self.embedder.name and meta.get("kb_hash") == self._kb_hash():
                self.index = VectorIndex(np.load(vp))
                return
        self.build_index()          # belum ada atau basis pengetahuan/model berubah -> bangun ulang

    def status(self) -> Dict[str, Any]:
        return {"dokumen": len(self.docs), "embedding": self.embedder.name, "indeks_siap": self.index is not None,
                "indeks_backend": self.index.backend if self.index else ("faiss" if faiss is not None else "numpy"),
                "llm_backend": self.llm_backend, "llm_model": self.llm_model, "ambang_skor": self.min_score}

    # ---- pencarian
    def retrieve(self, question: str, k: int = 3) -> List[Dict[str, Any]]:
        self.ensure_index()
        q = self.embedder.embed([question])[0]
        return [{**self.docs[i], "skor": s} for i, s in self.index.search(q, k)]

    # ---- generasi
    def _chat_ollama(self, user: str) -> str:
        try:
            r = requests.post(self.ollama_url + "/api/chat", timeout=180, json={
                "model": self.llm_model, "stream": False, "keep_alive": "10m",
                "options": {"temperature": 0.2, "num_ctx": 2048},
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]})
        except requests.exceptions.RequestException as e:
            raise RagUnavailable(f"Ollama tidak dapat dihubungi ({type(e).__name__})")
        if r.status_code != 200:
            raise RagUnavailable(f"Ollama menjawab {r.status_code}: {r.text[:150]}")
        return r.json()["message"]["content"]

    def _chat_gemini(self, user: str) -> str:
        from genai_service import GenAIService, response_text
        svc = GenAIService()
        if not svc.available:
            raise RagUnavailable("GEMINI_API_KEY belum diatur")
        from google.genai import types
        resp = svc.client.models.generate_content(model=svc.model, contents=user, config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, temperature=0.2))
        return response_text(resp)

    def ask(self, question: str, k: int = 3, llm: Optional[str] = None) -> Dict[str, Any]:
        hits = self.retrieve(question, k)
        sources = [{"id": h["id"], "judul": h["judul"], "kategori": h["kategori"], "skor": round(h["skor"], 3),
                    "kutipan": h["isi"][:220]} for h in hits]
        if not hits or hits[0]["skor"] < self.min_score:
            return {"jawaban": ABSTAIN_TEXT, "mode": "abstain", "abstained": True, "sumber": [], "peringatan":
                    [f"Skor kemiripan tertinggi {hits[0]['skor']:.2f} di bawah ambang {self.min_score:.2f}"] if hits else []}
        context = "\n".join(f"[{i}] {h['judul']}: {h['isi']}" for i, h in enumerate(hits, 1))
        user = USER_TEMPLATE.format(context=context, question=question)
        warnings: List[str] = []
        backend = llm or self.llm_backend
        order = {"ollama": ["ollama", "gemini"], "gemini": ["gemini"], "extractive": []}.get(backend, ["ollama", "gemini"])
        for b in order:
            try:
                text = (self._chat_ollama(user) if b == "ollama" else self._chat_gemini(user)).strip()
                if not text:
                    raise RagUnavailable("keluaran kosong")
                return {"jawaban": text, "mode": f"ollama:{self.llm_model}" if b == "ollama" else "gemini",
                        "abstained": False, "sumber": sources, "peringatan": warnings}
            except RagUnavailable as e:
                warnings.append(f"{b} gagal: {e}")
        return {"jawaban": hits[0]["isi"], "mode": "extractive", "abstained": False, "sumber": sources,
                "peringatan": warnings + (["Menampilkan kutipan sumber terdekat (tanpa model bahasa)"] if order else [])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--ask", default=None)
    a = ap.parse_args()
    svc = RagService()
    try:
        if a.build:
            print(json.dumps(svc.build_index(), indent=2))
        if a.ask:
            print(json.dumps(svc.ask(a.ask), indent=2, ensure_ascii=False))
        if a.status or not (a.build or a.ask):
            print(json.dumps(svc.status(), indent=2))
    except RagUnavailable as e:
        raise SystemExit(f"GAGAL: {e}")
