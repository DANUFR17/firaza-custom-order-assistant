"""Uji RAG tanpa Ollama: embedding 'hash' dan generasi 'extractive'; panggilan Ollama dipalsukan dengan monkeypatch."""
import json
import os

import pytest
import requests
from fastapi.testclient import TestClient

import api as api_module
import rag_service
from rag_service import RagService, RagUnavailable

KB = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBED_BACKEND", "hash")
    monkeypatch.setenv("RAG_LLM_BACKEND", "extractive")
    return RagService(kb_path=KB, index_dir=str(tmp_path / "rag"))


def test_retrieval_menemukan_dokumen_yang_benar(svc):
    cases = {"Bagaimana cara melacak status pesanan saya": "kb11", "Pilihan font dan posisi teks apa saja": "kb13",
             "Boleh kirim logo perusahaan sendiri": "kb24", "Saya mau jadi reseller": "kb20"}
    for q, expected in cases.items():
        assert expected in [h["id"] for h in svc.retrieve(q, k=3)], q


def test_menolak_pertanyaan_di_luar_domain(svc):
    out = svc.ask("Resep nasi goreng yang enak")
    assert out["abstained"] and out["mode"] == "abstain" and out["sumber"] == []
    ok = svc.ask("Bagaimana cara melacak pesanan saya?")
    assert not ok["abstained"] and ok["mode"] == "extractive" and ok["sumber"][0]["id"] in ("kb11", "kb12")


def test_indeks_disimpan_dan_dibangun_ulang_bila_basis_berubah(svc, tmp_path):
    meta = svc.build_index()
    assert os.path.exists(os.path.join(svc.index_dir, "vectors.npy")) and meta["n"] == len(svc.docs)
    s2 = RagService(kb_path=KB, index_dir=svc.index_dir)
    s2.ensure_index()
    assert s2.index is not None                                          # dimuat dari disk
    kb2 = tmp_path / "kb2.json"
    kb2.write_text(json.dumps(svc.docs[:5], ensure_ascii=False), encoding="utf-8")
    s3 = RagService(kb_path=str(kb2), index_dir=svc.index_dir)
    s3.ensure_index()
    assert len(s3.index.vectors) == 5                                    # hash basis berubah -> bangun ulang


def test_ollama_embedder_mengirim_permintaan_yang_benar(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_EMBED_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_URL", "http://ollama.test:11434")
    sent = {}

    class R:
        status_code = 200

        def json(self):
            return {"embeddings": [[1.0, 0.0]] * len(sent["json"]["input"])}

    def fake_post(url, json=None, timeout=None):
        sent.update(url=url, json=json)
        return R()

    monkeypatch.setattr(requests, "post", fake_post)
    s = RagService(kb_path=KB, index_dir=str(tmp_path / "r"))
    s.build_index()
    assert sent["url"] == "http://ollama.test:11434/api/embed" and sent["json"]["model"] == "bge-m3"
    assert len(sent["json"]["input"]) == len(s.docs)


def test_generasi_ollama_gagal_turun_ke_extractive(svc, monkeypatch):
    svc.llm_backend = "ollama"
    monkeypatch.setattr(svc, "_chat_ollama", lambda user: (_ for _ in ()).throw(RagUnavailable("mati")))
    monkeypatch.setattr(svc, "_chat_gemini", lambda user: (_ for _ in ()).throw(RagUnavailable("tanpa key")))
    out = svc.ask("Bagaimana cara melacak pesanan saya?")
    assert out["mode"] == "extractive" and len(out["peringatan"]) >= 2


def test_generasi_ollama_berhasil(svc, monkeypatch):
    svc.llm_backend = "ollama"
    monkeypatch.setattr(svc, "_chat_ollama", lambda user: "Gunakan menu Lacak Pesanan [1].")
    out = svc.ask("Bagaimana cara melacak pesanan saya?")
    assert out["mode"].startswith("ollama:") and "Lacak Pesanan" in out["jawaban"] and out["sumber"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRAZA_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("RAG_EMBED_BACKEND", "hash")
    monkeypatch.setenv("RAG_LLM_BACKEND", "extractive")
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "rag"))
    monkeypatch.setenv("RAG_KB", KB)
    with TestClient(api_module.app) as c:
        yield c


def test_api_rag(client):
    st = client.get("/rag/status").json()
    assert st["dokumen"] == 24 and st["embedding"] == "hash"
    r = client.post("/rag/ask", json={"question": "Bagaimana cara melacak pesanan saya?"})
    assert r.status_code == 200 and not r.json()["abstained"] and r.json()["sumber"]
    assert client.post("/rag/ask", json={"question": "Resep nasi goreng yang enak"}).json()["abstained"] is True
    assert client.post("/rag/ask", json={"question": "ab"}).status_code == 422


def test_api_rag_503_bila_ollama_mati(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRAZA_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("RAG_EMBED_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")                # port tertutup
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "rag"))
    monkeypatch.setenv("RAG_KB", KB)
    with TestClient(api_module.app) as c:
        r = c.post("/rag/ask", json={"question": "Bagaimana cara melacak pesanan saya?"})
        assert r.status_code == 503 and "Ollama" in r.json()["detail"]
