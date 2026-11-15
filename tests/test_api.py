"""
Uji otomatis API (jalankan dari ROOT proyek):  pytest -q
- Database memakai file sementara, bukan data/firaza.db.
- Gemini TIDAK dipanggil sungguhan: kunci dihapus (jalur fallback) atau metode pemanggilan di-mock.
- Uji yang butuh model klasifikasi dilewati bila models/final belum ada.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

import api as api_module
from genai_service import GenAIService

BRIEF = "Mau sajadah navy 20 pcs untuk nikahan, custom nama Budi & Sarah"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRAZA_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with TestClient(api_module.app) as c:
        yield c


@pytest.fixture
def has_model(client):
    if client.app.state.classifier is None:
        pytest.skip("models/final belum ada (jalankan python src/train_final.py)")


def make_customer(client, name="Rani"):
    r = client.post("/customers", json={"name": name, "phone": "0812", "channel": "whatsapp"})
    assert r.status_code == 201
    return r.json()


def test_health_dan_options(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["gemini_configured"] is False
    o = client.get("/options").json()
    assert o["statuses"] == ["New", "Design/Preparation", "Production", "Completed"]
    assert "Custom Sajadah" in o["order_types"]


def test_classify(client, has_model):
    r = client.post("/classify", json={"text": BRIEF})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Custom Sajadah"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-2
    assert client.post("/classify", json={"text": ""}).status_code == 422


def test_customer_crud(client):
    c = make_customer(client)
    assert c["channel"] == "WhatsApp"                      # dinormalisasi
    assert client.get(f"/customers/{c['id']}").json()["name"] == "Rani"
    r = client.put(f"/customers/{c['id']}", json={"name": "Rani A."})
    assert r.json()["name"] == "Rani A."
    assert len(client.get("/customers", params={"q": "rani"}).json()) == 1
    assert client.get("/customers/999").status_code == 404
    assert client.delete(f"/customers/{c['id']}").status_code == 204


def test_hapus_customer_dengan_order_ditolak(client):
    c = make_customer(client)
    client.post("/orders", json={"customer_id": c["id"], "raw_message": "tes", "auto_classify": False})
    assert client.delete(f"/customers/{c['id']}").status_code == 409


def test_produk(client):
    prods = client.get("/products").json()
    assert len(prods) >= 8
    r = client.post("/products", json={"sku": "SJD-POL", "name": "dup", "category": "sajadah"})
    assert r.status_code == 409
    r = client.post("/products", json={"sku": "NEW-1", "name": "Baru", "category": "handuk", "price_idr": 50000})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert client.put(f"/products/{pid}", json={"active": False}).json()["active"] is False
    assert all(p["id"] != pid for p in client.get("/products").json())


def test_alur_order_end_to_end(client, has_model):
    c = make_customer(client)
    r = client.post("/orders", json={"customer_id": c["id"], "raw_message": BRIEF, "confirm_prediction": True})
    assert r.status_code == 201
    o = r.json()
    assert o["status"] == "New" and o["order_code"].startswith("FRZ-")
    assert o["predicted_type"] == "Custom Sajadah" and o["order_type"] == "Custom Sajadah"
    oid = o["id"]

    r = client.post(f"/orders/{oid}/items", json={"product_id": 1, "quantity": 20, "color": "navy",
                                                  "custom_text": "Budi & Sarah", "font": "script", "layout": "tengah"})
    assert r.status_code == 201 and r.json()["font"] == "Script" and r.json()["layout"] == "Tengah"

    s = client.post(f"/orders/{oid}/summary").json()                 # tanpa key -> fallback
    assert s["sumber"] == "fallback" and s["jenis_order"] == "Custom Sajadah"
    assert s["items"][0]["jumlah"] == 20 and s["items"][0]["teks_custom"] == "Budi & Sarah"

    g = client.post(f"/orders/{oid}/greeting", json={"tone": "hangat"}).json()
    assert g["sumber"] == "fallback" and "Budi & Sarah" in g["kartu_ucapan"]
    assert client.get(f"/orders/{oid}").json()["greeting_card"] == g["kartu_ucapan"]

    r = client.put(f"/orders/{oid}", json={"greeting_card": "Edit admin", "notes": "tes"})
    assert r.json()["greeting_card"] == "Edit admin" and r.json()["notes"] == "tes"

    bad = client.post(f"/orders/{oid}/status", json={"status": "Completed"})
    assert bad.status_code == 409 and bad.json()["detail"]["transisi_diizinkan"] == ["Design/Preparation"]
    for st in ["Design/Preparation", "Production", "Completed"]:
        assert client.post(f"/orders/{oid}/status", json={"status": st, "note": "ok"}).json()["status"] == st
    hist = client.get(f"/orders/{oid}/history").json()
    assert [h["to_status"] for h in hist] == ["New", "Design/Preparation", "Production", "Completed"]
    assert len(client.get("/orders", params={"status": "Completed"}).json()) == 1

    assert client.delete(f"/orders/{oid}").status_code == 204
    assert client.get(f"/orders/{oid}").status_code == 404


def test_validasi(client):
    c = make_customer(client)
    o = client.post("/orders", json={"customer_id": c["id"], "auto_classify": False}).json()
    assert client.post(f"/orders/{o['id']}/items", json={"quantity": 0}).status_code == 422
    assert client.post(f"/orders/{o['id']}/items", json={"font": "Comic Sans"}).status_code == 422
    assert client.post(f"/orders/{o['id']}/items", json={"product_id": 9999}).status_code == 422
    assert client.post("/orders", json={"customer_id": 9999}).status_code == 404
    assert client.post(f"/orders/{o['id']}/status", json={"status": "Selesai"}).status_code == 422


def test_gemini_grounding_dan_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRAZA_DB", str(tmp_path / "g.db"))
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    halusinasi = {"ringkasan": "Pesanan sajadah.", "items": [{"produk": "sajadah", "jumlah": 99, "warna": "navy",
                  "teks_custom": "Andi", "font": None, "layout": None}], "acara": "nikahan", "tenggat": None,
                  "perlu_klarifikasi": []}
    monkeypatch.setattr(GenAIService, "_call_summary", lambda self, p: json.dumps(halusinasi))
    monkeypatch.setattr(GenAIService, "_call_greeting", lambda self, p: "Selamat menempuh hidup baru.")
    with TestClient(api_module.app) as c:
        cid = c.post("/customers", json={"name": "T"}).json()["id"]
        oid = c.post("/orders", json={"customer_id": cid, "raw_message": BRIEF, "auto_classify": False,
                                      "order_type": "Custom Sajadah"}).json()["id"]
        s = c.post(f"/orders/{oid}/summary").json()
        assert s["sumber"] == "gemini"
        assert s["items"][0]["teks_custom"] is None and s["items"][0]["jumlah"] is None   # tidak ada di pesan -> dikosongkan
        assert len(s["peringatan"]) == 2
        g = c.post(f"/orders/{oid}/greeting").json()
        assert g["sumber"] == "gemini"

        def boom(self, p):
            raise RuntimeError("kuota habis")
        monkeypatch.setattr(GenAIService, "_call_summary", boom)
        s2 = c.post(f"/orders/{oid}/summary", params={"force": True}).json()
        assert s2["sumber"] == "fallback" and "Gemini gagal" in s2["peringatan"][0]
        assert s2["items"][0]["jumlah"] == 20


def test_response_text_abaikan_bagian_non_teks():
    from types import SimpleNamespace as NS
    from genai_service import response_text
    parts = [NS(text=None, thought=False), NS(text="isi jawaban", thought=False), NS(text="pikiran internal", thought=True)]
    resp = NS(candidates=[NS(content=NS(parts=parts))], text="cadangan")
    assert response_text(resp) == "isi jawaban"
    assert response_text(NS(candidates=[], text="cadangan")) == "cadangan"


def test_pelanggan_buat_dan_lacak(client):
    r = client.post("/public/orders", json={"name": "Sinta", "phone": "0812-3456-7890", "message": BRIEF,
                    "items": [{"quantity": 20, "color": "navy", "custom_text": "Budi & Sarah", "font": "script", "layout": "tengah"}]})
    assert r.status_code == 201
    code = r.json()["order_code"]
    assert r.json()["status"] == "New"
    o = client.get("/orders").json()[0]
    assert o["channel"] == "Website" and o["order_type"] is None          # jenis final menunggu konfirmasi admin
    if client.app.state.classifier is not None:
        assert o["predicted_type"] == "Custom Sajadah"

    # format nomor berbeda tetap cocok; kode tidak peka huruf besar/kecil
    t1 = client.get("/public/track", params={"order_code": code.lower(), "phone": "+62 812 3456 7890"})
    assert t1.status_code == 200
    body = t1.json()
    assert body["items"][0]["font"] == "Script" and body["jenis_pesanan"] is None and len(body["riwayat"]) == 1
    assert "raw_message" not in body and "ai_summary" not in body and "notes" not in body    # tidak membocorkan data internal

    client.put(f"/orders/{o['id']}", json={"order_type": "Custom Sajadah"})
    client.post(f"/orders/{o['id']}/status", json={"status": "Design/Preparation"})
    t2 = client.get("/public/track", params={"order_code": code, "phone": "081234567890"}).json()
    assert t2["jenis_pesanan"] == "Custom Sajadah" and t2["status"] == "Design/Preparation" and len(t2["riwayat"]) == 2

    # nomor salah atau kode salah: pesan sama (tidak membocorkan mana yang salah)
    for params in ({"order_code": code, "phone": "081200000000"}, {"order_code": "FRZ-9999", "phone": "081234567890"}):
        assert client.get("/public/track", params=params).status_code == 404

    # nomor yang sama memesan lagi: memakai customer yang sama
    client.post("/public/orders", json={"name": "Sinta R", "phone": "+6281234567890", "message": "handuk bordir nama Dinda 2 pcs"})
    assert len(client.get("/customers").json()) == 1 and len(client.get("/orders").json()) == 2


def test_pelanggan_validasi(client):
    ok = {"name": "Sinta", "phone": "081234567890", "message": BRIEF}
    assert client.post("/public/orders", json={**ok, "phone": "123"}).status_code == 422
    assert client.post("/public/orders", json={**ok, "message": "hai"}).status_code == 422
    assert client.post("/public/orders", json={**ok, "items": [{"font": "Comic Sans"}]}).status_code == 422
    assert client.post("/public/orders", json={**ok, "items": [{"product_id": 9999}]}).status_code == 422
    assert client.post("/public/orders", json={**ok, "items": [{"quantity": 1}] * 6}).status_code == 422