"""
Uji web Streamlit tanpa server: AppTest menjalankan app/streamlit_app.py, dan panggilan HTTP ke API
dialihkan ke FastAPI TestClient (database sementara). Jalankan dari ROOT proyek:  pytest -q
Uji yang butuh model klasifikasi dilewati bila models/final belum ada.
"""
import os

import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import api as api_module

APP_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "streamlit_app.py"))
BASE = "http://127.0.0.1:8000"
BRIEF = "Mau sajadah navy 20 pcs untuk nikahan, custom nama Budi & Sarah"
PAGES = ["Dashboard", "Order Baru", "Daftar Order", "Customer", "Produk"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRAZA_DB", str(tmp_path / "web.db"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("API_URL", BASE)
    with TestClient(api_module.app) as client:
        def fake_request(method, url, **kw):
            kw.pop("timeout", None)
            return client.request(method, url.replace(BASE, ""), **kw)

        monkeypatch.setattr(requests, "request", fake_request)
        monkeypatch.setattr(requests, "get", lambda url, **kw: fake_request("GET", url, **kw))
        at = AppTest.from_file(APP_FILE, default_timeout=60)
        at.client = client
        yield at


def goto(at, page):
    at.sidebar.radio(key="menu").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_semua_halaman_tampil(app):
    app.run()
    assert not app.exception
    for p in PAGES:
        goto(app, p)


def test_alur_order_di_web(app):
    app.run()
    if app.client.app.state.classifier is None:
        pytest.skip("models/final belum ada")
    goto(app, "Order Baru")
    app.radio(key="cust_mode").set_value("Customer baru").run()
    app.text_input(key="nc_name").set_value("Rani")
    app.text_area(key="raw_msg").set_value(BRIEF).run()
    app.button(key="btn_classify").click().run()
    assert not app.exception
    assert app.metric[0].value == "Custom Sajadah"
    app.button(key="btn_save_order").click().run()
    assert not app.exception
    assert "FRZ-0001" in app.success[0].value

    # order tersimpan lewat API dengan jenis final = pilihan di form (default = usulan AI)
    o = app.client.get("/orders").json()[0]
    assert o["order_type"] == "Custom Sajadah" and o["predicted_type"] == "Custom Sajadah"
    app.client.post(f"/orders/{o['id']}/items", json={"product_id": 1, "quantity": 20, "color": "navy",
                                                       "custom_text": "Budi & Sarah", "font": "Script", "layout": "Tengah"})

    goto(app, "Daftar Order")
    assert any("Pratinjau kasar" in c.value for c in app.caption)
    app.button(key="btn_sum").click().run()
    assert not app.exception
    assert any("Fallback" in m.value for m in app.markdown)
    app.button(key="btn_greet").click().run()
    assert not app.exception
    assert "Budi & Sarah" in app.text_area[0].value
    app.button(key="btn_status_Design/Preparation").click().run()
    assert not app.exception
    assert "Design/Preparation" in app.subheader[0].value
    assert app.client.get(f"/orders/{o['id']}").json()["status"] == "Design/Preparation"
