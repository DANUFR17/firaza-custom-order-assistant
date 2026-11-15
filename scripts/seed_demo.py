"""
seed_demo.py - Isi data DEMO (customer dan order contoh) lewat REST API, supaya dashboard tidak kosong saat presentasi.

Semua data fiktif. Jalankan dari ROOT proyek setelah API berjalan:
    python scripts/seed_demo.py
    python scripts/seed_demo.py --force     # tetap isi walau sudah ada order
Mengubah alamat API: set API_URL (default http://127.0.0.1:8000).

Hasil: 4 customer dan 5 order di status berbeda (New, Design/Preparation, Production, Completed),
termasuk satu order dengan confidence rendah ("halo kak", memicu needs_review) yang belum dikonfirmasi admin.
Ringkasan AI dan kartu ucapan dibuat untuk dua order (Gemini bila aktif, fallback bila tidak).
"""
import argparse
import os
import sys

import requests

API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


def call(method, path, **kw):
    r = requests.request(method, API + path, timeout=90, **kw)
    if r.status_code >= 400:
        sys.exit(f"Gagal {method} {path}: {r.status_code} {r.text[:300]}")
    return r.json() if r.status_code != 204 and r.content else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    try:
        health = call("GET", "/health")
    except requests.exceptions.RequestException:
        sys.exit(f"API tidak dapat dihubungi di {API}. Jalankan dulu: uvicorn api:app --app-dir src --reload")
    if not health["classifier_loaded"]:
        sys.exit("Model klasifikasi belum dimuat. Jalankan: python src/train_final.py, lalu restart API.")
    if call("GET", "/orders", params={"limit": 1}) and not args.force:
        sys.exit("Sudah ada order di database. Pakai --force untuk menambah data demo, atau hapus data/firaza.db dulu.")

    prods = {p["name"]: p["id"] for p in call("GET", "/products")}
    cust = {}
    for name, phone, ch in [("Rani Wulandari", "081200000001", "WhatsApp"), ("Dimas Pratama", "081200000002", "Shopee"),
                            ("Nabila Putri", "081200000003", "TikTok Shop"), ("PT Maju Jaya (pesanan partai)", "081200000004", "WhatsApp")]:
        cust[name] = call("POST", "/customers", json={"name": name, "phone": phone, "channel": ch})["id"]

    def order(customer, msg, channel, items, to_status=(), confirm=True, ai=False):
        o = call("POST", "/orders", json={"customer_id": cust[customer], "raw_message": msg, "channel": channel,
                                          "confirm_prediction": confirm, "items": items})
        for st in to_status:
            call("POST", f"/orders/{o['id']}/status", json={"status": st, "note": "data demo"})
        if ai:
            call("POST", f"/orders/{o['id']}/summary")
            call("POST", f"/orders/{o['id']}/greeting", json={"tone": "hangat"})
        o = call("GET", f"/orders/{o['id']}")
        print(f"{o['order_code']}  {o['status']:<19} {str(o['order_type']):<15} (usulan AI: {o['predicted_type']}, "
              f"{o['predicted_confidence']:.0%})  {customer}")
        return o

    order("Rani Wulandari", "Mau sajadah navy 20 pcs untuk nikahan, custom nama Budi & Sarah", "WhatsApp",
          [{"product_id": prods["Sajadah Polos"], "quantity": 20, "color": "navy", "custom_text": "Budi & Sarah",
            "font": "Script", "layout": "Tengah"}],
          to_status=["Design/Preparation", "Production"], ai=True)
    order("Dimas Pratama", "handuk bordir nama Dinda 2 pcs warna putih", "Shopee",
          [{"product_id": prods["Handuk Mandi"], "quantity": 2, "color": "putih", "custom_text": "Dinda",
            "font": "Elegan", "layout": "Pojok Kanan Bawah"}],
          to_status=["Design/Preparation"])
    order("Nabila Putri", "kak sajadah polos navy ready ga? mau 1 aja", "TikTok Shop",
          [{"product_id": prods["Sajadah Polos"], "quantity": 1, "color": "navy"}])
    order("PT Maju Jaya (pesanan partai)", "mau hampers lebaran isi handuk + sajadah 25 box, kartu ucapan nama keluarga", "WhatsApp",
          [{"product_id": prods["Paket Hampers Handuk + Sajadah"], "quantity": 25, "custom_text": "Keluarga Besar Maju Jaya"}],
          to_status=["Design/Preparation", "Production", "Completed"], ai=True)
    order("Nabila Putri", "halo kak", "TikTok Shop", [], confirm=False)   # confidence rendah (needs_review), belum dikonfirmasi
    print("\nSelesai. Buka web: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
