# Dokumentasi API

Base URL lokal: `http://127.0.0.1:8000`. Dokumentasi interaktif (Swagger UI) dibuat otomatis oleh FastAPI di **`/docs`** dan **`/redoc`**; spesifikasi OpenAPI di `/openapi.json`.
MVP ini belum memakai autentikasi (hanya untuk penggunaan lokal oleh admin).

## Menjalankan
```powershell
# dari root proyek, venv aktif
python src/train_final.py                 # sekali, membuat models/final/
uvicorn api:app --app-dir src --reload
```
Konfigurasi lewat `.env` (lihat `.env.example`): `GEMINI_API_KEY`, `GEMINI_MODEL`, `FIRAZA_DB`, `MODEL_DIR`, `CORS_ORIGINS`.
Tanpa `GEMINI_API_KEY`, fitur Generative AI tetap berjalan dengan fallback berbasis aturan (`"sumber": "fallback"`).

## Endpoint

| Metode | Path | Fungsi |
|---|---|---|
| GET | `/health` | Status API, model klasifikasi, dan konfigurasi Gemini |
| GET | `/options` | Pilihan tetap untuk dropdown: jenis order, status, kanal, font, layout, transisi status |
| **POST** | **`/classify`** | **Inference API**: klasifikasi pesan ke 5 jenis order (label, confidence, `needs_review`, probabilitas) |
| POST / GET | `/customers` | Tambah / daftar customer (`q`, `limit`, `offset`) |
| GET / PUT / DELETE | `/customers/{id}` | Detail / ubah / hapus (ditolak 409 bila punya order) |
| GET / POST | `/products` | Daftar produk / tambah produk (SKU unik) |
| GET / PUT | `/products/{id}` | Detail / ubah produk (mis. `active`) |
| POST | `/orders` | Buat order; bila `raw_message` ada dan model dimuat, otomatis diklasifikasikan. `confirm_prediction=true` memakai prediksi sebagai jenis final |
| GET | `/orders` | Daftar + filter `status`, `order_type`, `customer_id`, `q` |
| GET / PUT / DELETE | `/orders/{id}` | Detail / ubah (mis. konfirmasi `order_type`) / hapus |
| POST | `/orders/{id}/status` | Ubah status; transisi tidak valid ditolak 409 beserta daftar transisi yang diizinkan |
| GET | `/orders/{id}/history` | Riwayat perubahan status |
| POST | `/orders/{id}/items` | Tambah item (produk, jumlah, warna, `custom_text`, font, layout, vendor) |
| PUT / DELETE | `/orders/{id}/items/{item_id}` | Ubah / hapus item |
| POST | `/orders/{id}/summary` | **Generative AI**: structured order summary (`?force=true` untuk membuat ulang) |
| POST | `/orders/{id}/greeting` | **Generative AI**: draft kartu ucapan, body opsional `{"tone": "hangat\|formal\|islami"}` |

Alur status yang diizinkan: `New → Design/Preparation → Production → Completed` (revisi: `Production → Design/Preparation`).

## Contoh (PowerShell)
```powershell
# 1. klasifikasi
Invoke-RestMethod -Uri http://127.0.0.1:8000/classify -Method Post -ContentType "application/json" `
  -Body '{"text":"Mau sajadah navy 20 pcs untuk nikahan, custom nama Budi & Sarah"}'

# 2. customer lalu order (otomatis diklasifikasikan dan dikonfirmasi)
$c = Invoke-RestMethod -Uri http://127.0.0.1:8000/customers -Method Post -ContentType "application/json" -Body '{"name":"Rani","phone":"0812","channel":"WhatsApp"}'
$o = Invoke-RestMethod -Uri http://127.0.0.1:8000/orders -Method Post -ContentType "application/json" `
  -Body ('{"customer_id":' + $c.id + ',"raw_message":"Mau sajadah navy 20 pcs untuk nikahan, custom nama Budi & Sarah","confirm_prediction":true}')

# 3. item, ringkasan, kartu ucapan, status
Invoke-RestMethod -Uri "http://127.0.0.1:8000/orders/$($o.id)/items" -Method Post -ContentType "application/json" `
  -Body '{"product_id":1,"quantity":20,"color":"navy","custom_text":"Budi & Sarah","font":"Script","layout":"Tengah"}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/orders/$($o.id)/summary" -Method Post
Invoke-RestMethod -Uri "http://127.0.0.1:8000/orders/$($o.id)/greeting" -Method Post
Invoke-RestMethod -Uri "http://127.0.0.1:8000/orders/$($o.id)/status" -Method Post -ContentType "application/json" -Body '{"status":"Design/Preparation","note":"desain disetujui"}'
```

## Kode status HTTP
`201` dibuat, `204` dihapus, `404` tidak ditemukan, `409` konflik (transisi status tidak valid, SKU ganda, customer masih punya order), `422` validasi gagal (font/layout/kanal tidak dikenal, jumlah < 1), `503` model klasifikasi belum dimuat.

## Pengujian
`pytest -q` dari root proyek menjalankan 8 uji otomatis (CRUD, alur order end-to-end, validasi, grounding Gemini, dan fallback). Gemini tidak dipanggil sungguhan dalam uji.
