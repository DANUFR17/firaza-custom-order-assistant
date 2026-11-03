# Firaza Custom Order Assistant

Capstone project Dicoding Bootcamp Batch 15 (individu, learning path **Generative AI**).
Use Case 2: **AI-Assisted Custom Order Management** untuk Firaza (retail hampers dan personalized gifts).

Sistem membantu admin mengubah pesan customer menjadi custom order yang terstruktur dan dapat dilacak:
klasifikasi jenis order oleh model yang dilatih sendiri, ringkasan order dan kartu ucapan lewat Generative AI,
serta pelacakan status New → Design/Preparation → Production → Completed.

## Status

| Bagian | Status |
|---|---|
| Label guideline, dataset sintetis, preprocessing | Selesai |
| Model klasifikasi (baseline, CNN/BiLSTM Keras, ensemble final) | Selesai: model final dibekukan, evaluasi blind dijalankan satu kali |
| Skema database (SQLite + SQLAlchemy) | Selesai |
| Inference API (FastAPI) + CRUD + status order | Selesai, teruji otomatis (`pytest`) |
| Fitur Gemini (structured summary, kartu ucapan) | Kode dan fallback selesai; uji dengan key Gemini sungguhan menunggu |
| Web admin (Streamlit) | Selesai (dashboard, order baru, daftar order, customer, produk) |
| Dokumentasi akhir, sitasi, demo | Berjalan |
| RAG (opsional) | Diputuskan belakangan |

## Struktur

```
data/        dataset_sintetis_v3.csv, manual_test.csv, manual_test_blind.csv
docs/        label_guideline.md, db_schema.md, (project plan)
src/         generate_dataset.py, preprocessing.py, common.py, train_baseline.py, train_keras.py,
             compare_models.py, train_final.py, predictor.py, evaluate_blind.py, database.py,
             schemas.py, genai_service.py, api.py
app/         streamlit_app.py (web admin; memanggil API lewat HTTP)
tests/       test_api.py, test_web.py (pytest)
reports/     metrik, confusion matrix, daftar error (dibuat skrip)
models/      baseline_tfidf.joblib, keras_cnn.keras, final/ (model beku)
```

## Cara menjalankan (dari root folder)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/generate_dataset.py --per-label 600 --seed 42 --out data/dataset_sintetis_v3.csv
python src/train_baseline.py --ablation
python src/train_keras.py
python src/compare_models.py          # dan: --cv
python src/train_final.py             # membekukan model final ke models/final/
python src/evaluate_blind.py          # SEKALI SAJA (dilindungi file lock)
python src/database.py --db data/firaza.db --seed

# Menjalankan aplikasi (dua terminal, venv aktif)
uvicorn api:app --app-dir src --reload        # API: http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py            # Web: http://localhost:8501

pytest -q                                     # uji otomatis API dan web
```

Fitur Gemini membutuhkan `GEMINI_API_KEY` di `.env` (lihat `.env.example`); tanpa key, ringkasan dan kartu ucapan memakai fallback berbasis aturan.
Dokumentasi: `docs/api.md`, `docs/prompts.md`, `docs/db_schema.md`, `docs/label_guideline.md`.

## Pendekatan model

- **Data:** sintetis berbahasa Indonesia (template + slot, 3 gaya bahasa, 5 label seimbang), karena data asli Firaza tidak tersedia.
  Cara pembuatan didokumentasikan di `src/generate_dataset.py` dan `docs/label_guideline.md`.
- **Preprocessing:** normalisasi slang/singkatan, koreksi typo terbatas, masking angka/nama, dan penanda negasi.
  Stopword sengaja tidak dibuang karena kata negasi menentukan niat order.
- **Model (dilatih dari nol, tanpa model pra-latih, TF Hub, atau AutoML):** TF-IDF + Logistic Regression,
  CNN 1D (Keras), dan BiLSTM. Model final adalah **ensemble CNN + Logistic Regression** (bobot 0,5/0,5).
- **Evaluasi:** split per template (grouped), GroupKFold 5 lipatan, test set manual (ditulis sendiri, sudah dipakai untuk
  analisis error sehingga berstatus data pengembangan), dan test set blind (dibuka sekali setelah model dibekukan).

## Hasil (macro F1)

| Model | GroupKFold (5 lipatan, per template) |
|---|---|
| TF-IDF + Logistic Regression | 0,853 ± 0,040 |
| CNN | 0,855 ± 0,066 |
| Ensemble CNN + Logistic Regression | 0,878 ± 0,057 |

Selisih antar model masih berada dalam standar deviasi, jadi tidak ada klaim bahwa satu model jauh lebih baik.
### Test blind (30 pesan, dibuka satu kali setelah model dibekukan)

Model final (ensemble CNN + Logistic Regression): **accuracy 0,90, macro F1 0,902** (3 salah dari 30).
Skor ini bersumber dari 30 pesan (6 per kelas), sehingga satu kesalahan menggeser skor sekitar 3 poin; baca sebagai
perkiraan kasar, bukan angka presisi. Detail ada di `reports/blind_metrics.json` dan `reports/blind_errors.csv`;
`reports/blind_evaluation.lock.json` menandakan evaluasi dilakukan sekali.

| Kelas | Precision | Recall | F1 |
|---|---|---|---|
| Ready Stock | 1,00 | 0,83 | 0,91 |
| Custom Sajadah | 1,00 | 1,00 | 1,00 |
| Custom Handuk | 1,00 | 0,83 | 0,91 |
| Hampers | 0,86 | 1,00 | 0,92 |
| Other | 0,71 | 0,83 | 0,77 |

**Analisis error (3 pesan):**
1. *"min box kado doang jual ga?"* (Ready Stock) diprediksi Other, confidence 0,42. Bahasa sangat santai dengan kata yang jarang muncul di data latih.
2. *"Handuk polosnya sudah saya beli di toko lain. Apakah di sini bisa jasa bordir inisial saja?"* (Custom Handuk) diprediksi Other, confidence 0,49. Kasus jasa bordir untuk barang bawaan sendiri belum punya template di generator.
3. *"Paket hampers yang datang kurang satu mug, padahal handuk dan kartu ucapannya lengkap. Bagaimana proses komplainnya?"* (Other) diprediksi Hampers, confidence 0,86. Komplain yang menyebut banyak isi hampers; ini satu-satunya error yang yakin.

Dua dari tiga error berconfidence di bawah 0,6 sehingga akan ditandai `needs_review` (5 dari 30 pesan ditandai; 3 di antaranya benar). Pada 25 pesan yang tidak ditandai, 24 benar. Pola ini searah dengan tujuan flag, tetapi sampelnya terlalu kecil untuk disimpulkan.
Kelas Other menjadi "penampung" bagi pesan informal atau tidak lazim, sehingga precision-nya paling rendah.

Karena error blind sudah dilihat, perbaikan data berdasarkan error ini akan membuat test blind tidak lagi buta; evaluasi ulang membutuhkan test blind baru.

## Keterbatasan

- Data sepenuhnya sintetis; performa pada pesan customer nyata Firaza belum teruji.
- Jumlah template terbatas, sehingga skor pada satu split val/test bergoyang; gunakan GroupKFold sebagai acuan.
- Ambang `needs_review` (0,6) belum dikalibrasi.

## Sitasi

Lihat [`CITATIONS.md`](CITATIONS.md).
