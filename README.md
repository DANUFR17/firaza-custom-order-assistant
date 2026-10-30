# Firaza Custom Order Assistant

Capstone project Dicoding Bootcamp Batch 15 (individu, learning path **Generative AI**).
Use Case 2: **AI-Assisted Custom Order Management** untuk Firaza (retail hampers dan personalized gifts).

Sistem membantu admin mengubah pesan customer menjadi custom order yang terstruktur dan dapat dilacak:
klasifikasi jenis order oleh model yang dilatih sendiri, ringkasan order dan kartu ucapan lewat Generative AI,
serta pelacakan status New → Design/Preparation → Production → Completed.

## Status (progres sekitar 51%)

| Bagian | Status |
|---|---|
| Label guideline, dataset sintetis, preprocessing | Selesai |
| Model klasifikasi (baseline, CNN/BiLSTM Keras, ensemble final) | Model final dibekukan; evaluasi blind menunggu dijalankan (satu kali) |
| Skema database (SQLite + SQLAlchemy) | Selesai |
| Inference API (FastAPI) + fitur Gemini | Berikutnya |
| Web admin (Streamlit) | Berikutnya |
| Dokumentasi akhir, sitasi, demo | Berjalan |
| RAG (opsional) | Diputuskan belakangan |

## Struktur

```
data/        dataset_sintetis_v3.csv, manual_test.csv, manual_test_blind.csv
docs/        label_guideline.md, db_schema.md, (project plan)
src/         generate_dataset.py, preprocessing.py, common.py, train_baseline.py, train_keras.py,
             compare_models.py, train_final.py, predictor.py, evaluate_blind.py, database.py
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
```

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
Hasil test blind: *diisi setelah `evaluate_blind.py` dijalankan*.

## Keterbatasan

- Data sepenuhnya sintetis; performa pada pesan customer nyata Firaza belum teruji.
- Jumlah template terbatas, sehingga skor pada satu split val/test bergoyang; gunakan GroupKFold sebagai acuan.
- Ambang `needs_review` (0,6) belum dikalibrasi.

## Sitasi

Lihat [`CITATIONS.md`](CITATIONS.md).