# Dokumentasi Preprocessing dan Model

Semua model dilatih **dari nol** pada data sintetis proyek ini. Tidak memakai TensorFlow Hub atau model pra-latih, tidak memakai AutoML, dan tidak berasal dari fork repositori AI lain.

## 1. Data

- **Sumber:** sintetis, dibuat oleh `src/generate_dataset.py` (template dengan slot untuk produk, warna, jumlah, nama, acara, font, layout, dan sebagainya). Data asli Firaza tidak tersedia.
- **Versi aktif:** `data/dataset_sintetis_v3.csv`, 3.000 pesan, 600 per kelas, tanpa duplikat.
- **Kelas (5):** Ready Stock, Custom Sajadah, Custom Handuk, Hampers, Other. Definisi dan 11 aturan kasus perbatasan ada di `docs/label_guideline.md`.
- **Gaya bahasa:** formal (sekitar 40%), santai (sekitar 40%), noisy dengan singkatan, typo, dan tanda baca hilang (sekitar 20%); sebagian pesan berisi kalimat tambahan netral.
- **Riwayat versi:** v1 (18 template per kelas) → v2 (negasi, komplain yang menyebut produk, pesan pendek, ID template unik) → v3 (kosakata dan parafrase lebih beragam).
- **Test set manual** (`data/manual_test.csv`, 100 pesan, 20 per kelas) dan **test set blind** (`data/manual_test_blind.csv`, 30 pesan, 6 per kelas) ditulis sendiri oleh penulis proyek. Test manual dipakai untuk analisis error sehingga berstatus data pengembangan; test blind dibuka sekali setelah model dibekukan.

### Split per template (grouped split)
Split dilakukan per **template**, bukan per baris, sehingga template di val/test tidak pernah dilihat saat training. Pada data sintetis, random split menghasilkan skor menyesatkan (macro F1 baseline 1,000 pada v1 vs sekitar 0,8 pada grouped split).

## 2. Preprocessing (`src/preprocessing.py`)

| Langkah | Contoh |
|---|---|
| Case folding, hapus emoji dan URL | `Halo 🙏` → `halo` |
| Masking | teks dalam tanda kutip → `NAMA`; `70x140` → `DIM`; `50rb`, `Rp200.000` → `UANG`; angka lain → `NUM` |
| Normalisasi slang/singkatan | `sjd` → `sajadah`, `brp` → `berapa`, `ga/gak/nggak` → `tidak`, `biji/buah` → `pcs` |
| Koreksi typo terbatas | fuzzy match hanya terhadap kamus kata domain (cutoff 0,84), agar kata umum tidak rusak |
| Penanda negasi | kata (maksimal 3) setelah *tidak/bukan/tanpa/belum/jangan* diberi awalan `NEG_` sampai tanda baca atau kata penghubung |
| Hapus tanda baca | |

**Stopword sengaja tidak dibuang**, karena kata seperti *tidak* dan *tanpa* menentukan niat ("tidak perlu bordir" berbeda dari "mau bordir").

### Ablation (macro F1, baseline TF-IDF + linear, data v3)

| Varian | Val | Test (grouped) | Manual | GroupKFold (5) |
|---|---|---|---|---|
| A. Mentah (tanpa preprocessing) | 0,857 | 0,840 | 0,869 | 0,869 ± 0,048 |
| B. + slang + masking | 0,887 | 0,830 | 0,910 | 0,853 ± 0,049 |
| C. + koreksi typo | 0,894 | 0,810 | 0,910 | 0,864 ± 0,051 |
| D. + negasi (pipeline penuh) | 0,907 | 0,805 | 0,930 | 0,863 ± 0,036 |

**Pembacaan jujur:** preprocessing menaikkan skor di val dan manual (pesan bergaya manusia: +0,05 sampai +0,06), tetapi tidak menaikkan, bahkan sedikit menurunkan, skor di test sintetis, dan GroupKFold relatif datar (selisih masih di dalam standar deviasi). Jadi manfaatnya nyata untuk pesan informal, tetapi tidak konsisten pada template baru. Pipeline penuh tetap dipakai karena kasus nyata lebih mirip pesan informal.

## 3. Model

| Model | Keterangan |
|---|---|
| **Baseline** | TF-IDF (1-2 gram, sublinear tf, min_df=2) + model linear. `train_baseline.py` memilih lewat val: LinearSVC (C=3) |
| **CNN 1D (Keras)** | TextVectorization (maks. 4.000 token, panjang 40) → Embedding 64 → SpatialDropout → Conv1D kernel 2/3/4 (64 filter) + GlobalMaxPool → Dropout → Dense 64 → Softmax 5 kelas. Adam 1e-3, early stopping pada val_loss |
| **BiLSTM (Keras)** | Embedding 64 → Bidirectional LSTM 48 → Dense 64 → Softmax; dipakai sebagai pembanding |
| **Ensemble final** | 0,5 × rata-rata probabilitas CNN (3 seed, 10 epoch) + 0,5 × Logistic Regression (C=1, dipilih agar menghasilkan probabilitas) |

## 4. Protokol evaluasi

- Pemilihan model/hyperparameter memakai split `val` dan GroupKFold; `test`, `manual`, dan `blind` hanya dilaporkan.
- Karena setiap split hanya berisi beberapa template per kelas, skor satu split bergoyang kuat (CNN: val 0,99 tetapi test 0,79). **GroupKFold per template (5 lipatan)** dipakai sebagai acuan utama.
- Model final dilatih ulang pada seluruh data generator, lalu dievaluasi **sekali** pada test blind (`evaluate_blind.py` dengan file lock).

## 5. Hasil (macro F1)

**Perbandingan kandidat (satu pembagian data):**

| Model | Val | Test | Manual |
|---|---|---|---|
| TF-IDF + LogReg | 0,906 | 0,805 | 0,931 |
| CNN (rata-rata 3 seed) | 0,987 | 0,790 | 0,858 |
| CNN + augmentasi token | 0,954 | 0,725 | 0,899 |
| Ensemble CNN + LogReg | 0,987 | 0,794 | 0,880 |

**GroupKFold per template (acuan utama):**

| Model | Rata-rata ± std |
|---|---|
| TF-IDF + LogReg | 0,853 ± 0,040 |
| CNN | 0,855 ± 0,066 |
| CNN + augmentasi | 0,860 ± 0,065 |
| **Ensemble CNN + LogReg** | **0,878 ± 0,057** |
| Ensemble CNN + augmentasi + LogReg | 0,864 ± 0,070 |

Selisih antar model masih di dalam standar deviasi, jadi tidak ada klaim bahwa satu model jauh lebih baik. Ensemble dipilih karena tertinggi di GroupKFold dan tidak pernah terburuk. Augmentasi token tidak dipakai karena menurunkan skor di test.

**Test blind (30 pesan, satu kali):** ensemble final **accuracy 0,90, macro F1 0,902**. Sebagai informasi (bukan untuk memilih model): CNN saja 0,873, LogReg saja 0,902.

| Kelas | Precision | Recall | F1 |
|---|---|---|---|
| Ready Stock | 1,00 | 0,83 | 0,91 |
| Custom Sajadah | 1,00 | 1,00 | 1,00 |
| Custom Handuk | 1,00 | 0,83 | 0,91 |
| Hampers | 0,86 | 1,00 | 0,92 |
| Other | 0,71 | 0,83 | 0,77 |

Target di project plan (macro F1 minimal 80%, target 85%) terpenuhi pada GroupKFold dan test blind.

### Analisis error (3 dari 30 pesan blind)
1. "min box kado doang jual ga?" (Ready Stock) diprediksi Other, confidence 0,42. Bahasa sangat santai dengan kata yang jarang muncul di data latih.
2. "Handuk polosnya sudah saya beli di toko lain. Apakah di sini bisa jasa bordir inisial saja?" (Custom Handuk) diprediksi Other, confidence 0,49. Kasus jasa bordir untuk barang bawaan belum punya template di generator.
3. "Paket hampers yang datang kurang satu mug, padahal handuk dan kartu ucapannya lengkap. Bagaimana proses komplainnya?" (Other) diprediksi Hampers, confidence 0,86. Komplain yang menyebut banyak isi hampers; satu-satunya error yang yakin.

Dua dari tiga error berconfidence di bawah 0,6 sehingga akan ditandai `needs_review`; pada 25 pesan yang tidak ditandai, 24 benar. Sampelnya terlalu kecil untuk disimpulkan. Kelas Other menjadi "penampung" bagi pesan informal atau tidak lazim.

## 6. Keterbatasan
- Data sintetis dengan jumlah template terbatas; model belum teruji pada pesan nyata Firaza dan tidak punya embedding pra-latih, sehingga kosakata baru sulit dikenali.
- Test blind hanya 30 pesan: satu kesalahan menggeser skor sekitar 3 poin.
- Karena error blind sudah dilihat, memperbaiki data berdasarkan error tersebut berarti membutuhkan test blind baru.
- Ambang `needs_review` (0,6) belum dikalibrasi.

## 7. Reproduksi
```powershell
python src/generate_dataset.py --per-label 600 --seed 42 --out data/dataset_sintetis_v3.csv
python src/train_baseline.py --ablation
python src/train_keras.py
python src/compare_models.py        # dan: --cv
python src/train_final.py
python src/evaluate_blind.py        # sekali saja
```
Angka dapat bergeser sekitar 0,01 antar mesin karena perbedaan numerik floating-point.
