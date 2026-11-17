# Notebook Penjelasan

Enam notebook ini menjelaskan **apa yang dilakukan proyek, mengapa, dan bagaimana hasilnya**, dengan kode yang bisa dijalankan ulang (*Run All*). Skrip `.py` di `src/` tetap menjadi implementasi yang dijalankan lewat terminal; notebook memakai skrip itu langsung (`import`), jadi tidak ada logika yang berbeda.

| # | Notebook | Isi | Perkiraan waktu |
|---|---|---|---|
| 1 | `01_data_dan_label.ipynb` | Masalah bisnis, 5 label dan aturannya, data sintetis, mengapa split per template | < 1 menit |
| 2 | `02_preprocessing.ipynb` | Tiap langkah preprocessing dengan contoh, penanda negasi, ablation | 1-2 menit |
| 3 | `03_model.ipynb` | TF-IDF + Logistic Regression, CNN 1D, BiLSTM, early stopping, **mengapa CNN** | 2-4 menit |
| 4 | `04_ensemble_dan_evaluasi.ipynb` | Ensemble, protokol evaluasi, validasi silang per template, hasil test blind, analisis error, keterbatasan | beberapa menit |
| 5 | `05_inferensi_dan_genai.ipynb` | Memuat model beku, `needs_review`, Gemini dengan pengaman (grounding check, fallback) | < 1 menit |
| 6 | `06_rag.ipynb` | RAG Tanya Firaza: basis pengetahuan, embedding bge-m3, FAISS, retrieval, jawaban lokal Qwen2.5, menolak menjawab, evaluasi hit@k. **Butuh Ollama berjalan** | 1-3 menit |

## Cara menjalankan
1. Aktifkan venv dan pasang dependensi: `pip install -r requirements.txt` (termasuk `ipykernel`).
2. Buka folder proyek di VS Code, buka notebook, pilih kernel `.venv`, lalu **Run All**.
3. Notebook 05 membutuhkan `models/final/` (`python src/train_final.py`). Bagian Gemini memakai `GEMINI_API_KEY` dari `.env` bila ada; jika tidak, otomatis memakai fallback (lihat kolom `sumber` pada keluaran).

## Catatan
- Notebook yang tersimpan di repo berisi keluaran hasil eksekusi. Angka dapat bergeser sekitar 0,01 di mesin lain karena perbedaan numerik; jalankan ulang untuk memperbarui keluaran dengan mesinmu.
- **Test blind tidak dievaluasi di notebook.** Evaluasi blind hanya lewat `src/evaluate_blind.py` (sekali pakai, dikunci file lock); notebook 04 hanya menampilkan hasilnya.
- Pembaca yang ingin urutan singkat: 01 → 03 → 04.