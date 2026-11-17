# RAG: Tanya Firaza (Side Quest)

Fitur **Tanya Firaza** menjawab pertanyaan pelanggan (cara memesan, produk custom, kartu ucapan, status pesanan, dan sebagainya) dari sebuah basis pengetahuan, dengan menyebut sumbernya. Fitur ini tampil sebagai menu **Tanya Firaza** di tampilan Pelanggan dan tersedia lewat `POST /rag/ask`.

## Komponen (sesuai ketentuan Side Quest)

| Komponen | Pilihan | Keterangan |
|---|---|---|
| Model embedding (open-source, Hugging Face) | **BAAI/bge-m3** | Multibahasa (termasuk Indonesia), dijalankan lokal lewat Ollama |
| Vector database | **FAISS** (`IndexFlatIP`) | Cosine similarity karena vektor dinormalisasi; cadangan numpy bila FAISS tidak terpasang |
| Model generation (SLM open-source, **lokal**) | **Qwen2.5-3B-Instruct** (kuantisasi 4-bit) lewat Ollama | Berjalan 100% di GPU RTX 3050 4 GB; cadangan: Gemini, lalu kutipan sumber |
| Integrasi antarmuka | Streamlit (menu Tanya Firaza) dan REST API | Antarmuka yang sama dengan Main Quest |

## Alur
```
pertanyaan → embedding (bge-m3) → FAISS: 3 dokumen terdekat → skor tertinggi ≥ ambang?
   ├─ tidak → MENOLAK menjawab, arahkan ke admin
   └─ ya → prompt (aturan + konteks bernomor + pertanyaan) → Qwen2.5-3B → jawaban + sumber [1], [2]
```

## Basis pengetahuan
`data/knowledge_base.json`: **24 dokumen pendek** (profil, produk, layanan, proses, kebijakan umum, sistem). **Isinya contoh sintetis** yang disusun dari brief Use Case 2 Firaza dan fitur aplikasi ini, **bukan FAQ resmi Firaza**. Hal yang tidak diketahui (harga, estimasi waktu, ekspedisi, kebijakan pembatalan) sengaja ditulis sebagai "dikonfirmasi oleh admin" dan tidak diisi angka karangan. Untuk penggunaan nyata, ganti isinya dengan FAQ resmi lalu bangun ulang indeks (otomatis saat isi berubah).

## Pengaman terhadap jawaban mengarang
1. **Abstain:** jika skor kemiripan teratas di bawah ambang (`RAG_MIN_SCORE`, awal 0,45), sistem menjawab bahwa informasi belum ada dan mengarahkan ke admin; sumber tidak ditampilkan.
2. **Prompt terikat konteks:** model hanya boleh menjawab dari konteks, dilarang mengarang harga, waktu, dan kebijakan, dan diminta menyebut nomor sumber.
3. **Sumber selalu ditampilkan** (judul, skor, kutipan) agar jawaban bisa diperiksa.
4. **Fallback berlapis:** Ollama → Gemini → kutipan dokumen terdekat, sehingga layanan tetap menjawab saat model bahasa tidak tersedia. Bila embedding (Ollama) mati, API memberi 503 dengan pesan jelas.

## Evaluasi (`python src/eval_rag.py`)
28 pertanyaan (`data/rag_eval.csv`): 22 dalam domain dengan dokumen jawaban yang diharapkan, dan 6 di luar domain yang seharusnya ditolak. Metrik: **hit@1**, **hit@3**, **MRR**, serta persentase pertanyaan luar domain yang benar ditolak dan pertanyaan dalam domain yang salah ditolak pada ambang saat ini. Skrip juga mengusulkan ambang skor (`ambang_disarankan`).

| Metrik | bge-m3 + FAISS (diuji di laptop pengembang) | Pembanding leksikal (hash kata, tanpa model) |
|---|---|---|
| hit@1 | **0,909** (20 dari 22) | 0,545 |
| hit@3 | **0,955** (21 dari 22) | 0,818 |
| MRR | **0,932** | 0,670 |
| Pertanyaan luar domain yang benar ditolak (ambang 0,45) | **100%** (6 dari 6) | 83% |
| Pertanyaan dalam domain yang salah ditolak (ambang 0,45) | **0%** (0 dari 22) | 27% |

Ambang yang disarankan skrip (titik tengah antara skor teratas tertinggi di luar domain dan terendah di dalam domain) adalah **0,469**; ambang 0,45 yang dipakai sudah memisahkan kedua kelompok dengan sempurna pada set ini. Waktu respons rata-rata sekitar **1-2 detik** pada RTX 3050 4 GB.

Pembanding leksikal adalah embedding sederhana tanpa model, sebagai garis dasar bahwa embedding semantik jauh lebih baik. **Catatan:** set evaluasi kecil (22 dalam domain, 6 luar domain) dan pertanyaannya ditulis oleh pengembang yang sama dengan basis pengetahuannya, sehingga angka ini cenderung optimis; perlakukan sebagai bukti kelayakan, bukan jaminan performa pada pertanyaan pelanggan nyata. Daftar pertanyaan yang gagal hit@3 ada di `reports/rag_eval.json`.

## Cara menjalankan
```powershell
ollama pull bge-m3 ; ollama pull qwen2.5:3b          # sekali; Ollama harus berjalan
pip install -r requirements.txt                       # termasuk faiss-cpu
python src/rag_service.py --build                     # bangun indeks (otomatis juga saat pertama kali bertanya)
python src/rag_service.py --ask "Bagaimana cara melacak pesanan saya?"
python src/eval_rag.py                                # evaluasi + saran ambang
```
Konfigurasi di `.env`: `RAG_MIN_SCORE`, `RAG_LLM_MODEL` (mis. `qwen2.5:1.5b` bila 3B terlalu lambat), `RAG_LLM_BACKEND` (`ollama`, `gemini`, atau `extractive`). Nama model sebaiknya dicek dengan `ollama list`.

## Keterbatasan
- Basis pengetahuan contoh dan kecil (24 dokumen); cakupan pertanyaan terbatas.
- Model 3B kuantisasi bisa keliru pada pertanyaan rumit; karena itu jawaban dibatasi konteks dan disertai sumber, tetapi tetap perlu pemeriksaan manusia untuk hal penting.
- Ambang skor dikalibrasi pada 28 pertanyaan saja; sesuaikan dengan data nyata.
- Tidak ada riwayat percakapan (setiap pertanyaan berdiri sendiri) dan belum ada pembatasan laju.
- Semua pemrosesan embedding dan generasi utama berjalan **lokal**; pertanyaan tidak dikirim ke layanan luar kecuali jalur cadangan Gemini dipakai.