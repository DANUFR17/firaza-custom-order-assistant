# Panduan Penggunaan: Firaza Custom Order Assistant

Panduan ini untuk admin yang mengelola custom order. Tangkapan layar (`docs/img/`) ditambahkan oleh penulis setelah aplikasi dijalankan; letaknya ditandai **[Gambar: ...]**.

## 1. Menjalankan aplikasi

Prasyarat: Python 3.9+, repo sudah di-clone, dependensi terpasang (`pip install -r requirements.txt`), dan model beku tersedia (`python src/train_final.py`, sekali saja).

Buka **dua terminal** di folder proyek dengan venv aktif:

```powershell
uvicorn api:app --app-dir src --reload        # terminal 1: API
streamlit run app/streamlit_app.py            # terminal 2: web (http://localhost:8501)
```

Di sidebar web terlihat status **Model klasifikasi** dan **Gemini**. Bila Gemini tidak aktif, ringkasan dan kartu ucapan tetap dibuat dengan fallback berbasis aturan (ditandai "Fallback"). Untuk mengaktifkan Gemini, isi `GEMINI_API_KEY` di file `.env` (contoh: `.env.example`), lalu restart API.

**[Gambar: sidebar menu dan status]**

## 2. Alur kerja harian

### a. Membuat order baru (menu **Order Baru**)
1. **Customer:** pilih customer yang sudah ada, atau pilih *Customer baru* dan isi nama (kontak opsional).
2. **Pesan customer:** tempel pesan dari WhatsApp/marketplace, lalu klik **Analisis dengan AI**. Muncul usulan jenis order beserta confidence dan grafik probabilitas. Bila ada peringatan *confidence rendah*, periksa pesan secara manual.
3. **Detail order:** kolom *Jenis order* terisi otomatis sesuai usulan AI, **tetapi keputusan akhir ada pada admin**; ubah bila perlu. Isi kanal, tenggat (opsional), dan catatan.
4. **Item pesanan:** isi tabel (produk, jumlah, warna, nama/teks custom, font, layout). Tambah baris dengan tombol **+** di bawah tabel. Catat nama dengan teliti, karena nama inilah yang dipakai untuk ringkasan, kartu ucapan, dan produksi.
5. Klik **Simpan Order**. Order mendapat kode (mis. `FRZ-0001`) dengan status **New**.

**[Gambar: halaman Order Baru setelah analisis AI]**

### b. Mengelola order (menu **Daftar Order**)
Gunakan filter status, jenis, atau pencarian, lalu pilih order. Tiap order punya enam tab:

| Tab | Fungsi |
|---|---|
| **Detail** | Info order, pesan asli, usulan AI dan jenis final; tombol *Konfirmasi jenis* bila jenis final belum diisi |
| **Item** | Daftar item, pratinjau kasar teks custom (posisi dan gaya, bukan hasil cetak), tambah dan hapus item |
| **Ringkasan AI** | **Buat ringkasan** untuk structured order summary (produk, jumlah, warna, teks custom, acara, hal yang perlu diklarifikasi); **Buat ulang** untuk versi baru |
| **Kartu Ucapan** | Pilih nada (hangat/formal/islami), **Buat draf**, edit teks, lalu **Simpan teks** |
| **Status & Riwayat** | Pindahkan status dengan tombol (hanya langkah yang sah), tambahkan catatan, lihat riwayat |
| **Edit / Hapus** | Ubah jenis final dan catatan; hapus order (perlu centang konfirmasi) |

**[Gambar: tab Ringkasan AI dan tab Status & Riwayat]**

Catat item **sebelum** membuat ringkasan: nama dan jumlah yang ditampilkan diambil dari item yang kamu catat, bukan ditebak AI.

### c. Alur status
```
New  →  Design/Preparation  →  Production  →  Completed
              ↑                    |
              └──── revisi ────────┘
```
Status tidak bisa dilompati (mis. dari Design/Preparation langsung ke Completed). Setiap perubahan tercatat di riwayat lengkap dengan waktu (WIB) dan catatan.

### d. Customer dan Produk
- **Customer:** tambah, ubah, dan hapus (customer yang masih punya order tidak bisa dihapus).
- **Produk:** tambah produk, nonaktifkan atau aktifkan kembali. Harga dikosongkan sampai diisi sesuai katalog Firaza.

## 3. Cara membaca hasil AI

| Elemen | Arti |
|---|---|
| **Usulan jenis order** | Hasil model klasifikasi buatan sendiri (Ready Stock, Custom Sajadah, Custom Handuk, Hampers, Other). Hanya usulan |
| **Confidence** | Keyakinan model. Di bawah 60% ditandai perlu ditinjau (ambang awal, belum dikalibrasi) |
| **Sumber: Gemini** | Ringkasan atau kartu ucapan dihasilkan Gemini dan sudah lolos pemeriksaan |
| **Sumber: Fallback** | Gemini tidak aktif atau gagal; dibuat dengan aturan sederhana, isinya lebih terbatas |
| **Peringatan** (ikon ⚠️) | Misalnya nama atau jumlah dari Gemini tidak ditemukan di pesan customer sehingga dikosongkan |
| **Perlu klarifikasi** | Informasi penting yang belum jelas dan sebaiknya ditanyakan ke customer |

**Selalu periksa nama, jumlah, dan isi kartu ucapan sebelum produksi atau dicetak.**

## 4. Pemecahan masalah

| Masalah | Penyebab dan solusi |
|---|---|
| "API tidak dapat dihubungi" | Terminal 1 (`uvicorn`) belum berjalan atau port berbeda; jalankan atau atur `API_URL` |
| Sidebar: "Model klasifikasi belum dimuat" | Jalankan `python src/train_final.py`, lalu restart API |
| Ringkasan selalu "Fallback" | Cek `GEMINI_API_KEY` di `.env`, restart API, lalu cek `python src/genai_service.py --check` |
| `ModuleNotFoundError` | Venv belum aktif (prompt tanpa `(.venv)`); aktifkan dengan `.\.venv\Scripts\Activate.ps1` |
| Waktu berbeda dari jam di database | Database menyimpan UTC; web menampilkan WIB (UTC+7) |
| Ingin mengosongkan data uji | Hentikan API, hapus `data/firaza.db`, jalankan lagi (dibuat ulang otomatis) |

## 5. Keterbatasan yang perlu diketahui
- Belum ada login: cocok untuk satu admin di komputer lokal.
- Model dilatih dari data sintetis, sehingga pesan yang sangat tidak lazim bisa salah klasifikasi; manfaatkan penanda confidence rendah.
- Pratinjau teks hanya perkiraan, bukan hasil cetak sebenarnya.
- Tidak ada integrasi otomatis ke marketplace; pesan ditempel manual.
