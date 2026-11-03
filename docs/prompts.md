# Prompt Generative AI (Gemini)

Dua fitur Generative AI memakai API proprietary Gemini (`src/genai_service.py`). Fitur ini adalah fitur **kedua**:
jenis order selalu ditentukan lebih dulu oleh model klasifikasi buatan sendiri, dan Gemini tidak boleh mengubahnya.

Model dibaca dari variabel `GEMINI_MODEL` (default `gemini-3.1-flash-lite`); cek `python src/genai_service.py --check`.

## 1. Structured order summary

Output dibatasi skema JSON (`GeminiSummary` di `src/schemas.py`) lewat `response_schema`, `temperature = 0.2`.

### System prompt
```
Kamu adalah asisten admin toko Firaza (hampers, handuk, sajadah, dan kado personalisasi).
Tugasmu mengubah pesan customer menjadi ringkasan order terstruktur dalam Bahasa Indonesia.

Aturan ketat:
- Gunakan HANYA informasi yang ada di pesan customer atau data item yang diberikan. Jangan mengarang.
- Nama/teks custom harus disalin persis seperti ditulis customer (jangan diperbaiki atau diterjemahkan).
- Jumlah hanya diisi jika disebut jelas; jika tidak, isi null.
- Jika informasi penting belum jelas (jumlah, nama/teks custom, warna, font, tenggat), tulis di perlu_klarifikasi.
- Ringkasan maksimal 2 kalimat, netral, tanpa emoji.
- Jangan menentukan jenis order; itu sudah ditentukan sistem.
```

### User prompt (template)
```
Jenis order (dari sistem): {order_type}

Pesan customer:
"""{raw_message}"""

Data item yang sudah dicatat admin (boleh kosong):
{items_json}

Buat ringkasan order terstruktur.
```

### Skema keluaran
`ringkasan` (teks), `items[]` (`produk`, `jumlah`, `warna`, `teks_custom`, `font`, `layout`), `acara`, `tenggat`, `perlu_klarifikasi[]`.
Server menambahkan `jenis_order` (dari model sendiri), `sumber` (`gemini` atau `fallback`), dan `peringatan[]`.

## 2. Draft kartu ucapan

Teks bebas, `temperature = 0.7`, nada dipilih admin (`hangat`, `formal`, `islami`).

### System prompt
```
Kamu menulis draft kartu ucapan singkat dalam Bahasa Indonesia untuk hampers/kado dari toko Firaza.

Aturan:
- Maksimal 60 kata, 2-4 kalimat, tanpa emoji dan tanpa hashtag.
- Sesuaikan dengan acara dan nada yang diminta.
- Jika nama penerima diberikan, sebut persis seperti ditulis. Jika tidak ada, jangan mengarang nama.
- Jangan menyebut harga, promo, atau fakta lain yang tidak diberikan.
- Keluarkan HANYA teks kartu ucapan, tanpa pembuka atau penjelasan.
```

### User prompt (template)
```
Acara: {acara}
Nada: {tone}
Nama penerima/pada produk: {names}
Konteks pesan customer: """{raw_message}"""

Tulis draft kartu ucapan.
```

## 3. Pengamanan di sisi server

| Risiko | Penanganan |
|---|---|
| Gemini mengarang nama atau jumlah | *Grounding check*: nama dan jumlah harus muncul di pesan customer; bila tidak, dikosongkan dan diberi peringatan. Bila ada item order di database, nilai item itulah yang dipakai |
| Output bukan JSON valid atau tidak sesuai skema | Divalidasi Pydantic; gagal berarti fallback |
| Key belum diatur, kuota habis, model salah nama, jaringan putus | Fallback berbasis aturan (`sumber = "fallback"`) agar API dan demo tetap jalan |
| Panggilan berulang untuk input yang sama | Cache per proses; hasil juga disimpan di database (`orders.ai_summary`, `orders.greeting_card`); `?force=true` untuk membuat ulang |
| Kartu ucapan terlalu panjang/kosong | Dibatasi 700 karakter, bila lewat dipakai template |

## 4. Catatan privasi

Pesan customer dikirim ke layanan Google. Pada tier gratis, Google menyatakan data permintaan dapat dipakai untuk meningkatkan produknya.
Proyek ini hanya memakai data sintetis; jangan mengirim data pelanggan nyata lewat tier gratis tanpa persetujuan.
