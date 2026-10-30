# Label Guideline & Desain Dataset: Firaza Custom Order Classifier

Dokumen ini jadi acuan tunggal untuk (1) generator data sintetis, (2) test set manual, dan (3) dokumentasi preprocessing/model di deliverable capstone.

## 1. Definisi label

| Label | Kapan dipakai |
|---|---|
| **Ready Stock** | Beli atau tanya produk standar tanpa personalisasi: stok, warna, ukuran, harga, ongkir, packaging saja |
| **Custom Sajadah** | Sajadah dengan nama, tulisan, logo, desain, font, layout, printing, atau bordir |
| **Custom Handuk** | Handuk dengan nama, inisial, logo, atau bordir |
| **Hampers** | Paket kado berisi lebih dari satu jenis item, atau menyebut hampers/parcel/seserahan/paket kado |
| **Other** | Komplain, tracking resi, info umum, reseller, ucapan terima kasih, pembatalan/retur, pesan ambigu atau tidak relevan |

## 2. Aturan kasus perbatasan

1. **Niat utama menentukan label.** Kata hampers/parcel/paket/seserahan, atau dua jenis item atau lebih, berarti **Hampers**, walaupun ada custom nama.
2. **Satu jenis produk + personalisasi** berarti Custom Sajadah atau Custom Handuk. Contoh: souvenir 200 handuk bordir tetap Custom Handuk.
3. **Tanya stok produk yang akan dipersonalisasi** ("handuk putih ready? mau dibordir nama") berarti Custom, karena ada permintaan personalisasi.
4. **Komplain atau after-sales order custom** ("nama salah ketik", "bordir miring") berarti **Other**, karena niatnya bukan memesan.
5. **Tanya custom tanpa menyebut produk** ("bisa custom nama gak?") berarti **Other** (ambigu, admin perlu klarifikasi).
6. **Tanya harga atau minimal order produk custom** dengan produk jelas berarti label Custom produk itu.
7. **Beli packaging atau material saja** berarti Ready Stock.
8. Kalau masih ragu, pesan **tidak dimasukkan** ke dataset.
9. **Jasa bordir/printing dengan barang bawa sendiri** ("saya bawa handuk sendiri, bisa dibordir nama?") berarti Custom Handuk atau Custom Sajadah sesuai barangnya, karena permintaannya tetap personalisasi.
10. **Semua komplain atau after-sales, termasuk paket hampers yang rusak**, berarti **Other** (perluasan aturan 4).
11. **Negasi menentukan niat.** "Tidak perlu bordir", "tanpa nama", dan "bukan paket kado" berarti kelas non-custom/non-hampers (umumnya Ready Stock); "bukan beli yang polos" berarti Custom.

## 3. Contoh per kelas

| Label | Contoh |
|---|---|
| Ready Stock | "kak sajadah polos navy ready ga? mau 1 aja" / "handuk mandi putih 70x140 masih ada stok?" / "box packaging ukuran sedang beli 10 ada?" |
| Custom Sajadah | "Mau sajadah navy 20 pcs untuk nikahan, custom nama Budi & Sarah" / "sajadah bisa dicetak nama ga? nama Aisyah, font kaligrafi ya" |
| Custom Handuk | "handuk bordir nama Dinda 2 pcs warna putih" / "handuk muka bordir inisial A&R buat souvenir nikahan 200 pcs" |
| Hampers | "mau hampers lebaran isi handuk + sajadah, 25 box" / "seserahan: sajadah, mukena, handuk, packing mewah ya" |
| Other | "pesanan saya kok belum sampai, resi sudah 5 hari" / "namanya salah ketik di sajadah kemarin, gimana?" / "bisa custom nama gak?" |

## 4. Skema dataset (`dataset_sintetis_v1.csv`)

| Kolom | Isi |
|---|---|
| `id` | nomor unik |
| `text` | pesan customer (input model) |
| `label` | salah satu dari 5 kelas (target model) |
| `produk`, `jumlah`, `nama_custom`, `acara` | metadata hasil generator; dipakai untuk analisis error dan untuk membandingkan structured summary Gemini dengan nilai asli |
| `gaya_bahasa` | formal / casual / noisy |
| `sumber` | `generator` atau `manual` |
| `template_id` | template inti pembentuk pesan (dipakai untuk grouped split) |
| `split` | train / val / test |

## 5. Desain generator (`generate_dataset.py`)

- **18 template inti per label** (90 total) dengan slot: warna, ukuran, kota, acara, font, layout, budget, jumlah, nama/logo/inisial.
- **Tiga gaya bahasa:** formal (~40%), santai (~40%), noisy (~20%: singkatan seperti `sjd`, `brp`, `utk`, typo, tanda baca hilang, emoji sesekali). Opener dan closer divariasikan.
- **Nama dilindungi token** saat transformasi supaya tidak ikut diberi typo, lalu dikembalikan (kadang huruf kecil agar realistis).
- **Kasus perbatasan sengaja dibuat** sesuai aturan di bagian 2, misalnya komplain bordir di kelas Other dan tanya stok + custom di kelas Custom.
- **Deduplikasi** teks (case-insensitive) sehingga tidak ada pesan identik.
- **Ukuran:** 28 pesan per template, 504 per label, 2.520 total, seimbang antar kelas.

### Grouped split (penting untuk evaluasi yang jujur)

Split dilakukan **per template**, bukan per baris: 12 template train, 3 val, 3 test per label. Template di test tidak pernah dilihat model saat training, jadi skor tidak membengkak karena menghafal struktur kalimat.

Cek cepat dengan baseline TF-IDF + Logistic Regression pada data ini:

| Skema evaluasi | Macro F1 |
|---|---|
| Random split (bocor, template yang sama ada di train dan test) | 1.000 |
| Grouped split, val | ~0.89 |
| Grouped split, test | ~0.78 |
| GroupKFold 5 lipatan (per template), rata-rata | ~0.84 |

Angka ini baru sanity check, bukan hasil final. Dua catatan:
1. Skor test bervariasi karena hanya 3 template per label di test. Untuk laporan final, laporkan juga GroupKFold agar tidak bergantung pada satu pembagian.
2. Selisih random split (1.0) vs grouped split (~0.8) menunjukkan risiko utama data sintetis. Gunakan angka grouped split dan test set manual sebagai angka resmi.

## 6. Test set manual (tugas kamu, ~100 pesan)

Tulis sendiri, terpisah dari generator, sekitar 20 per label, simpan sebagai `data/manual_test.csv` dengan kolom `text,label,sumber=manual`. Tips supaya independen dari template generator:
- Pakai kalimat yang sengaja berbeda dari template (urutan kata lain, bahasa chat WhatsApp sungguhan, pesan panjang beberapa kalimat).
- Sertakan pesan campuran (misal tanya ongkir sekaligus custom), dan kasus perbatasan dari bagian 2.
- Jangan lihat output generator saat menulis.
- Catat di mana kamu ragu menentukan label; itu bahan bagus untuk bab analisis error.

## 7. Sitasi
Dataset ini sepenuhnya sintetis (dibuat dengan `generate_dataset.py`, tanpa data asli Firaza). Cantumkan di dokumentasi: library yang dipakai (pandas, scikit-learn, TensorFlow), dan referensi publik apa pun yang nanti ditambahkan.
