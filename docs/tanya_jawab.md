# Lembar Persiapan: Pertanyaan yang Mungkin Ditanyakan dan Jawabannya

Tujuannya bukan menghafal kalimat, tapi **memahami alasannya**. Tiap jawaban singkat (sekitar 20 detik) dan menunjuk ke notebook yang menjelaskannya lebih dalam. Kalau ditanya sesuatu yang tidak ada di sini, jawab jujur: "belum saya uji", lalu jelaskan apa yang kamu tahu.

## A. Angka kunci (hafalkan ini saja dulu)

| Hal | Angka |
|---|---|
| Data | 3.000 pesan sintetis, 5 kelas seimbang (600 per kelas) |
| Test manual / test blind | 100 pesan / 30 pesan, ditulis sendiri |
| Model final | Ensemble CNN 1D (Keras, 3 seed) + TF-IDF Logistic Regression, bobot 0,5/0,5 |
| Validasi silang per template, 5×3 (acuan utama) | Ensemble 0,907; CNN 0,903; LogReg 0,869. CNN > LogReg di 12 dari 15 lipatan; ensemble ≈ CNN |
| Test blind (satu kali dibuka) | Macro F1 **0,902**, accuracy 0,90 (27 dari 30 benar) |
| Error blind | 3 dari 30; 2 berconfidence rendah, 1 yang yakin (komplain hampers) |
| Target di project plan | Macro F1 minimal 80%, target 85% |
| Uji otomatis | 11 uji lulus (API dan web) |

## B. Tentang model (yang paling mungkin ditanyakan)

**1. Kenapa pakai CNN?**
Pembeda utama jenis order ada di frasa pendek (2-4 kata) seperti "custom nama", "tanpa bordir". Filter konvolusi dengan jendela 2/3/4 kata dirancang mendeteksi pola itu di posisi mana pun. Data kecil dan pesan pendek, jadi BiLSTM tidak diperlukan; CNN ringan di CPU dan arsitekturnya dipilih dari skor validasi. Hasilnya terbukti: di validasi silang per template CNN lebih tinggi dari TF-IDF + LogReg sekitar 3 poin (di 12 dari 15 lipatan). *(Notebook 03 dan 04)*

**2. Kenapa tidak BiLSTM atau Transformer?**
BiLSTM sudah dicoba: hasilnya campur (unggul di satu split, kalah di split lain), jadi tidak ada alasan memilihnya. Transformer pra-latih (mis. IndoBERT) dilarang karena brief melarang model yang sudah ada; melatih Transformer dari nol pada 3.000 pesan tidak masuk akal. *(Notebook 03)*

**3. Kenapa ensemble?**
Ensemble sedikit di atas CNN di validasi silang (0,907 vs 0,903), selisih yang kecil dan tidak konsisten, jadi saya tidak mengklaim jauh lebih akurat. Alasannya **keamanan**: di test blind, LogReg setara atau lebih tinggi daripada CNN, jadi menggabungkan keduanya melindungi dari kelemahan salah satunya dan probabilitasnya lebih halus untuk `needs_review`. *(Notebook 04)*

**4. Apa itu TF-IDF?**
Mengubah pesan menjadi angka berdasarkan kata. TF = seberapa sering kata muncul di pesan; IDF = kata yang ada di hampir semua pesan (mis. "kak") diberi bobot kecil, kata khas (mis. "bordir") bobot besar. *(Notebook 03)*

**5. Apa itu Conv1D, kernel, dan max-pooling?**
Conv1D menggeser "jendela" 2, 3, atau 4 kata di sepanjang pesan untuk mencari pola frasa. Kernel = ukuran jendela. Max-pooling mengambil seberapa kuat pola itu muncul di mana saja dalam pesan. *(Notebook 03)*

**6. Apa itu embedding?**
Tiap kata diubah menjadi vektor angka (64 angka) yang dipelajari model selama pelatihan, sehingga kata yang dipakai di konteks mirip punya vektor yang dekat. Embedding saya dilatih dari nol, bukan pra-latih.

**7. Apa itu overfitting dan early stopping?**
Overfitting: model menghafal data latih sehingga buruk di data baru. Early stopping: pelatihan dihentikan saat loss validasi tidak membaik selama 5 epoch, lalu bobot terbaik dipulihkan. *(Notebook 03)*

**8. Kenapa macro F1, bukan accuracy?**
Macro F1 merata-ratakan F1 tiap kelas dengan bobot sama, jadi kelas yang sulit (Other) tidak tertutup kelas lain. Accuracy tetap dilaporkan.

**9. Kenapa skormu tidak 99%?**
Skor setinggi itu hanya muncul di split acak yang bocor (template yang sama ada di data latih dan uji). Pada template yang belum pernah dilihat, skor jujurnya sekitar 0,8-0,9. *(Notebook 01)*

## C. Tentang data dan evaluasi

**10. Kenapa data sintetis, dan apa kelemahannya?**
Data asli Firaza tidak disediakan. Dibuat dengan template dan slot, tiga gaya bahasa. Kelemahannya: bisa lebih mudah daripada dunia nyata dan kosakatanya terbatas. Untuk menguranginya: split per template, validasi silang per template, dan dua test set yang ditulis sendiri. *(Notebook 01)*

**11. Apa itu split per template dan GroupKFold?**
Semua pesan dari satu template masuk ke satu split saja (atau satu lipatan), jadi data uji berisi struktur kalimat yang belum pernah dilihat. Validasi silang per template mengulang itu 5 lipatan × 3 ulangan dengan pembagian berbeda sehingga skor tidak bergantung pada satu pembagian. Pembagi lipatannya saya buat sendiri karena `GroupKFold` bawaan scikit-learn berubah antar versi. *(Notebook 01 dan 04)*

**12. Kenapa test blind hanya dibuka sekali?**
Agar tidak ada "mengintip" dan menyetel model berdasarkan skor itu. Dikunci dengan file lock. Konsekuensinya jujur: setelah error blind dilihat, perbaikan berdasarkan error itu butuh test blind baru.

**13. Apakah 0,902 bisa dipercaya?**
Sebagian saja: n hanya 30, satu kesalahan menggeser sekitar 3 poin. Pegangan utama saya validasi silang per template sekitar 0,9; angka blind sebagai pemeriksaan tambahan.

**14. Apa error terbesar modelmu?**
Komplain yang menyebut banyak isi hampers diprediksi Hampers dengan yakin (0,86), padahal seharusnya Other. Dua error lain berconfidence rendah sehingga akan ditandai `needs_review`.

**15. Apa itu `needs_review`?**
Penanda untuk prediksi dengan confidence di bawah 0,6 agar admin memeriksa manual. Ambangnya nilai awal dan belum dikalibrasi.

## D. Tentang preprocessing

**16. Kenapa ada penanda negasi, dan kenapa stopword tidak dibuang?**
"Tidak perlu bordir" dan "mau bordir" sama-sama memuat kata bordir tetapi niatnya berlawanan. Kata setelah tidak/bukan/tanpa diberi awalan `NEG_`. Stopword "tidak/tanpa/bukan" dipertahankan karena justru menentukan niat. *(Notebook 02)*

**17. Apakah preprocessing benar-benar membantu?**
Menaikkan skor pada pesan bergaya manusia (val dan manual), tetapi tidak membantu (bahkan sedikit menurun) pada template sintetis baru dan di validasi silang per template. Jadi bermanfaat untuk pesan informal, tapi saya tidak mengklaim "selalu meningkatkan akurasi". *(Notebook 02)*

## E. Tentang Generative AI dan sistem

**18. Kenapa Gemini hanya fitur kedua?**
Aturan Main Quest: fitur pertama harus model buatan sendiri; API proprietary tidak boleh jadi fitur pertama. Klasifikasi jenis order dari model sendiri; Gemini untuk ringkasan terstruktur dan kartu ucapan.

**19. Bagaimana mencegah Gemini mengarang?**
Keluaran dibatasi skema JSON, nama dan jumlah dicek terhadap pesan asli (tidak ada di pesan berarti dikosongkan dan diberi peringatan), dan kalau ada item di database nilai itu yang dipakai. Ada fallback berbasis aturan bila Gemini gagal. *(Notebook 05)*

**20. Apa yang terjadi kalau Gemini mati saat demo?**
Sistem otomatis memakai fallback (badge berubah menjadi "Fallback"), jadi fitur tetap jalan dengan isi lebih sederhana.

**21. Kenapa FastAPI, Streamlit, dan SQLite?**
FastAPI: REST API dan dokumentasi otomatis, validasi dengan Pydantic. Streamlit: antarmuka cepat dibangun untuk satu orang dan diizinkan oleh checklist. SQLite: tanpa server, cukup untuk MVP lokal satu admin (advisor menyatakan autentikasi dan deployment tidak diperlukan).

**22. Apa keterbatasan terbesar proyek ini?**
Data sintetis (belum teruji pada pesan nyata Firaza), test blind kecil, dan tidak ada autentikasi. Kalau ada data nyata dianonimkan, itu langkah berikutnya untuk menguji dan melatih ulang.

## F. Tentang proses

**23. Apakah kamu memakai asisten AI?**
Jawab jujur sesuai kenyataan. Versi yang konsisten dengan lembar kerja advisor: asisten AI (Claude) membantu penulisan kode, analisis, dan dokumentasi; arah proyek, penulisan test set, dan pengujian dijalankan sendiri. Siapkan dirimu untuk **menjelaskan setiap bagian dengan kata-katamu sendiri**; itulah gunanya notebook ini.

**24. Kalau ada waktu lebih, apa yang kamu kerjakan?**
RAG dengan model Hugging Face (embedding, vector database, SLM lokal) untuk menjawab pertanyaan customer dari katalog dan FAQ, dan test blind baru untuk mengevaluasi perbaikan data.

## G. Cara berlatih
1. Jalankan notebook 01 sampai 05 berurutan dan baca teks penjelasannya.
2. Tutup notebook, lalu jawab pertanyaan B1-B3 dan C9-C11 dengan kata-katamu sendiri, sambil bicara keras.
3. Minta temanmu memilih tiga pertanyaan acak dari lembar ini dan jawab tanpa melihat.
