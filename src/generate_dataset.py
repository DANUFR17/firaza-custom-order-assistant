"""
generate_dataset.py - Generator dataset sintetis pesan customer Firaza (v2)

Riwayat versi:
- v1: 18 template per label, split per template.
- v3: kosakata dan parafrase lebih beragam (kata kemasan, kata kerja custom, isi hampers selain handuk/sajadah,
      item packaging, acara tambahan), template permintaan admin/komplain baru, dan kalimat tambahan netral
      (pesan 2 kalimat). Vocab dibuat luas dan umum, bukan hanya kata yang muncul di manual_test.csv.
- v2: + template negasi (Ready Stock), + komplain yang menyebut produk (Other),
      + pesan pendek/messy (semua label), target seimbang per label, split proporsional
      per grup template, dan ID template unik (di v1 Custom Sajadah dan Custom Handuk
      sama-sama berawalan "CU").

Pendekatan:
- 5 label: Ready Stock, Custom Sajadah, Custom Handuk, Hampers, Other
- Tiap label punya 18 template inti (core) dengan slot (warna, jumlah, nama, acara, dll).
- Opener/closer + 3 gaya bahasa (formal, santai, noisy) ditambahkan setelahnya.
- SPLIT BERDASARKAN TEMPLATE (grouped split): template di test TIDAK muncul di train,
  supaya skor evaluasi tidak membengkak karena menghafal template.
- Aturan label mengikuti label_guideline.md.

Jalankan:  python generate_dataset.py --per-label 600 --seed 42 --out dataset_sintetis_v3.csv
"""
import argparse
import csv
import random
import re
from collections import Counter

# ---------------------------------------------------------------- vocab
FIRST = ["Budi", "Sarah", "Aisyah", "Rizky", "Dinda", "Fadhil", "Nabila", "Hanif", "Salsa", "Ahmad",
         "Siti", "Rahmat", "Putri", "Dimas", "Nadia", "Fauzan", "Intan", "Bagas", "Kirana", "Yusuf",
         "Zahra", "Alif", "Maya", "Reza", "Laras", "Hasan", "Naufal", "Citra", "Umar", "Tiara"]
LAST = ["Pratama", "Hidayat", "Rahman", "Saputra", "Wijaya", "Nugroho", "Lestari", "Kusuma",
        "Ramadhan", "Santoso", "Fadhilah", "Permana"]
LOGOS = ["PT Maju Jaya", "CV Berkah Abadi", "TK Ceria", "Masjid Al-Ikhlas", "Yayasan Nurul Huda",
         "SDIT Cahaya Bangsa", "Klinik Sehat Sentosa", "Komunitas Hijrah Bandung", "Pondok Darul Ilmi",
         "Koperasi Sejahtera", "Majelis Taklim An-Nur", "PT Karya Mandiri"]
WARNA_S = ["navy", "hitam", "maroon", "hijau botol", "abu-abu", "krem", "biru dongker", "coklat", "ungu", "tosca"]
WARNA_H = ["putih", "abu", "navy", "krem", "pink", "hijau sage", "coklat", "biru muda", "hitam"]
WARNA_BENANG = ["emas", "silver", "putih", "hitam", "navy", "maroon"]
UKURAN_H = ["mandi", "70x140", "dewasa", "handuk muka", "handuk tangan", "ukuran 50x100", "jumbo"]
UKURAN_BOX = ["kecil", "sedang", "besar", "25x25", "30x20"]
KOTA = ["Bandung", "Jakarta", "Surabaya", "Medan", "Yogyakarta", "Semarang", "Makassar", "Bekasi", "Depok", "Malang", "Palembang"]
ACARA = ["nikahan", "pernikahan", "pengajian", "wisuda", "ulang tahun", "aqiqah", "lebaran",
         "acara kantor", "reuni", "umroh", "tasyakuran", "khitanan", "perpisahan", "lamaran", "akad",
         "syukuran", "gathering kantor", "arisan"]
KEMASAN = ["bingkisan", "kemasan", "dus", "kotak", "box", "gift box", "paket"]
VERB_S = ["dicetak", "ditulis", "diberi", "dipasang", "ditambah", "disablon", "dicantumkan"]
VERB_H = ["dibordir", "dijahit", "diberi", "ditambah", "dipasang", "dicantumkan"]
ITEM_LAIN = ["mug", "sabun", "tumbler", "snack", "notes", "pulpen", "lilin", "tasbih", "mukena", "pouch",
             "tas kain", "alat tulis", "parfum", "kurma", "buku yasin", "sandal", "kain sarung"]
ITEM_PACK = ["bubble wrap", "kotak pengiriman", "pouch kain", "gift box kosong", "kardus", "lakban",
             "kertas kado", "tali pita", "plastik packing", "box mika"]
FILLER = ["Terima kasih sebelumnya.", "Saya tahu toko ini dari Instagram.", "Agak mendesak soalnya.",
          "Kemarin sudah sempat lihat-lihat di Shopee.", "Kalau bisa dibalas hari ini ya.",
          "Soalnya buat kebutuhan keluarga.", "Maaf ya banyak tanya.", "Sebelumnya sudah follow akunnya."]
FONT = ["kaligrafi", "script", "arab", "elegan", "klasik", "sans simple", "bold", "handwriting"]
LAYOUT = ["tengah", "pojok kanan bawah", "atas", "bawah", "pojok kiri atas"]
BUDGET = ["50rb", "75 ribu", "100rb", "150rb", "Rp200.000", "250rb", "Rp 300.000", "120rb"]
KONDISI = ["warnanya beda dengan foto", "kurang satu pcs", "agak kotor", "jahitannya lepas", "ukurannya lebih kecil dari deskripsi"]
JML = [1, 2, 3, 5, 10, 12, 20, 24, 25, 30, 50, 100, 150, 200, 300]
JML_W = [8, 5, 4, 4, 6, 3, 8, 3, 5, 6, 7, 6, 2, 3, 1]

NAME_TOKEN = "§N§"


class Ctx:
    """Context pengisi slot. Slot yang sama dalam satu template menghasilkan nilai yang sama."""

    def __init__(self, rng):
        self.rng = rng
        self.cache = {}
        self.name = None      # nilai asli nama/logo/inisial (disembunyikan dulu di balik token)
        self.jumlah = None

    def __getitem__(self, key):
        if key not in self.cache:
            self.cache[key] = getattr(self, "g_" + key)()
        return self.cache[key]

    def g_warna(self):
        return self.rng.choice(WARNA_S + WARNA_H)

    def g_warna_s(self):
        return self.rng.choice(WARNA_S)

    def g_warna_h(self):
        return self.rng.choice(WARNA_H)

    def g_warna_benang(self):
        return self.rng.choice(WARNA_BENANG)

    def g_ukuran_h(self):
        return self.rng.choice(UKURAN_H)

    def g_ukuran_box(self):
        return self.rng.choice(UKURAN_BOX)

    def g_kota(self):
        return self.rng.choice(KOTA)

    def g_acara(self):
        return self.rng.choice(ACARA)

    def g_font(self):
        return self.rng.choice(FONT)

    def g_layout(self):
        return self.rng.choice(LAYOUT)

    def g_budget(self):
        return self.rng.choice(BUDGET)

    def g_hari(self):
        return str(self.rng.randint(2, 10))

    def g_kondisi(self):
        return self.rng.choice(KONDISI)

    def g_kemasan(self):
        return self.rng.choice(KEMASAN)

    def g_verb_s(self):
        return self.rng.choice(VERB_S)

    def g_verb_h(self):
        return self.rng.choice(VERB_H)

    def g_item_lain(self):
        return self.rng.choice(ITEM_LAIN)

    def g_item_lain2(self):
        return self.rng.choice([i for i in ITEM_LAIN if i != self["item_lain"]])

    def g_item_pack(self):
        return self.rng.choice(ITEM_PACK)

    def g_produk_any(self):
        return self.rng.choice(["sajadah", "handuk"])

    def _n(self):
        self.jumlah = self.rng.choices(JML, JML_W)[0]
        return self.jumlah

    def g_jml(self):  # dengan satuan
        n = self._n()
        return self.rng.choice([f"{n} pcs", f"{n} pcs", f"{n} biji", f"{n} buah", f"{n}pcs"])

    def g_jml_n(self):  # angka saja (diikuti kata benda)
        return str(self._n())

    def g_nama(self):
        r = self.rng.random()
        a, b = self.rng.sample(FIRST, 2)
        if r < 0.40:
            self.name = a
        elif r < 0.65:
            self.name = f"{a} & {b}"
        elif r < 0.90:
            self.name = f"{a} {self.rng.choice(LAST)}"
        else:
            self.name = f"{a} {b} {self.rng.choice(LAST)}"
        return NAME_TOKEN

    def g_logo(self):
        self.name = self.rng.choice(LOGOS)
        return NAME_TOKEN

    def g_inisial(self):
        a, b = self.rng.sample("ABDFHIKMNRSYZ", 2)
        self.name = self.rng.choice([f"{a}&{b}", f"{a}.{b}.", f"{a}{b}"])
        return NAME_TOKEN


# ---------------------------------------------------------------- templates
T = {
    "Ready Stock": [
        "Sajadah polos {warna_s} ready tidak? mau beli {jml}",
        "Handuk {ukuran_h} warna {warna_h} masih ada stok?",
        "Berapa harga sajadah travel {warna_s}? bisa dikirim ke {kota}?",
        "Mau order handuk {jml}, warna {warna_h}, langsung kirim saja ya",
        "Stok {produk_any} {warna} ready hari ini tidak? butuh cepat",
        "Mau beli box packaging ukuran {ukuran_box}, {jml} ada tidak?",
        "Ongkir ke {kota} untuk {jml_n} {produk_any} berapa ya?",
        "Sajadah {warna_s} yang di Shopee itu masih ada? saya mau ambil {jml}",
        "Ada diskon untuk pembelian {jml_n} {produk_any} polos tidak?",
        "Handuk {ukuran_h} bahannya apa? katun semua? mau beli {jml}",
        "Bisa kirim hari ini {produk_any} {warna} {jml_n} pcs ke {kota}?",
        "Pembelian sajadah polos {jml} apakah dapat harga grosir?",
        "Tanya dulu ya, {produk_any} {warna} ukurannya berapa dan harganya?",
        "Saya mau beli {jml_n} sajadah tanpa nama, yang polos saja warna {warna_s}",
        "Pita dan kertas kado ready tidak? mau beli {jml}",
        "{produk_any} ready stock warna {warna} ada? mau COD ke {kota} bisa?",
        "Tolong kirim katalog {produk_any} yang siap kirim, tanpa custom",
        "Boleh minta foto asli {produk_any} {warna}? mau beli {jml} kalau cocok",
    ],
    "Custom Sajadah": [
        "Mau sajadah {warna_s} {jml} untuk {acara}, custom nama {nama}",
        "Sajadah bisa dicetak nama tidak? namanya {nama}, font {font} ya",
        "Pesan sajadah custom {jml} untuk {acara}, nama dan logo di pojok",
        "Sajadah bordir nama dong {jml} warna {warna_s}, nama {nama}",
        "Custom print nama {nama} di sajadah, {jml}, posisi nama di {layout}",
        "Minimal order sajadah printing nama berapa ya? untuk {acara}",
        "Ada sajadah dengan tulisan {nama}? mau {jml} warna {warna_s}",
        "Sajadah {warna_s} ready tidak? mau saya custom nama {nama}",
        "Berapa harga sajadah custom nama untuk {jml}? buat {acara}",
        "Saya mau pesan sajadah dengan desain sendiri, nanti saya kirim file desainnya, {jml}",
        "Sajadah souvenir {acara} {jml}, nama {nama}, tulisan arab di atas ya",
        "Order sajadah personalisasi {nama}, layout {layout}, font {font}, {jml}",
        "Bisa sajadah warna {warna_s} dicetak nama {nama} dan tanggal acara? {jml}",
        "Mau tanya proses pesan sajadah custom nama, estimasi jadi berapa hari? {jml} untuk {acara}",
        "Sajadah print nama anak-anak {jml}, nama {nama} di tengah, warna {warna_s}",
        "Pesanan sajadah custom untuk {acara}: {jml}, warna {warna_s}, nama {nama}, font {font}",
        "Sajadah dengan logo {logo} bisa? untuk {acara}, {jml}",
        "Preview desain sajadah nama {nama} bisa dikirim dulu sebelum produksi? {jml}",
    ],
    "Custom Handuk": [
        "Handuk bordir nama {nama} {jml} warna {warna_h}",
        "Mau handuk custom logo {logo} {jml} bordir, bisa?",
        "Handuk untuk {acara} bisa bordir nama {nama} dan tanggal tidak?",
        "Pesan handuk custom nama untuk kado, ukuran {ukuran_h}, {jml}",
        "Handuk muka bordir inisial {inisial} untuk souvenir {acara} {jml}",
        "Bordirnya bisa font {font} tidak? nama {nama} di handuk saya",
        "Handuk {warna_h} ready tidak? mau dibordir nama {nama}",
        "Berapa harga handuk bordir nama untuk {jml}? buat {acara}",
        "Minimal order handuk bordir berapa pcs ya?",
        "Handuk {ukuran_h} dibordir nama {nama} {jml}, estimasi berapa hari?",
        "Souvenir {acara} handuk custom {jml}, nama {nama}, warna {warna_h}",
        "Saya mau pesan handuk dengan nama {nama} dibordir di {layout}, {jml}",
        "Handuk dengan bordir logo perusahaan {logo}, {jml}, ukuran {ukuran_h}",
        "Bisa handuk {warna_h} ditambah bordir nama {nama}? {jml}",
        "Order handuk personalisasi {nama} font {font} warna benang {warna_benang}, {jml}",
        "Bordir nama di handuk anak, nama {nama}, {jml}, warna {warna_h}",
        "Mau kirim desain logo untuk dibordir di handuk {ukuran_h}, jumlah {jml}",
        "Handuk custom nama {nama} buat {acara}, {jml}, bahan katun ya",
    ],
    "Hampers": [
        "Mau hampers {acara} isi handuk dan sajadah, {jml_n} box",
        "Ada paket hampers {acara} budget {budget} per paket?",
        "Bisa bikin parcel handuk sama sajadah, kartu ucapan nama {nama}? {jml_n} paket",
        "Seserahan: sajadah, mukena, handuk, packing mewah ya, untuk {acara}",
        "Hampers corporate {jml_n} pcs, isi 2 item custom logo {logo} plus kartu ucapan",
        "Kado {acara} untuk teman, paket handuk dan sajadah dibungkus cantik",
        "Pesan hampers {jml_n} box untuk {acara}, isinya handuk bordir nama {nama} dan sajadah",
        "Ada pilihan paket kado {acara}? berapa harganya untuk {jml_n} paket",
        "Mau parcel {acara} {jml_n} pcs, isi handuk, sajadah, dan tasbih, boleh custom isi?",
        "Hampers {acara} budget {budget}, request packaging {warna} dan kartu ucapan",
        "Paket souvenir {acara}: handuk + sajadah, {jml_n} paket, nama {nama} di kartu",
        "Bisa tidak hampers isi sajadah custom nama {nama} dan handuk {warna_h}? {jml_n} box",
        "Tolong rekomendasikan isi hampers {acara} budget {budget}",
        "Mau pesan {jml_n} hampers {acara}, dibungkus box {warna}, pita dan kartu ucapan ya",
        "Paket kado {acara}, 1 handuk 1 sajadah, nama {nama}, dikirim ke {kota}",
        "Hampers lebaran kantor {jml_n} pcs, isi handuk bordir dan sajadah polos",
        "Ada katalog hampers untuk {acara}? mau lihat pilihan paketnya",
        "Tanya paket seserahan lengkap handuk sajadah, {jml_n} paket, untuk {acara}",
    ],
    "Other": [
        "Pesanan saya kok belum sampai, resi sudah {hari} hari",
        "Nama {nama} di sajadah pesanan saya salah ketik, gimana solusinya?",
        "Admin online sampai jam berapa?",
        "Mau jadi reseller, syaratnya apa saja?",
        "Bisa custom nama tidak?",
        "Terima kasih kak, barangnya sudah sampai, bagus",
        "Bisa nego harga tidak?",
        "Barang yang datang {kondisi}, tolong dicek ya",
        "Alamat toko di mana ya? bisa ambil langsung?",
        "Mau tanya cara bayar lewat transfer atau Xendit bagaimana?",
        "Bordir nama di handuk pesanan saya {hari} hari lalu hasilnya miring, mau komplain",
        "Apakah bisa kerja sama endorse atau kolaborasi dengan toko kakak?",
        "Pesanan atas nama {nama} sudah diproses belum? mohon update",
        "Mau batalkan pesanan, salah pilih {produk_any}",
        "Halo, mau tanya dong",
        "Apakah toko ini juga jual mukena atau tasbih?",
        "Bisa custom ya? berapa harganya?",
        "Retur bisa tidak kalau ukuran tidak sesuai?",
    ],
}


# ---------------------------------------------------------------- template tambahan v2
# Dibuat dari analisis error (negasi, komplain yang menyebut produk, pesan pendek/messy).
# Kalimat ditulis ulang dengan struktur sendiri, TIDAK menyalin pesan di manual_test.csv.
EXTRA = {
    "Ready Stock": {
        "v3": [
            "{item_pack} dijual terpisah dari produknya ga? butuh {jml_n} pcs",
            "Beli {item_pack} saja ada? ukuran {ukuran_box}",
            "Ada {item_pack} ready? untuk packing sendiri, {jml_n} biji",
            "Cek ketersediaan {produk_any} {warna} untuk pengiriman minggu ini, {jml_n} pcs",
            "Boleh minta daftar harga {produk_any} polos? belum butuh tambahan apa-apa",
        ],
        "negasi": [
            "Mau beli {produk_any} {warna}, polos aja nggak usah diapa-apain",
            "Cuma perlu {jml_n} {produk_any} buat dipakai sendiri, bukan buat kado atau souvenir",
            "{produk_any} {warna} yang biasa, ga perlu dicetak apa-apa, ready kan?",
            "Beli satuan aja ya, bukan paket. {produk_any} {warna}, {jml_n} pcs",
            "Saya cari {produk_any} tanpa embel-embel nama, model standar yang ready",
            "Kardus pengiriman ukuran {ukuran_box} dijual terpisah ga? beli kardusnya aja",
            "Kantong plastik dan selotip untuk packing ready? tidak sekalian beli produknya",
            "Handuk {warna_h} {ukuran_h} polos, belum mau dibordir, harganya berapa?",
        ],
        "pendek": [
            "{produk_any} {warna} ready?",
            "rdy kak {produk_any} polos?",
            "brp hrg {produk_any} polos {jml_n} pcs",
        ],
    },
    "Custom Sajadah": {
        "v3": [
            "Sajadah {warna_s} mau {verb_s} nama {nama}, bisa? {jml}",
            "Untuk {acara}, sajadah {jml_n} pcs, nanti {nama} {verb_s} di bagian {layout}",
            "Apakah sajadah bisa {verb_s} tulisan {nama} pakai font {font}? untuk {acara}",
            "Sajadah buat {acara} mau {verb_s} nama {nama} dan logo kecil, {jml}",
            "Bisakah sajadah {warna_s} {verb_s} nama {nama} di {layout}? kira-kira jadi berapa hari",
        ],
        "pendek": [
            "sajadah + nama {nama} bisa?",
            "sjd nama {nama} {jml} bs ga",
            "sajadah cetak nama brp?",
        ],
    },
    "Custom Handuk": {
        "v3": [
            "Handuk {warna_h} mau {verb_h} nama {nama}, {jml}",
            "Untuk {acara}, handuk {jml_n} pcs, nanti {nama} {verb_h} di {layout}",
            "Apakah handuk bisa {verb_h} {nama} pakai benang {warna_benang}? untuk {acara}",
            "Handuk {ukuran_h} buat {acara} mau {verb_h} nama {nama} dan logo kecil, {jml}",
            "Bisakah handuk {warna_h} {verb_h} nama {nama} di {layout}? kira-kira jadi berapa hari",
        ],
        "pendek": [
            "handuk bordir nama {nama} bisa?",
            "hnduk bordir {jml} brp",
            "bordir inisial di handuk bs?",
        ],
    },
    "Hampers": {
        "v3": [
            "Mau {kemasan} {acara} isi handuk, {item_lain}, dan {item_lain2}, {jml_n} paket",
            "Bisa dibuatkan {kemasan} berisi sajadah dan {item_lain} untuk {acara}? budget {budget}",
            "Susun {kemasan} {acara} dengan {item_lain}, {item_lain2}, dan handuk {warna_h}, {jml_n} set",
            "Ada {kemasan} isi {item_lain} dan {item_lain2}? untuk {acara}",
            "Request {kemasan} {acara}: 1 sajadah, 1 handuk, plus {item_lain}, kartu ucapan {nama}",
        ],
        "pendek": [
            "paket hampers {acara} ada?",
            "hampers handuk sajadah brp",
            "parcel {acara} {jml_n} pcs bisa?",
        ],
    },
    "Other": {
        "v3": [
            "Apakah ada lowongan kerja di toko ini?",
            "Ada gerai fisik yang bisa saya kunjungi?",
            "Pesanan {nama} mau diganti alamat pengirimannya, masih sempat?",
            "Boleh refund kalau pesanan belum diproses {hari} hari ini?",
            "Garansi berapa lama kalau jahitannya lepas?",
            "Nomor rekening untuk transfer yang resmi yang mana ya?",
            "Terima kasih admin, pelayanannya ramah sekali",
            "Saya salah isi nomor HP waktu pesan, bisa dibenerin?",
        ],
        "komplain": [
            "Hasil bordir di handuk pesanan saya tidak sesuai desain yang disetujui, mohon solusinya",
            "Handuk bordir yang saya terima ukurannya beda dari yang dipesan, bisa ditukar?",
            "Hampers pesanan saya sampai dalam kondisi penyok, apakah bisa diklaim?",
            "Pesanan sajadah custom atas nama {nama} sudah {hari} hari belum ada kabar produksi",
            "Tulisan di sajadah luntur setelah dicuci, garansinya bagaimana?",
            "Kartu ucapan di hampers saya tertukar dengan nama orang lain, tolong dicek",
            "Pesanan handuk custom saya lewat dari estimasi, kapan dikirim?",
        ],
        "pendek": [
            "resi mana kak?",
            "pesanan blm sampe",
            "min kok belum dibalas",
            "komplain barang rusak",
        ],
    },
}

# ---------------------------------------------------------------- gaya bahasa
OPEN = {
    "formal": ["", "Selamat pagi, ", "Selamat siang, ", "Selamat sore, ", "Assalamualaikum, ", "Halo kak, ", "Permisi, "],
    "casual": ["", "Halo kak, ", "Kak, ", "Permisi kak, ", "Assalamualaikum kak, ", "Min, ", "Hai Firaza, "],
    "noisy": ["", "kak ", "min ", "hlo kak ", "p ", "ka ", "mba "],
}
CLOSE = {
    "formal": ["", " Terima kasih.", " Mohon informasinya.", " Terima kasih atas bantuannya.", " Mohon dibantu ya."],
    "casual": ["", " Makasih ya kak", " Ditunggu infonya", " Thanks", " Mohon info ya"],
    "noisy": ["", " ya", " thanks", " mksh", " tlg info ya", " ditunggu"],
}
CASUAL_SLANG = [
    (r"\btidak\b", ["ga", "gak", "nggak"]), (r"\bsaja\b", ["aja"]), (r"\bsudah\b", ["udah"]),
    (r"\bbagaimana\b", ["gimana"]), (r"\bsama\b", ["sama", "sm"]), (r"\bapakah\b", [""]),
    (r"\bsaya\b", ["aku", "saya", "gue"]), (r"\bbelum\b", ["blm", "belum"]),
]
NOISY_ABBR = [
    (r"\bsajadah\b", ["sjd", "sajadah", "sajadh", "sjdh"]), (r"\bhanduk\b", ["hnduk"]),
    (r"\bhanduk\b", ["handuk"]), (r"\bberapa\b", ["brp", "brpa"]), (r"\buntuk\b", ["utk", "buat"]),
    (r"\byang\b", ["yg"]), (r"\bdengan\b", ["dg", "dgn"]), (r"\bbisa\b", ["bs", "bisa"]),
    (r"\bpcs\b", ["pcs", "biji", "bj", "pc"]), (r"\btolong\b", ["tlg"]), (r"\bkirim\b", ["krm", "kirim"]),
    (r"\bhari\b", ["hr"]), (r"\bterima kasih\b", ["makasih", "mksh"]), (r"\bdi\b", ["di", "d"]),
    (r"\bhampers\b", ["hampers", "hmpers", "parcel"]), (r"\bbordir\b", ["bordir", "bordr", "bordiran"]),
    (r"\bmau\b", ["mau", "mo", "pgn"]), (r"\bready\b", ["ready", "redy", "rdy"]),
]


def sub_rules(text, rules, rng, p):
    for pat, opts in rules:
        if rng.random() < p:
            text = re.sub(pat, lambda m: rng.choice(opts), text)
    return re.sub(r"\s+", " ", text).strip()


def add_typos(text, rng, p=0.07):
    out = []
    for w in text.split(" "):
        if len(w) >= 5 and "§" not in w and not any(c.isdigit() for c in w) and rng.random() < p:
            i = rng.randint(1, len(w) - 3)
            kind = rng.choice(["swap", "drop", "double"])
            if kind == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif kind == "drop":
                w = w[:i] + w[i + 1:]
            else:
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def style_text(core, style, rng, short=False):
    opener, closer = rng.choice(OPEN[style]), rng.choice(CLOSE[style])
    if short and rng.random() < 0.7:               # pesan pendek biasanya tanpa salam/penutup
        opener = closer = ""
    if opener and core[0].isupper():
        core = core[0].lower() + core[1:]          # "Halo kak, bisa ..." bukan "Halo kak, Bisa ..."
    if closer and core[-1] not in ".?!":
        core += "." if style == "formal" else rng.choice([".", ",", ""])
    text = opener + core + closer
    if style == "formal":
        text = text[0].upper() + text[1:]
        if text[-1] not in ".?!":
            text += "."
        return text
    text = text.lower()
    text = sub_rules(text, CASUAL_SLANG, rng, 0.75)
    if style == "noisy":
        text = sub_rules(text, NOISY_ABBR, rng, 0.6)
        text = add_typos(text, rng)
        if rng.random() < 0.6:
            text = re.sub(r"[,.]", "", text)
        if rng.random() < 0.3:
            text = re.sub(r"\byaa*\b", "yaa", text)
        if rng.random() < 0.15:
            text += rng.choice([" 🙏", " 😊", " 🙏🙏"])
    return text


def make_sample(rng, template, short=False):
    ctx = Ctx(rng)
    core = template.format_map(ctx)
    if not short and rng.random() < 0.15:           # pesan 2 kalimat dengan kalimat tambahan netral
        core = (core if core[-1] in ".?!" else core + ".") + " " + rng.choice(FILLER)
    style = rng.choices(["formal", "casual", "noisy"], [0.4, 0.4, 0.2])[0]
    text = style_text(core, style, rng, short)
    nama = ctx.name or ""
    if nama:
        shown = nama.lower() if (style != "formal" and rng.random() < 0.4) else nama
        text = text.replace(NAME_TOKEN.lower(), shown).replace(NAME_TOKEN, shown)
        nama = shown
    low = text.lower()
    has_s = bool(re.search(r"\bsj|sajadh", low))
    has_h = bool(re.search(r"handuk|hnduk", low))
    produk = "sajadah+handuk" if has_s and has_h else "sajadah" if has_s else "handuk" if has_h else ""
    return {
        "text": text,
        "produk": produk,
        "jumlah": ctx.jumlah if ctx.jumlah is not None else "",
        "nama_custom": nama,
        "acara": ctx.cache.get("acara", ""),
        "gaya_bahasa": style,
    }


def build_groups(label):
    """Daftar (template, grup) untuk satu label; grup dipakai agar split proporsional."""
    items = [(t, "base") for t in T[label]]
    for grp, tpls in EXTRA.get(label, {}).items():
        items += [(t, grp) for t in tpls]
    return items


def assign_splits(items, rng):
    """Split per template, proporsional di tiap grup (70/15/15). Grup kecil (<5) masuk train."""
    by_grp = {}
    for idx, (_, g) in enumerate(items):
        by_grp.setdefault(g, []).append(idx)
    split_of = {}
    for g, idxs in sorted(by_grp.items()):
        rng.shuffle(idxs)
        n = len(idxs)
        k = int(round(0.15 * n)) if n >= 5 else 0
        for pos, idx in enumerate(idxs):
            split_of[idx] = "test" if pos < k else "val" if pos < 2 * k else "train"
    return split_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="dataset_sintetis_v3.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, seen = [], set()
    for label, code in [("Ready Stock", "RS"), ("Custom Sajadah", "CS"), ("Custom Handuk", "CH"),
                        ("Hampers", "HP"), ("Other", "OT")]:
        items = build_groups(label)
        split_of = assign_splits(items, rng)
        n_tpl = len(items)
        base, rem = divmod(args.per_label, n_tpl)
        for tid, (tpl, grp) in enumerate(items):
            quota = base + (1 if tid < rem else 0)
            made, tries = 0, 0
            while made < quota and tries < quota * 60:
                tries += 1
                s = make_sample(rng, tpl, short=(grp == "pendek"))
                key = s["text"].lower()
                if key in seen:
                    continue
                seen.add(key)
                s.update(label=label, sumber="generator", template_id=f"{code}{tid + 1:02d}",
                         template_group=grp, split=split_of[tid])
                rows.append(s)
                made += 1
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = i

    cols = ["id", "text", "label", "produk", "jumlah", "nama_custom", "acara",
            "gaya_bahasa", "sumber", "template_id", "template_group", "split"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Total: {len(rows)} baris -> {args.out}")
    print("Per split :", dict(Counter(r["split"] for r in rows)))
    print("Per label :", dict(Counter(r["label"] for r in rows)))
    print("Per gaya  :", dict(Counter(r["gaya_bahasa"] for r in rows)))
    print("Per grup  :", dict(Counter(r["template_group"] for r in rows)))


if __name__ == "__main__":
    main()
