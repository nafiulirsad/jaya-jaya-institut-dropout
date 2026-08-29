# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

**Jaya Jaya Institut** — perguruan tinggi yang berdiri sejak tahun 2000 dan telah mencetak
banyak lulusan dengan reputasi sangat baik, namun menghadapi **angka dropout yang tinggi**.

| | |
|---|---|
| **Nama** | Nafiul Irsad |
| **Email** | nafiulirsad@gmail.com |
| **Id Dicoding** | nafiulirsad |

---

## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan tinggi yang telah berdiri
sejak tahun 2000. Hingga saat ini institut telah mencetak banyak lulusan dengan reputasi yang
sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias
**dropout**.

Jumlah dropout yang tinggi tentunya menjadi salah satu masalah besar untuk sebuah institusi
pendidikan. Dropout menggerus rasio kelulusan, menghilangkan pendapatan dari sisa masa studi
mahasiswa, dan dalam jangka panjang menurunkan reputasi akademik yang selama ini dijaga.

Persoalan utamanya adalah **institut baru mengetahui seorang mahasiswa dropout setelah kejadiannya
berlangsung**. Padahal jejak masalahnya — tunggakan pembayaran, mata kuliah yang tidak lulus,
nilai semester yang jatuh — umumnya sudah terlihat jauh sebelum mahasiswa benar-benar berhenti
kuliah. Tanpa alat bantu, pihak akademik tidak dapat mengetahui mahasiswa mana yang sedang menuju
dropout, sehingga bimbingan khusus selalu diberikan terlambat.

Oleh karena itu, Jaya Jaya Institut ingin **mendeteksi secepat mungkin siswa yang mungkin akan
melakukan dropout sehingga dapat diberi bimbingan khusus**. Selain itu, institut juga meminta
dibuatkan **dashboard** agar mereka mudah dalam memahami data dan memonitor performa siswa.

### Permasalahan Bisnis

1. **Angka dropout tinggi** dan menjadi ancaman langsung bagi target kelulusan, pendapatan, serta
   reputasi institut.
2. **Faktor pendorong dropout belum diketahui secara terukur.** Manajemen belum memiliki bukti
   kuantitatif mengenai variabel apa yang paling menentukan dan segmen mahasiswa mana yang paling
   rentan.
3. **Belum ada mekanisme deteksi dini.** Tidak ada sistem yang menandai mahasiswa berisiko sebelum
   mereka berhenti kuliah, sehingga intervensi selalu terlambat.
4. **Tidak ada alat monitoring.** Data mahasiswa masih berupa berkas mentah sehingga bagian
   akademik tidak dapat memantau performa per program studi, jalur masuk, atau kondisi keuangan.
5. **Mahasiswa yang masih aktif kuliah belum terpantau kondisinya.** Institut tidak mengetahui
   siapa di antara mereka yang perlu diintervensi pada semester berjalan.

### Cakupan Proyek

| Tahap | Keluaran |
|---|---|
| Data Understanding | Profil kualitas data, kamus data, pemeriksaan nilai kosong & duplikat, sebaran status mahasiswa |
| Exploratory Data Analysis | Dropout rate per segmen akademik, keuangan, demografi, program studi, dan jalur masuk (10 visualisasi) |
| Data Preparation | Penyaringan data berlabel (Dropout & Graduate), pembentukan target biner, 9 fitur turunan, pemisahan data mahasiswa aktif |
| Modeling | Perbandingan 3 algoritma + hyperparameter tuning `RandomizedSearchCV` (40 kombinasi × 5-fold) |
| Evaluation | ROC-AUC, PR-AUC, recall, F1/F2, pemilihan ambang operasional, permutation importance, kalibrasi band risiko |
| Deployment | Model `.joblib`, prototipe **Streamlit** (Streamlit Community Cloud), **business dashboard Metabase** di atas PostgreSQL via Docker |

### Persiapan

**Sumber data:** [Students' Performance — Dicoding Academy](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)
(turunan dari *Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository).
Dataset berisi **4.424 baris × 37 kolom**, dipisahkan dengan titik koma (`;`), tanpa nilai kosong
maupun baris duplikat. Salinan tersimpan di [data/data.csv](data/data.csv) agar seluruh proyek
dapat dijalankan ulang tanpa koneksi internet.

**Setup environment:**

```bash
# 1. Masuk ke direktori proyek
cd "Menyelesaikan Permasalahan Institusi Pendidikan"

# 2. Buat dan aktifkan virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Pasang seluruh dependency
pip install -r requirements.txt

# 4. Jalankan notebook analisis (opsional — notebook sudah berisi seluruh output)
jupyter notebook notebook.ipynb
```

**Menjalankan business dashboard (Docker):**

```bash
# Siapkan berkas database Metabase yang sudah berisi dashboard jadi
mkdir -p metabase-data && cp metabase.db.mv.db metabase-data/

# Nyalakan PostgreSQL (otomatis ter-seed dari data/students_scored.csv) + Metabase
docker compose up -d

# Tunggu ±60 detik, lalu buka http://localhost:3030
```

**Membangun ulang dashboard dari nol (opsional):**

```bash
python scripts/setup_metabase.py --url http://localhost:3030
```

---

## Ringkasan Proses Data Science

### Data Understanding

Dari 4.424 mahasiswa, status akhirnya terbagi menjadi **2.209 lulus (49,93%)**,
**1.421 dropout (32,12%)**, dan **794 masih aktif kuliah / Enrolled (17,95%)**. Data bersih
sepenuhnya: tidak ada nilai kosong, tidak ada duplikat, dan tidak ada kolom konstan.

### Data Preparation

Kelompok **Enrolled diperlakukan berbeda**: status mereka belum final — belum lulus, tetapi juga
belum dropout — sehingga labelnya tidak diketahui. Memasukkan mereka ke data latih sebagai "bukan
dropout" berarti memberi model label yang belum tentu benar.

Karena itu data **disaring hanya untuk mahasiswa berstatus `Dropout` dan `Graduate`**, lalu
dibentuk target biner **1 = Dropout, 0 = Graduate**. Sementara **794 mahasiswa `Enrolled`
disimpan terpisah** di [data/enrolled_students.csv](data/enrolled_students.csv) dan hanya dipakai
sebagai sasaran prediksi di masa mendatang.

Verifikasi distribusi target setelah penyaringan:

| Kelas | Jumlah | Proporsi |
|---|---|---|
| 0 = Graduate | 2.209 | 60,85% |
| 1 = Dropout | 1.421 | 39,15% |
| **Total data pemodelan** | **3.630** | rasio 1,55 : 1 |

Proporsi ini **cukup seimbang** sehingga model dapat mempelajari pola kedua kelas tanpa teknik
*oversampling*. Pelatihan tetap memakai `class_weight="balanced"` dan evaluasi bertumpu pada
ROC-AUC, PR-AUC, serta recall — bukan akurasi.

Selain itu ditambahkan **9 fitur turunan** (`approval_rate_1st/2nd/total`, `total_approved`,
`avg_grade`, `grade_trend`, `approval_trend`, `financial_risk`, `no_pass_flag`) dan fitur dipangkas
dari 36 menjadi **29 kolom** yang mudah diperoleh staf akademik.

### Temuan utama EDA

Seluruh dropout rate di bawah dihitung pada **3.630 mahasiswa yang status akhirnya sudah pasti**,
dengan rata-rata kohort **39,15%**.

| Faktor | Temuan |
|---|---|
| **Kelunasan pembayaran** | Belum lunas **94,03%** vs lunas 30,66% — pemisah paling tajam di seluruh data |
| **Tunggakan (debtor)** | Menunggak **75,54%** vs tidak 34,47% |
| **Beasiswa** | Penerima **13,83%** vs non-penerima 48,37% (bersifat protektif) |
| **Rasio kelulusan mata kuliah** | Lulus 1-50% → **99,03%**; tidak lulus satu pun → 87,92%; lulus 81-100% → **8,69%** |
| **Rata-rata nilai semester** | Di bawah 10 → **99,07%**; di atas 14 → **9,23%** |
| **Usia saat mendaftar** | 17-19 tahun 25,23% → 26-30 tahun **70,25%** |
| **Jenis kelamin** | Laki-laki **56,12%** vs perempuan 30,24% |
| **Waktu kuliah** | Kelas malam **50,74%** vs kelas siang 37,68% |
| **Jalur masuk** | Jalur usia >23 tahun **65,51%** vs seleksi reguler 28,98% |
| **Program studi** | Teknik Informatika **86,79%** vs Keperawatan **17,72%** |

---

## Business Dashboard

Dashboard **"Student Performance Dashboard - Jaya Jaya Institut"** dibangun dengan
**Metabase v0.63.14.3** di atas **PostgreSQL 15** dan berisi **25 kartu visualisasi** yang
tersusun dalam enam seksi naratif — dari *seberapa parah* kondisinya, *mengapa* mahasiswa
berhenti, *siapa* yang rentan, sampai *siapa yang harus ditindaklanjuti semester ini*.

![Student Performance Dashboard](nafiulirsad-dashboard.png)

| Akses | Nilai |
|---|---|
| URL | <http://localhost:3030/dashboard/2> |
| Email | `root@mail.com` |
| Password | `root123` |
| Nama dashboard | Student Performance Dashboard - Jaya Jaya Institut |
| Berkas database | [metabase.db.mv.db](metabase.db.mv.db) |
| Screenshot | [nafiulirsad-dashboard.png](nafiulirsad-dashboard.png) |

### Isi dashboard

| Seksi | Kartu | Pertanyaan bisnis yang dijawab |
|---|---|---|
| **1. Ringkasan Kondisi** | 4 KPI: Total Mahasiswa (4.424), Mahasiswa Dropout (1.421), Dropout Rate (39,15%), Mahasiswa Aktif Perlu Intervensi (532) | Seberapa parah kondisinya sekarang? |
| **2. Performa Akademik Tahun Pertama** | Rasio kelulusan mata kuliah, rata-rata nilai semester, performa rata-rata per status, tren antarsemester | Apakah performa akademik memisahkan dropout? |
| **3. Kondisi Keuangan** | Status keuangan gabungan, kelunasan pembayaran, tunggakan, beasiswa | Seberapa besar peran faktor keuangan? |
| **4. Demografi & Pola Perkuliahan** | Kelompok usia, jenis kelamin, waktu kuliah, status pernikahan | Siapa yang paling rentan? |
| **5. Program Studi & Jalur Masuk** | Dropout rate per program studi (min. 30 mahasiswa), 7 jalur masuk, band nilai seleksi | Di mana dropout terkonsentrasi? |
| **6. Deteksi Dini Berbasis ML** | 10 faktor paling berpengaruh, sebaran band risiko mahasiswa berlabel, validasi kalibrasi band, sebaran band risiko mahasiswa aktif, skor risiko per program studi, dan daftar 15 mahasiswa prioritas intervensi | Siapa yang harus ditindaklanjuti semester ini? |

### Prinsip desain yang diterapkan

- **Satu pesan per grafik.** Setiap kartu menjawab satu pertanyaan; judul kartu memakai nama
  dimensinya dan sumbu nilai selalu diberi nama `Dropout Rate (%)`.
- **Konteks pembanding.** Seluruh grafik dropout memakai **garis putus-putus pada 39,15%**
  (rata-rata kohort berlabel), sehingga pembaca langsung melihat segmen mana yang di atas normal —
  bukan sekadar membandingkan tinggi-rendah batang.
- **Warna bermakna, bukan dekoratif.** Oranye-merah `#E4572E` konsisten untuk metrik dropout,
  biru `#2E86AB` untuk metrik netral/model, hijau `#2E9E5B` untuk kondisi yang bersifat protektif
  (beasiswa) dan agregat skor risiko.
- **Urutan kategori mengikuti maknanya.** Kategori berjenjang (band usia, band nilai, band
  kelulusan, band risiko) diurutkan sesuai tingkatannya sehingga terbaca sebagai gradien.
- **Integritas data.** Dropout rate dihitung memakai view `students_final` (hanya 3.630 mahasiswa
  yang status akhirnya pasti) sehingga 794 mahasiswa aktif tidak menekan angka secara keliru;
  mereka ditampilkan pada kartu tersendiri sebagai hasil prediksi. Program studi dengan populasi
  di bawah 30 mahasiswa disembunyikan, dan kategori berlebih tidak pernah dilipat menjadi batang
  "Other" yang nilainya dijumlahkan. Skor mahasiswa berlabel adalah prediksi **out-of-fold**,
  sehingga setiap mahasiswa dinilai oleh model yang tidak pernah melihat datanya.

---

## Menjalankan Sistem Machine Learning

Prototipe dibangun dengan **Streamlit** dan sudah di-deploy ke **Streamlit Community Cloud**.

| Akses | Nilai |
|---|---|
| **URL prototipe** | **<https://nafiulirsad-jaya-jaya-institut-dropout.streamlit.app>** |
| Berkas aplikasi | [app.py](app.py) |
| Modul bersama | [preprocessing.py](preprocessing.py) |
| Model | [model/dropout_model.joblib](model/dropout_model.joblib) |

**Menjalankan secara lokal:**

```bash
# Pastikan dependency sudah terpasang (lihat bagian Persiapan)
source .venv/bin/activate
streamlit run app.py

# Aplikasi terbuka otomatis di http://localhost:8501
```

### Cara memakai aplikasi

Aplikasi memiliki empat halaman yang dapat dipilih pada sidebar.

**1. Prediksi Individu** — isi 20 data mahasiswa (dikelompokkan ke dalam tab Akademik, Keuangan,
Pendaftaran, dan Demografi), lalu tekan **Hitung Risiko Dropout**. Aplikasi menampilkan
probabilitas dropout, band risiko beserta bukti historisnya, *gauge* dengan penanda ambang
institut, daftar faktor risiko yang terdeteksi, dan rekomendasi tindakan yang menyesuaikan profil
mahasiswa.

![Halaman prediksi individu](assets/11_app_prediksi.png)

**2. Prediksi Massal (CSV)** — unduh templat CSV, isi dengan data satu angkatan, lalu unggah
kembali. Aplikasi memberi skor seluruh baris sekaligus, menampilkan ringkasan sebaran band risiko,
dan menyediakan tombol unduh hasil skoring. Contoh berkas siap unggah tersedia di
[data/contoh_unggah_massal.csv](data/contoh_unggah_massal.csv).

**3. Monitoring Angkatan** — ringkasan kondisi seluruh mahasiswa: dropout rate per program studi
dan per status keuangan (dihitung pada kohort berlabel), sebaran band risiko mahasiswa aktif,
serta **daftar prioritas intervensi** yang dapat disaring per band dan diunduh.

![Halaman monitoring angkatan](assets/12_app_monitoring.png)

**4. Tentang Model** — metrik evaluasi, alasan pemilihan ambang, tabel kalibrasi band risiko,
faktor paling berpengaruh, dan batasan penggunaan model.

### Ringkasan model

| Aspek | Nilai |
|---|---|
| Data latih | **3.630 mahasiswa berlabel** (1 = Dropout, 0 = Graduate); 794 mahasiswa *Enrolled* tidak dilibatkan |
| Algoritma terpilih | **Random Forest** (`n_estimators=706`, `max_depth=16`, `max_features='sqrt'`, `min_samples_leaf=4`, `min_samples_split=6`, `class_weight='balanced'`) |
| Kandidat yang dibandingkan | Logistic Regression, Random Forest, Hist Gradient Boosting |
| Dasar pemilihan | ROC-AUC validasi silang 5-fold pada data latih (**0,9531**) |
| ROC-AUC data uji | **0,9709** |
| PR-AUC data uji | **0,9692** (garis dasar kelas positif 0,39) |
| ROC-AUC out-of-fold (3.630 mahasiswa) | 0,9562 |
| Ambang operasional | **0,35** (memaksimalkan F2-score) |
| Recall pada ambang operasional | **94,72%** — hanya 15 dari 284 mahasiswa dropout yang lolos |
| Precision pada ambang operasional | 80,78% |

**Mengapa ambang 0,35 dan bukan 0,50?** Kedua jenis kesalahan tidak berbiaya sama. Melewatkan
mahasiswa yang benar-benar berisiko (*false negative*) berarti kehilangan mahasiswa beserta
pendapatan sisa masa studinya, sedangkan menandai mahasiswa yang ternyata aman (*false positive*)
hanya berbiaya satu sesi konseling. Menurunkan ambang dari 0,50 ke 0,35 menaikkan recall dari
91,2% menjadi **94,7%** — jumlah mahasiswa dropout yang lolos dari deteksi turun dari 25 menjadi
**15 orang**.

**Kalibrasi band risiko** (prediksi out-of-fold pada 3.630 mahasiswa berlabel):

| Band | Rentang probabilitas | Jumlah mahasiswa | Dropout aktual |
|---|---|---|---|
| Rendah | < 20% | 1.773 | **5,4%** |
| Sedang | 20% – 35% | 309 | 12,3% |
| Tinggi | 35% – 60% | 283 | 39,6% |
| Sangat Tinggi | ≥ 60% | 1.265 | **93,0%** |

Kenaikannya monoton dengan rentang 17 kali lipat, sehingga label band dapat langsung dipercaya
sebagai urutan prioritas kerja unit bimbingan.

---

## Conclusion

**Dropout adalah masalah nyata di Jaya Jaya Institut: dari 3.630 mahasiswa yang perjalanan
studinya sudah selesai, 1.421 orang (39,15%) berhenti sebelum lulus.** Proyek ini membuktikan
bahwa mayoritas kasus tersebut dapat diprediksi jauh sebelum terjadi. Model Random Forest yang
dibangun mencapai ROC-AUC **0,9709** pada data uji dan menangkap **94,7% mahasiswa dropout** pada
ambang operasional 0,35 — hanya dengan data yang **sudah dimiliki institut pada akhir tahun
pertama**, tanpa survei atau pengumpulan data baru.

**1. Apa penyebab utama dropout?** Ada dua akar masalah yang berdiri sendiri.

- **Kondisi keuangan.** Mahasiswa yang pembayarannya belum lunas dropout **94,03%** berbanding
  30,66% pada yang lunas — dari 486 mahasiswa yang menunggak, 457 berakhir dropout. Status
  *debtor* menghasilkan pola serupa (75,54% vs 34,47%), sementara beasiswa terbukti protektif
  (13,83% vs 48,37%). Dua fitur keuangan (`Tuition_fees_up_to_date` dan `financial_risk`) juga
  menempati **posisi ketiga dan keempat** pada permutation importance model.
- **Keterlibatan akademik tahun pertama.** Mahasiswa yang hanya lulus 1-50% mata kuliah dropout
  **99,03%** dan yang tidak lulus satu pun dropout **87,92%**, sementara yang lulus 81-100% mata
  kuliah hanya **8,69%**. Rasio kelulusan mata kuliah menempati dua posisi teratas pada
  permutation importance, dan semester kedua lebih prediktif daripada semester pertama
  (korelasi -0,74 vs -0,67).

**2. Siapa yang paling rentan?** Profil risiko tinggi terbentuk konsisten di beberapa dimensi:
mahasiswa **berusia 26-30 tahun (dropout 70,25%)**, **laki-laki (56,12%)**, **kelas malam
(50,74%)**, **berstatus menikah (54,74%)**, masuk lewat **jalur usia di atas 23 tahun (65,51%)**
atau **pindahan/pemegang ijazah lain (± 48%)**, dan terdaftar pada program studi **Teknik
Informatika (86,79%)**, **Pengelolaan Kuda (65,00%)**, atau **Manajemen kelas malam (63,55%)**.
Sebaliknya, Keperawatan — program studi terbesar dengan 666 mahasiswa — hanya dropout 17,72%.
**Masalah dropout tidak merata**, sehingga intervensi paling efisien bila difokuskan pada
segmen-segmen tersebut.

**3. Faktor apa yang paling menentukan menurut model?** Urutan sepuluh besarnya:
`approval_rate_total`, `approval_rate_2nd`, `Tuition_fees_up_to_date`, `financial_risk`,
`approval_rate_1st`, `Course`, `Curricular_units_2nd_sem_approved`, `approval_trend`,
`total_approved`, dan `Curricular_units_2nd_sem_enrolled`. Enam di antaranya adalah fitur hasil
rekayasa. Kehadiran `approval_trend` berarti model tidak hanya melihat posisi mahasiswa, tetapi
juga **arah perubahannya**.

**4. Apa hasil konkretnya sekarang?** Model diterapkan pada **794 mahasiswa yang masih aktif
kuliah** — kelompok yang sengaja tidak dilibatkan dalam pelatihan. Hasilnya: **362 mahasiswa masuk
band Sangat Tinggi** dan 170 band Tinggi, total **532 mahasiswa berada di atas ambang intervensi**.
Karena model dilatih pada dua kelompok dengan hasil yang sudah pasti dan saling berlawanan,
skornya cenderung terpolarisasi pada mahasiswa yang perjalanannya masih berlangsung — angka ini
**dibaca sebagai urutan prioritas, bukan vonis**. Daftar nama lengkap beserta urutannya tersedia
di [data/watchlist_enrolled.csv](data/watchlist_enrolled.csv), pada panel terakhir dashboard, dan
pada halaman *Monitoring Angkatan* aplikasi Streamlit.

### Rekomendasi Action Items

- **1. Jadikan tunggakan pembayaran sebagai alarm dropout, bukan sekadar urusan administrasi.**
  Terapkan aturan: mahasiswa yang pembayarannya lewat jatuh tempo **lebih dari 30 hari otomatis
  dirujuk ke unit bimbingan**, bukan hanya menerima surat tagihan. Dasarnya: 94,03% mahasiswa
  dalam kondisi ini berakhir dropout. Sediakan opsi cicilan, penundaan, atau beasiswa darurat —
  penerima beasiswa hanya dropout 13,83%, kurang dari sepertiga non-penerima, sehingga memperluas
  beasiswa ke mahasiswa berisiko tinggi berpotensi menekan dropout paling besar per rupiah yang
  dikeluarkan.

- **2. Pasang sistem peringatan dini berbasis mata kuliah pada minggu ke-6 setiap semester.**
  Mahasiswa yang lulus di bawah 50% mata kuliah dropout **99,03%**, dan lebih dari 1.000 mahasiswa
  berada pada kondisi tersebut. Jangan menunggu nilai akhir semester: pantau kehadiran dan hasil
  evaluasi tengah semester, lalu wajibkan konseling akademik bagi mahasiswa dengan rasio kelulusan
  di bawah 50%. Prioritaskan pula mahasiswa dengan `approval_trend` negatif karena performa yang
  memburuk antarsemester termasuk sepuluh besar penentu model.

- **3. Kerjakan daftar prioritas 362 mahasiswa aktif band Sangat Tinggi pada semester berjalan.**
  Gunakan [data/watchlist_enrolled.csv](data/watchlist_enrolled.csv) atau panel "Daftar Prioritas
  Intervensi" pada dashboard. Tetapkan satu dosen wali sebagai penanggung jawab per mahasiswa,
  dengan target kontak maksimal dua minggu dan pencatatan hasil intervensi. Pencatatan ini penting
  karena menjadi data pelatihan putaran berikutnya untuk mengukur **program bimbingan mana yang
  benar-benar bekerja**.

- **4. Bentuk program pendampingan khusus untuk mahasiswa non-tradisional.**
  Mahasiswa jalur usia di atas 23 tahun (dropout 65,51%), kelas malam (50,74%), dan yang sudah
  menikah (54,74%) memiliki pola risiko yang sama: **kuliah bersaing dengan pekerjaan dan tanggung
  jawab keluarga**. Solusinya bukan bimbingan akademik biasa, melainkan fleksibilitas — kelas
  daring/rekaman, batas mata kuliah lebih rendah pada tahun pertama, dan orientasi manajemen waktu
  di awal semester.

- **5. Lakukan audit kurikulum pada tiga program studi dengan dropout tertinggi.**
  Teknik Informatika (86,79%), Pengelolaan Kuda (65,00%), dan Manajemen kelas malam (63,55%)
  memiliki dropout **lebih dari tiga kali lipat** Keperawatan (17,72%). Selisih sebesar itu tidak
  dapat dijelaskan oleh profil mahasiswa saja. Periksa beban mata kuliah tahun pertama, tingkat
  kelulusan per mata kuliah, dan kualitas pembimbingan akademik pada ketiga program studi
  tersebut, lalu pelajari apa yang membuat Keperawatan berhasil dan replikasikan.

**Cara mengukur keberhasilan.** Pantau tiga indikator pada dashboard setiap akhir semester:
(1) dropout rate keseluruhan dengan **target turun dari 39,15% menjadi di bawah 30% dalam dua
tahun**, (2) jumlah mahasiswa band Sangat Tinggi yang berhasil bertahan setelah diintervensi, dan
(3) dropout rate pada tiga program studi prioritas. Model perlu **dilatih ulang setiap akhir tahun
akademik** dengan data angkatan terbaru — termasuk mahasiswa `Enrolled` yang status akhirnya sudah
diketahui — agar tetap relevan.

---

## Struktur Berkas

```
Menyelesaikan Permasalahan Institusi Pendidikan/
├── model/
│   ├── dropout_model.joblib          # Pipeline preprocessing + Random Forest terlatih
│   └── model_metadata.json           # Kontrak model: fitur, ambang, metrik, band risiko
├── data/
│   ├── data.csv                      # Dataset mentah dari Dicoding (4.424 x 37)
│   ├── students_scored.csv           # Tabel analitik + skor risiko (sumber dashboard)
│   ├── enrolled_students.csv         # 794 mahasiswa aktif, dipisah untuk prediksi ke depan
│   ├── feature_importance.csv        # Permutation importance seluruh fitur
│   ├── watchlist_enrolled.csv        # Daftar prioritas mahasiswa aktif + skor risiko
│   └── contoh_unggah_massal.csv      # Contoh berkas untuk fitur Prediksi Massal
├── assets/                           # 13 visualisasi hasil notebook & tangkapan layar aplikasi
├── docker/init/01_schema.sql         # Skema + seeding PostgreSQL
├── scripts/setup_metabase.py         # Pembangun dashboard Metabase via REST API
├── notebook.ipynb                    # Notebook analisis lengkap (sudah dieksekusi)
├── app.py                            # Prototipe Streamlit
├── preprocessing.py                  # Modul bersama notebook & aplikasi
├── docker-compose.yml                # PostgreSQL + Metabase
├── metabase.db.mv.db                 # Database Metabase berisi dashboard jadi
├── nafiulirsad-dashboard.png         # Tangkapan layar dashboard
├── nafiulirsad-video.mp4             # Video penjelasan singkat
├── requirements.txt                  # Dependency terkunci versinya
└── README.md
```
