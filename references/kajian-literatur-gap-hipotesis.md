# Kajian Literatur, Research Gap, Hipotesis, dan Rencana Penelitian

Diagnosis PCOS hari ini masih bergantung pada mata operator USG: citra yang sama
bisa dibaca berbeda oleh tiap sonografer, dan folikel-folikel kecil
mudah terlewat. Dua kelompok peneliti mencoba menjawab persoalan yang sama
dari dua arah berbeda: Tiwari dkk. lewat arsitektur ringan ber-attention,
Sundari dkk. lewat optimasi dan interpretabilitas. Dokumen ini mengisahkan apa
yang mereka capai, benang apa yang mereka tinggalkan, hipotesis apa yang lahir
dari benang itu, dan penelitian seperti apa yang akan menuntaskannya. Disusun
sebagai bahan pertahanan topik tugas akhir di hadapan dosen, dokumen ini
menjawab empat pertanyaan secara berurutan: pertama, apa yang sebenarnya
dibahas oleh dua literatur terdahulu; kedua, celah (gap) apa yang tersisa
dari keduanya; ketiga, hipotesis apa yang dapat diangkat dari kekurangan
kedua artikel tersebut; dan keempat, penelitian seperti apa yang akan
dijalankan untuk menjawabnya.

Judul kerja: **Pengaruh Wavelet Denoising dan Segmentasi Folikel terhadap
Klasifikasi PCOS dengan DenseNet-121 + Attention dan Grad-CAM**.

---

## 1. Apa yang dibahas dua literatur terdahulu

### 1.1. Artikel 1: Tiwari, Shukla, & Sharma (2026)

> _Attention-guided lightweight deep learning architecture for classification
> of polycystic ovary syndrome from ultrasound images._
> Intelligence-Based Medicine, 13, 100358.

**Masalah yang diangkat.** Arsitektur CNN konvensional sulit menangkap
_long-range contextual dependencies_ pada citra USG grayscale, sehingga
diskriminasi folikel ovarium menjadi kurang andal. Di sisi lain, model yang
akurat umumnya berat dan tidak cocok untuk layanan primer atau aplikasi
kesehatan seluler dengan sumber daya terbatas.

**Metode.** Empat backbone (CNN kustom, VGG19, DenseNet-121, EfficientNet-B0)
masing-masing dipadukan modul _self-attention_ (proyeksi Q, K, V dengan
_scaled dot-product attention_). Preprocessing memakai denoising wavelet
metode BayesShrink dilanjutkan Gaussian smoothing; augmentasi berupa rotasi,
_flip_ horizontal, dan _zoom_. Pelatihan memakai optimizer Adam dengan
_binary cross-entropy loss_. Dataset yang dipakai adalah PCOS-XAI Ultrasound
Dataset (11.784 citra: 6.784 PCOS, 5.000 sehat) yang memang memuat persoalan
dunia nyata seperti duplikat, multi-resolusi, dan
ketidakseimbangan kelas, namun tanpa metadata pasien. Yang perlu digarisbawahi:
artikel ini menerapkan denoising pada seluruh citra tanpa pernah melaporkan
ablasi denoise vs non-denoise, sehingga kontribusi denoising terhadap akurasi
tidak terukur secara terpisah.

**Hasil dan klaim.** Keempat model mencapai akurasi 99,13%.
DenseNet-121 + Attention menjadi yang terbaik (presisi 1,000; recall 0,9851;
AUC 0,9997), sedangkan EfficientNet-B0 + Attention (4,87 juta parameter)
menjadi solusi paling hemat parameter dengan akurasi setara. Klaim utama:
arsitektur ringan ber-attention layak untuk skrining _real-time_ dan
_low-resource_. Penulis menutup dengan peluang pendekatan multimodal
sebagai pekerjaan lanjutan.

### 1.2. Artikel 2: Sundari et al. (2025)

> _Transfer Learning-Enhanced CNN Model for Integrative Ultrasound and
> Biomarker-Based Diagnosis of Polycystic Ovarian Disease._
> Scientific Reports, 15:34519.

**Masalah yang diangkat.** Interpretasi USG transvaginal bersifat subjektif dan
bergantung pada operator, sehingga akurasi antar sonografer bervariasi.
Studi-studi terdahulu cenderung terbatas pada dataset kecil, minim optimasi model,
dan minim integrasi data klinis multimodal.

**Metode.** Backbone Enhanced EfficientNet-B3 dengan modul attention,
_batch normalization_, dan dropout (0,35, ditemukan melalui Bayesian
Optimization). Hyperparameter (learning rate, batch size, dropout, pilihan
optimizer, unit fully-connected) disetel dengan Bayesian Optimization.
Ketidakseimbangan kelas ditangani dengan SMOTE pada fitur klinis tabular
(memakai pustaka `imbalanced-learn`), augmentasi geometris, dan
interpretabilitas disediakan lewat visualisasi Grad-CAM. Dataset memakai
PCOSGen (~3.200 citra berlabel) beserta metadata klinis (LH, FSH, AMH).

**Catatan umum kedua artikel:** keduanya hanya melaporkan metrik
_point-estimate_ tunggal (akurasi, AUC, dan sejenisnya) tanpa interval
kepercayaan, kalibrasi, maupun kuantifikasi ketidakpastian prediksi.

**Hasil dan klaim.** Akurasi 94,8% (sensitivitas 93,2%; spesifisitas 95,5%;
presisi 94,0%; F1 93,6%; AUC 0,97). Studi ablasi menunjukkan penambahan modul
attention saja menaikkan akurasi 91,0% → 93,2%, dan kombinasi
attention + augmentasi mencapai 94,8%. **Catatan penting yang dinyatakan
penulis sendiri:** data biomarker (LH, FSH, AMH) _tidak_ dimasukkan ke model
klasifikasi final yang menghasilkan angka-angka tersebut. Biomarker hanya
dipakai untuk analisis komparatif/visualisasi karena data
citra-biomarker berpasangan yang tersedia terbatas. Arsitektur fusi multimodal
dideskripsikan di makalah, tetapi dinyatakan sebagai arah pekerjaan
berikutnya, bukan yang diimplementasikan dan dievaluasi.

Dibaca berdampingan, kedua artikel ini berhasil di atas kertas, 99,13% dan
94,8%, tetapi meninggalkan benang yang tak terselesaikan. Keduanya
menjanjikan multimodal tanpa membangunnya, memakai attention tanpa
membandingkannya, dan mengklaim ringan tanpa mengukurnya. Benang-benang
itulah yang dipetakan pada Section 2.

---

## 2. Research gap dari kedua artikel

| No  | Gap                                                                                            | Bukti dari artikel                                                                                      |
| --- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| G1  | Fusi multimodal citra + biomarker dideskripsikan tetapi **tidak diimplementasikan/dievaluasi** | Artikel 2 §4.1: biomarker tidak masuk model final; integrasi penuh dinyatakan sebagai _subsequent work_ |
| G2  | **Tanpa validasi silang antar dataset;** keduanya hanya validasi internal pada satu dataset    | Artikel 1 memakai PCOS-XAI saja; Artikel 2 memakai PCOSGen saja                                         |
| G3  | **Tanpa kuantifikasi ketidakpastian;** output hanya prediksi biner tanpa interval kepercayaan  | Tidak ada MC Dropout, ensemble, maupun kalibrasi pada keduanya                                          |
| G4  | **Tanpa perbandingan sistematis mekanisme attention;** masing-masing memakai satu varian       | Artikel 1: satu modul SA; Artikel 2: modul attention tidak dirinci variannya dan tanpa pembanding       |
| G5  | Klaim **lightweight/low-resource tanpa benchmark deployment** (latency, memori, uji perangkat) | Artikel 1 mengklaim cocok untuk _portable ultrasound_ tanpa pengukuran latency/inferensi                |

G1 adalah gap terbesar secara akademik, tetapi tidak diambil karena dataset
citra-biomarker berpasangan yang terbuka sulit diperoleh. G2-G3 dicatat sebagai
keterbatasan dan peluang lanjutan. Fitur-fitur Artikel 2 yang tidak diadopsi
(Bayesian Optimization, SMOTE, dan backbone EfficientNet-B3) sengaja
dikeluarkan dari scope karena di luar judul kerja (fokus: wavelet denoising,
segmentasi folikel, DenseNet-121 + Attention, Grad-CAM). Penelitian ini
mengambil **G4 sebagai gap utama** (dengan unsur G5 melalui pengukuran
parameter dan latency), karena dapat dikerjakan tuntas dengan dataset yang
tersedia.

Dari lima benang tersebut, tidak semuanya dapat ditarik dengan sumber daya
yang ada: multimodal butuh data berpasangan yang tak tersedia, sedangkan
validasi antar dataset dan kuantifikasi ketidakpastian melampaui scope tugas
kuliah. Tiga benang yang tersisa: klaim denoising yang belum diabalasi,
fokus folikel yang masih implisit, dan efisiensi attention yang belum
dibandingkan, dirumuskan sebagai tiga hipotesis pada Section 3.

---

## 3. Hipotesis yang diangkat dari kekurangan kedua artikel

Penelitian ini menjawab tiga rumusan masalah berikut:

1. Seberapa besar pengaruh wavelet denoising (BayesShrink) terhadap kualitas citra USG ovarium?
2. Apakah ROI/segmentasi folikel unsupervised meningkatkan akurasi klasifikasi PCOS?
3. Mekanisme attention mana yang paling efisien: self-attention vs CBAM?

Setiap hipotesis ditulis dalam pasangan H0/H1 beserta kriteria penerimaan
deskriptif (desain memakai satu split train/val/test sehingga uji signifikansi
formal tidak valid, batasan ini dinyatakan sejak awal).

### H1: Wavelet denoising meningkatkan kualitas citra dan performa klasifikasi

- **H0:** Wavelet denoising (BayesShrink) tidak mengubah kualitas citra dan
  akurasi klasifikasi secara signifikan dibanding baseline.
- **H1:** Wavelet denoising meningkatkan kualitas citra (PSNR/SSIM lebih
  tinggi) serta akurasi/AUC klasifikasi dibanding baseline.
- **Landasan dari literatur:** Artikel 1 menerapkan BayesShrink + Gaussian
  smoothing pada seluruh citra dan melaporkan metode wavelet "superior dalam
  mempertahankan detail struktural sambil mereduksi noise" (bagian 3.3-3.4;
  prosedur pada 3.6). Namun Artikel 1 tidak pernah mengukur kontribusi
  denoising secara terpisah (tanpa ablasi denoise vs non-denoise). H1 mengisi
  kekurangan itu dengan menguji klaim preprocessing tersebut secara langsung.
- **Diuji oleh:** eksperimen E1 vs E2. **Kriteria:** H1 diterima jika akurasi
  atau AUC E2 lebih tinggi secara ketat (">") dari E1 pada minimal satu metrik,
  tanpa regresi melebihi 0,005 pada metrik satunya; hasil identik tidak
  dihitung diterima. PSNR/SSIM dilaporkan sebagai bukti deskriptif sisi PCD,
  bukan penentu verdict: tanpa citra ground-truth bersih, PSNR/SSIM
  (raw vs denoised) mengukur seberapa besar citra berubah, bukan kualitas
  yang lebih baik.

### H2: Segmentasi/ROI folikel meningkatkan akurasi klasifikasi

- **H0:** Input ROI/_masked_ dari segmentasi folikel unsupervised tidak
  meningkatkan akurasi dibanding citra penuh.
- **H1:** Input ROI/_masked_ meningkatkan akurasi/F1/AUC karena model fokus
  pada area folikel dan kurang terpengaruh background.
- **Landasan dari literatur:** Artikel 1 menyatakan modul self-attention
  membuat model "fokus pada area folikular dan ovarial yang salient"
  (Pendahuluan; §3.2) dan diskusinya mengonfirmasi model berfokus pada
  "_cystic follicles and abnormal ovarian texture instead of background
  noise_" (§5). Artikel 2 menyatakan modul attention "menekankan region
  spasial yang relevan dan menekan background noise" untuk mendeteksi
  "_follicular clusters and cystic formations_" (§3.1.3). Kriteria Rotterdam
  sendiri berbasis jumlah folikel/volume ovarium (§1). Kedua artikel
  mengandalkan _attention implisit_ untuk fokus folikel, tetapi tidak pernah
  menguji _fokus eksplisit_ lewat segmentasi/ROI. Celah inilah yang diuji H2.
- **Diuji oleh:** eksperimen E3 vs E4 vs E5. **Kriteria:** varian utama
  diprespesifikasi sejak awal, yaitu E4 (ROI crop Otsu), dengan AUC sebagai
  metrik utama. H2 diterima jika AUC E4 > E3 dan F1 E4 tidak lebih buruk
  melebihi 0,005. E5 (masked) dilaporkan deskriptif sebagai pembanding
  sekunder dan tidak ikut menentukan verdict, sehingga tidak terjadi seleksi
  varian terbaik di atas test set.

### H3: CBAM setara akurat dengan parameter lebih sedikit (efisien)

- **H0:** Tidak ada perbedaan efisiensi-akurasi yang berarti antara
  self-attention dan CBAM.
- **H1:** CBAM mencapai akurasi/AUC setara atau lebih baik dengan parameter
  lebih sedikit, sehingga lebih efisien untuk skrining low-resource.
- **Landasan dari literatur:** Artikel 1 menunjukkan arsitektur ringan
  (EfficientNet-B0 + attention, 4,87 juta parameter) mencapai akurasi setara
  model besar (99,13%) dan menyimpulkan keduanya "memberikan trade-off
  terbaik antara akurasi, sensitivitas, dan efisiensi" (Tabel 2; §5).
  Artikel 2 lewat studi ablasi membuktikan penambahan modul attention saja
  menaikkan akurasi 91,0% → 93,2% (Tabel 3). Namun keduanya belum membandingkan
  _antar mekanisme attention_ maupun mengukur latency. Celah inilah yang
  dijawab H3 (G4 + G5).
- **Diuji oleh:** eksperimen E4 vs E6. **Kriteria:** H3 diterima jika
  akurasi/AUC E6 ≥ E4 − 0,5 poin dengan parameter lebih sedikit dan latency
  tidak lebih besar.

Hipotesis tanpa eksperimen hanyalah dugaan. Section 4 menunjukkan setiap
hipotesis diuji oleh eksperimen siapa (E1-E6), diukur dengan metrik apa, dan
buktinya tersimpan di file mana. Dengan begitu tiap klaim dalam dokumen ini dapat
dilacak sampai ke angka.

---

## 4. Penelitian yang akan dijalankan

### 4.1. Desain umum

- **Dataset:** PCOS-XAI Ultrasound Dataset (11.784 citra mentah: 6.784 PCOS,
  5.000 sehat). Sebelum split, salinan byte-identik dibuang via hash md5
  (keep-first) karena split acak level citra bisa menaruh salinan yang sama
  di train sekaligus test dan menggelembungkan metrik. Hasil dedup: 3.996
  citra unik (3.184 PCOS, 812 sehat), tanpa duplikat lintas kelas. Konsekuensi
  rasio kelas menjadi 4:1, sehingga akurasi saja menyesatkan dan specificity,
  F1, serta AUC wajib dilaporkan bersamaan. Ketimpangan ini ditangani secara
  prespecified dan seragam di seluruh eksperimen lewat pos_weight BCE
  (rasio n_negatif/n_positif dari split train saja, tanpa resampling data;
  dapat dinonaktifkan dengan --no-pos-weight untuk keperluan kontrol).
  Split 70/15/15 dengan seed tetap
  (42) untuk split, shuffle, augmentasi, dan inisialisasi model agar seluruh
  eksperimen sebanding dan reprodusibel.
- **Pipeline:** citra USG → deduplikasi (hash md5) → enhancement (CLAHE,
  opsional) → wavelet denoising (BayesShrink) → segmentasi folikel
  unsupervised pada skala kerja tetap 224x224 (Otsu/adaptive + morfologi +
  hapus komponen tepi + fallback persentil), mask di-upscale ke resolusi asli
  → input model (full / ROI crop / masked) → DenseNet-121 +
  Attention → Grad-CAM.
- **Perangkat:** Google Colab GPU T4 (±10 GB VRAM), estimasi ±50-60 menit per
  run 30 epoch. Pemilihan device default `auto`: CUDA dipakai bila tersedia,
  CPU hanya sebagai fallback untuk mesin lokal dengan torch CPU-only. Untuk
  memaksa GPU dan gagal cepat saat runtime tanpa GPU, pakai `--device cuda`.

### 4.2. Matriks eksperimen dan pemetaan hipotesis

| ID  | Preprocessing       | Segmentasi/ROI | Attention      | Menguji         |
| --- | ------------------- | -------------- | -------------- | --------------- |
| E1  | baseline            | none / full    | self-attention | baseline H1     |
| E2  | BayesShrink         | none / full    | self-attention | H1              |
| E3  | BayesShrink + CLAHE | none / full    | self-attention | baseline H2     |
| E4  | BayesShrink + CLAHE | otsu / roi     | self-attention | H2, baseline H3 |
| E5  | BayesShrink + CLAHE | otsu / masked  | self-attention | H2              |
| E6  | BayesShrink + CLAHE | otsu / roi     | cbam           | H3              |

Setiap eksperimen menyimpan checkpoint, plot, `stdout.log`, dan ringkasan
metrik (`experiments/results/results.csv`, termasuk `total_params` dan
`infer_ms_cpu`). Bukti sisi pengolahan citra (PSNR/SSIM, statistik ROI)
dihasilkan tanpa training oleh `experiments/evaluate_quality.py` ke
`experiments/quality/` (`quality.csv`, `summary.txt`, figure overlay).

### 4.3. Hasil eksperimen dan pembacaan hipotesis

Seluruh E1-E6 selesai pada commit `70e0de3` (Colab T4, AMP float16, 30 epoch
maksimum, early stopping patience 10, test 600 citra). Angka lengkap ada di
`experiments/results/results.csv`; verdict dihasilkan
`matrix_summary.json` dengan kriteria yang sudah ditetapkan sebelum run.

| ID  | Accuracy | Precision | Recall | F1     | Specificity | AUC    |
| --- | -------- | --------- | ------ | ------ | ----------- | ------ |
| E1  | 98.17%   | 99.58%    | 98.12% | 98.84% | 98.35%      | 99.65% |
| E2  | 94.67%   | 100.0%    | 93.32% | 96.54% | 100.0%      | 99.70% |
| E3  | 96.00%   | 99.78%    | 95.20% | 97.44% | 99.17%      | 99.56% |
| E4  | 95.00%   | 100.0%    | 93.74% | 96.77% | 100.0%      | 99.74% |
| E5  | 99.17%   | 99.58%    | 99.37% | 99.48% | 98.35%      | 99.72% |
| E6  | 96.00%   | 99.78%    | 95.20% | 97.44% | 99.17%      | 99.55% |

Catatan metodologis dari history per-epoch: early stopping terjadi pada epoch
12-22, sedangkan model terbaik dipilih pada epoch 2-12. Val loss berfluktuasi
cukup besar antar epoch (mis. E3 dari 0,0320 ke 0,1232 lalu 0,0355). Dengan
satu seed, epoch terbaik karena itu mengandung unsur keberuntungan dan tidak
boleh ditafsirkan sebagai konvergensi stabil.

**H1 tidak diterima.** E2 menurunkan accuracy 3,50 poin dan recall 4,80 poin
dibanding E1. AUC memang naik tipis 0,05 poin (0,9970 vs 0,9965), tetapi
regresi accuracy melewati toleransi 0,5 poin yang disepakati. PSNR/SSIM
denoise yang tinggi (51,21 dB; SSIM 0,9963) hanya menunjukkan citra berubah
sedikit, bukan bahwa noise hilang lebih baik, karena tidak ada citra bersih
sebagai ground truth. BayesShrink pada implementasi ini konservatif dan tidak
memberi manfaat klasifikasi yang terukur.

**H2 tidak diterima untuk varian utama E4.** AUC E4 lebih tinggi 0,18 poin
dari E3, tetapi F1 turun 0,67 poin (96,77% vs 97,44%) sehingga melewati
toleransi 0,5 poin. E5 (masked) mencapai hasil terbaik secara deskriptif
(accuracy 99,17%, F1 99,48%, 3 false negative dan 2 false positive), tetapi
E5 tidak boleh dipakai untuk menyelamatkan verdict H2 karena varian utama
sudah diprespesifikasi sebagai E4; memilih E5 setelah melihat test set akan
menjadi seleksi post-hoc. Sinyal E5 tetap layak dilaporkan sebagai temuan
sekunder yang perlu diuji ulang, bukan klaim.

**H3 tidak diterima.** Accuracy E6 setara dalam toleransi (96,00% vs 95,00%)
dan AUC hanya turun 0,20 poin (99,55% vs 99,74%), sementara parameter turun
35,6% (7,35 juta vs 11,42 juta). Namun latency CPU E6 110,07 ms sedikit lebih
besar dari E4 108,73 ms, sehingga kriteria "latency tidak lebih besar" tidak
terpenuhi. Selisih 1,34 ms berada di bawah rentang pengukuran: latency
self-attention pada E1-E5 (arsitektur sama) bergerak dari 117,41 ke 108,20 ms,
jadi 9,2 ms. E6 diukur paling akhir sehingga arah "CBAM tidak lebih cepat"
masih mungkin benar, tetapi besarnya tidak dapat dipisahkan dari noise
pengukuran. Kriteria prespecified tidak diubah setelah melihat hasil; klaim
latency dilaporkan sebagai indikatif, bukan bukti kuat.

**Audit near-duplicate setelah run.** Dedup md5 hanya membuang salinan
byte-identik (7.788 dari 11.784 file). Dengan kemiripan 64x64 grayscale
(korelasi >= 0,995), 57 dari 600 citra test punya near-duplicate di train,
atau 68 bila val ikut dihitung; 11 di antaranya berlabel berlawanan. Di seluruh
dataset, 34 dari 812 citra noninfected unik (4,2%) near-identik (>= 0,9997)
dengan citra infected. Setelah 68 citra test tersebut dikeluarkan, accuracy
menjadi 98,87% (E1), 99,44% (E5), dan 96,62% (E3/E6), sehingga verdict tidak
berubah. Citra near-duplicate berlabel sama diprediksi benar 56/56 oleh E1, E3,
E5, dan E6 (empat model yang diperiksa), sedangkan yang berlabel konflik
menjadi sumber error. Kebocoran ini
karena itu tidak menggelembungkan accuracy agregat, tetapi label noise 4,2%
pada kelas noninfected tetap menjadi batasan validitas.

**ROI dan mask.** Bounding box gabungan region Otsu mencakup lebih dari 95%
frame pada 492 dari 600 citra test (median 100%), sehingga crop ROI praktis
tidak memotong; hanya 3 citra yang crop-nya di bawah 5% frame lalu di-upscale
sekitar 8x. Ini menjelaskan mengapa E4 hampir setara E3. Mask kosong terjadi
pada 24/600 citra test dan coverage di bawah 5% pada 44/600. E5 tetap benar
44/44 pada subset coverage rendah, dan kelima error E5 berada pada coverage
normal, sehingga mask degeneratif bukan penyebab error. Tidak ada guard
coverage minimum; guard itu sengaja tidak ditambahkan setelah melihat hasil
karena akan mengubah definisi E4-E6 dan memerlukan rerun.

**Grad-CAM.** Model biner satu logit membuat implementasi lama selalu
menargetkan logit PCOS. Prediksi Normal karena itu menghasilkan heatmap kosong
setelah ReLU: 3 dari 12 prediksi Normal pada 60 citra test pertama E1, dan
baris Normal pada figure E6 kosong total. Panel tersebut bukan bukti ketiadaan
sinyal. Kode sekarang memakai -logit untuk prediksi Normal, tetapi figure di
`experiments/results/` masih dihasilkan versi lama dan perlu diregenerasi dari
checkpoint.

**Pembacaan klinis.** Pada test set dengan 479 infected dan 121 noninfected,
accuracy didominasi kelas mayoritas. Untuk skrining, false negative lebih
berbahaya: E5 menghasilkan 3 false negative, E1 9, E3/E6 23, dan E2/E4 30-32.
Pola ini menjelaskan mengapa E1 unggul pada accuracy tetapi E5 lebih baik pada
recall dan F1. Karena perbedaan tersebut hanya dari satu split dan satu seed,
temuan ini belum cukup untuk menyimpulkan bahwa masked input lebih unggul.

**Bukti PCD.** PSNR denoise 51,21 dB dan SSIM 0,9963; PSNR CLAHE 17,32 dB dan
SSIM 0,7561. Otsu menghasilkan mask kosong 8/200 (4,0%). Coverage Otsu berbeda
antar kelas (0,285 infected vs 0,385 noninfected), sedangkan adaptive
berlawanan arah (0,352 vs 0,279); cacah region tidak berbeda berarti. Arah yang
tidak konsisten ini menegaskan deskriptor region belum dapat dipakai sebagai
penanda diagnostik PCOS.

### 4.4. Metrik dan keluaran per sudut matakuliah

- **Pengolahan Citra Digital (H1 + metode H2):** PSNR/SSIM, histogram/spektrum,
  overlay mask, statistik ROI (jumlah region, coverage, luas rata-rata dan
  maksimum).
- **Analisis Citra Medis (H1-H3):** Accuracy, Precision, Recall, F1-Score,
  Specificity, AUC, confusion matrix, kurva ROC, heatmap Grad-CAM, serta
  pembahasan false negative/positive (false negative lebih berbahaya untuk
  skrining).
- Pendahuluan, dataset, dan pipeline boleh sama untuk kedua laporan; sudut
  analisis dan kesimpulan dibedakan per matakuliah.

### 4.5. Batasan yang dinyatakan sejak awal

1. Dataset tidak menyediakan mask ground truth → segmentasi bersifat
   unsupervised/pendukung ROI (statistik region + overlay kualitatif, bukan
   Dice klinis).
2. Satu split train/val/test → perbandingan metrik bersifat deskriptif, tanpa
   klaim signifikansi statistik.
3. Split dilakukan pada level citra, bukan level pasien/video. Deduplikasi md5
   hanya membuang salinan byte-identik (7.788 dari 11.784 file); audit
   kemiripan 64x64 menemukan 57/600 citra test yang punya near-duplicate di
   train (68 dengan val), 11 di antaranya berlabel berlawanan, serta 34/812
   citra noninfected unik yang near-identik dengan citra infected. Setelah 68
   citra test tersebut dikeluarkan, accuracy E1/E5/E3-E6 berubah dari
   98,17/99,17/96,00 menjadi 98,87/99,44/96,62 dan verdict tidak berubah. Split
   level pasien tetap perbaikan lanjutan yang disarankan.
4. Setiap eksperimen dijalankan sekali dengan satu seed (42). Variasi antar
   seed tidak diukur, dan toleransi verdict 0,005 berada di dekat rentang
   noise run-to-run; replikasi multi-seed disarankan bila kuota Colab
   memungkinkan. Latency CPU juga diukur 20 forward berurutan dalam satu proses
   tanpa penguncian thread; rentang antar run self-attention 108,20-117,41 ms
   lebih besar dari selisih E6-E4 (1,34 ms), sehingga klaim latency H3
   indikatif.
5. ROI Otsu memakai bounding box gabungan semua region. Pada 492/600 citra test
   kotak itu mencakup lebih dari 95% frame, sehingga crop praktis tidak
   memotong; 3 citra justru di-crop di bawah 5% frame lalu di-upscale. Mask
   kosong (24/600) dikembalikan sebagai citra asli, tetapi mask dengan coverage
   sangat kecil tetap dipakai dan menghasilkan input nyaris hitam pada mode
   masked.
6. Grad-CAM menjelaskan kelas prediksi; pada model satu logit, prediksi Normal
   dijelaskan lewat -logit. Figure Grad-CAM di `experiments/results/`
   dihasilkan sebelum perbaikan ini, sehingga panel Normal pada figure lama
   kosong dan tidak boleh ditafsirkan sebagai tidak adanya bukti.
7. Fusi multimodal (G1), validasi antar dataset (G2), dan kuantifikasi
   ketidakpastian (G3) di luar scope dan dicatat sebagai saran lanjutan.

---

## Daftar pustaka acuan

1. Tiwari, S., Shukla, A., & Sharma, A.K. (2026). Attention-guided
   lightweight deep learning architecture for classification of polycystic
   ovary syndrome from ultrasound images. _Intelligence-Based Medicine_, 13, 100358. https://doi.org/10.1016/j.ibmed.2026.100358
2. Sundari, M.S., et al. (2025). Transfer Learning-Enhanced CNN Model for
   Integrative Ultrasound and Biomarker-Based Diagnosis of Polycystic Ovarian
   Disease. _Scientific Reports_, 15:34519.
   https://doi.org/10.1038/s41598-025-17711-w

---

## Epilog

Penelitian ini tidak dimaksudkan menyaingi kedua artikel pendahulunya.
Keduanya tetap unggul pada apa yang mereka kerjakan. Posisinya melengkapi:
menguji klaim yang mereka tinggalkan setengah jalan: bahwa denoising
membantu, bahwa fokus pada folikel membantu, dan bahwa attention bisa ringan
tanpa kehilangan akurasi. Jika ketiga hipotesis terbukti, klaim-klaim tersebut
berdiri di atas data.
