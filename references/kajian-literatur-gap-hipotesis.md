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

> *Attention-guided lightweight deep learning architecture for classification
> of polycystic ovary syndrome from ultrasound images.*
> Intelligence-Based Medicine, 13, 100358.

**Masalah yang diangkat.** Arsitektur CNN konvensional sulit menangkap
*long-range contextual dependencies* pada citra USG grayscale, sehingga
diskriminasi folikel ovarium menjadi kurang andal. Di sisi lain, model yang
akurat umumnya berat dan tidak cocok untuk layanan primer atau aplikasi
kesehatan seluler dengan sumber daya terbatas.

**Metode.** Empat backbone (CNN kustom, VGG19, DenseNet-121, EfficientNet-B0)
masing-masing dipadukan modul *self-attention* (proyeksi Q, K, V dengan
*scaled dot-product attention*). Preprocessing memakai denoising wavelet
metode BayesShrink dilanjutkan Gaussian smoothing; augmentasi berupa rotasi,
*flip* horizontal, dan *zoom*. Pelatihan memakai optimizer Adam dengan
*binary cross-entropy loss*. Dataset yang dipakai adalah PCOS-XAI Ultrasound
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
arsitektur ringan ber-attention layak untuk skrining *real-time* dan
*low-resource*. Penulis menutup dengan peluang pendekatan multimodal
sebagai pekerjaan lanjutan.

### 1.2. Artikel 2: Sundari et al. (2025)

> *Transfer Learning-Enhanced CNN Model for Integrative Ultrasound and
> Biomarker-Based Diagnosis of Polycystic Ovarian Disease.*
> Scientific Reports, 15:34519.

**Masalah yang diangkat.** Interpretasi USG transvaginal bersifat subjektif dan
bergantung pada operator, sehingga akurasi antar sonografer bervariasi.
Studi-studi terdahulu cenderung terbatas pada dataset kecil, minim optimasi model,
dan minim integrasi data klinis multimodal.

**Metode.** Backbone Enhanced EfficientNet-B3 dengan modul attention,
*batch normalization*, dan dropout (0,35, ditemukan melalui Bayesian
Optimization). Hyperparameter (learning rate, batch size, dropout, pilihan
optimizer, unit fully-connected) disetel dengan Bayesian Optimization.
Ketidakseimbangan kelas ditangani dengan SMOTE pada fitur klinis tabular
(memakai pustaka `imbalanced-learn`), augmentasi geometris, dan
interpretabilitas disediakan lewat visualisasi Grad-CAM. Dataset memakai
PCOSGen (~3.200 citra berlabel) beserta metadata klinis (LH, FSH, AMH).

**Catatan umum kedua artikel:** keduanya hanya melaporkan metrik
*point-estimate* tunggal (akurasi, AUC, dan sejenisnya) tanpa interval
kepercayaan, kalibrasi, maupun kuantifikasi ketidakpastian prediksi.

**Hasil dan klaim.** Akurasi 94,8% (sensitivitas 93,2%; spesifisitas 95,5%;
presisi 94,0%; F1 93,6%; AUC 0,97). Studi ablasi menunjukkan penambahan modul
attention saja menaikkan akurasi 91,0% → 93,2%, dan kombinasi
attention + augmentasi mencapai 94,8%. **Catatan penting yang dinyatakan
penulis sendiri:** data biomarker (LH, FSH, AMH) *tidak* dimasukkan ke model
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

| No | Gap | Bukti dari artikel |
|----|-----|--------------------|
| G1 | Fusi multimodal citra + biomarker dideskripsikan tetapi **tidak diimplementasikan/dievaluasi** | Artikel 2 §4.1: biomarker tidak masuk model final; integrasi penuh dinyatakan sebagai *subsequent work* |
| G2 | **Tanpa validasi silang antar dataset;** keduanya hanya validasi internal pada satu dataset | Artikel 1 memakai PCOS-XAI saja; Artikel 2 memakai PCOSGen saja |
| G3 | **Tanpa kuantifikasi ketidakpastian;** output hanya prediksi biner tanpa interval kepercayaan | Tidak ada MC Dropout, ensemble, maupun kalibrasi pada keduanya |
| G4 | **Tanpa perbandingan sistematis mekanisme attention;** masing-masing memakai satu varian | Artikel 1: satu modul SA; Artikel 2: modul attention tidak dirinci variannya dan tanpa pembanding |
| G5 | Klaim **lightweight/low-resource tanpa benchmark deployment** (latency, memori, uji perangkat) | Artikel 1 mengklaim cocok untuk *portable ultrasound* tanpa pengukuran latency/inferensi |

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
- **Diuji oleh:** eksperimen E1 vs E2. **Kriteria:** H1 diterima jika rata-rata
  PSNR/SSIM E2 > E1 dan akurasi/AUC E2 ≥ E1.

### H2: Segmentasi/ROI folikel meningkatkan akurasi klasifikasi

- **H0:** Input ROI/*masked* dari segmentasi folikel unsupervised tidak
  meningkatkan akurasi dibanding citra penuh.
- **H1:** Input ROI/*masked* meningkatkan akurasi/F1/AUC karena model fokus
  pada area folikel dan kurang terpengaruh background.
- **Landasan dari literatur:** Artikel 1 menyatakan modul self-attention
  membuat model "fokus pada area folikular dan ovarial yang salient"
  (Pendahuluan; §3.2) dan diskusinya mengonfirmasi model berfokus pada
  "*cystic follicles and abnormal ovarian texture instead of background
  noise*" (§5). Artikel 2 menyatakan modul attention "menekankan region
  spasial yang relevan dan menekan background noise" untuk mendeteksi
  "*follicular clusters and cystic formations*" (§3.1.3). Kriteria Rotterdam
  sendiri berbasis jumlah folikel/volume ovarium (§1). Kedua artikel
  mengandalkan *attention implisit* untuk fokus folikel, tetapi tidak pernah
  menguji *fokus eksplisit* lewat segmentasi/ROI. Celah inilah yang diuji H2.
- **Diuji oleh:** eksperimen E3 vs E4 vs E5. **Kriteria:** H2 diterima jika
  F1/AUC E4 atau E5 > E3.

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
  *antar mekanisme attention* maupun mengukur latency. Celah inilah yang
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

- **Dataset:** PCOS-XAI Ultrasound Dataset (11.784 citra: 6.784 PCOS, 5.000
  sehat), split 70/15/15 dengan seed tetap (42) agar seluruh eksperimen
  sebanding.
- **Pipeline:** citra USG → enhancement (CLAHE, opsional) → wavelet denoising
  (BayesShrink) → segmentasi folikel unsupervised (Otsu/adaptive + morfologi +
  hapus komponen tepi + fallback persentil) → input model (full / ROI crop /
  masked) → DenseNet-121 + Attention → Grad-CAM.
- **Perangkat:** Google Colab GPU T4 (±10 GB VRAM), estimasi ±50-60 menit per
  run 30 epoch.

### 4.2. Matriks eksperimen dan pemetaan hipotesis

| ID | Preprocessing | Segmentasi/ROI | Attention | Menguji |
|----|---------------|----------------|-----------|---------|
| E1 | baseline | none / full | self-attention | baseline H1 |
| E2 | BayesShrink | none / full | self-attention | H1 |
| E3 | BayesShrink + CLAHE | none / full | self-attention | baseline H2 |
| E4 | BayesShrink + CLAHE | otsu / roi | self-attention | H2, baseline H3 |
| E5 | BayesShrink + CLAHE | otsu / masked | self-attention | H2 |
| E6 | BayesShrink + CLAHE | otsu / roi | cbam | H3 |

Setiap eksperimen menyimpan checkpoint, plot, `stdout.log`, dan ringkasan
metrik (`experiments/results/results.csv`, termasuk `total_params` dan
`infer_ms_cpu`). Bukti sisi pengolahan citra (PSNR/SSIM, statistik ROI)
dihasilkan tanpa training oleh `experiments/evaluate_quality.py` ke
`experiments/quality/` (`quality.csv`, `summary.txt`, figure overlay).

### 4.3. Metrik dan keluaran per sudut matakuliah

- **Pengolahan Citra Digital (H1 + metode H2):** PSNR/SSIM, histogram/spektrum,
  overlay mask, statistik ROI (jumlah region, coverage, luas rata-rata dan
  maksimum).
- **Analisis Citra Medis (H1-H3):** Accuracy, Precision, Recall, F1-Score,
  Specificity, AUC, confusion matrix, kurva ROC, heatmap Grad-CAM, serta
  pembahasan false negative/positive (false negative lebih berbahaya untuk
  skrining).
- Pendahuluan, dataset, dan pipeline boleh sama untuk kedua laporan; sudut
  analisis dan kesimpulan dibedakan per matakuliah.

### 4.4. Batasan yang dinyatakan sejak awal

1. Dataset tidak menyediakan mask ground truth → segmentasi bersifat
   unsupervised/pendukung ROI (statistik region + overlay kualitatif, bukan
   Dice klinis).
2. Satu split train/val/test → perbandingan metrik bersifat deskriptif, tanpa
   klaim signifikansi statistik.
3. Fusi multimodal (G1), validasi antar dataset (G2), dan kuantifikasi
   ketidakpastian (G3) di luar scope dan dicatat sebagai saran lanjutan.

---

## Daftar pustaka acuan

1. Tiwari, S., Shukla, A., & Sharma, A.K. (2026). Attention-guided
   lightweight deep learning architecture for classification of polycystic
   ovary syndrome from ultrasound images. *Intelligence-Based Medicine*, 13,
   100358. https://doi.org/10.1016/j.ibmed.2026.100358
2. Sundari, M.S., et al. (2025). Transfer Learning-Enhanced CNN Model for
   Integrative Ultrasound and Biomarker-Based Diagnosis of Polycystic Ovarian
   Disease. *Scientific Reports*, 15:34519.
   https://doi.org/10.1038/s41598-025-17711-w

---

## Epilog

Penelitian ini tidak dimaksudkan menyaingi kedua artikel pendahulunya.
Keduanya tetap unggul pada apa yang mereka kerjakan. Posisinya melengkapi:
menguji klaim yang mereka tinggalkan setengah jalan: bahwa denoising
membantu, bahwa fokus pada folikel membantu, dan bahwa attention bisa ringan
tanpa kehilangan akurasi. Jika ketiga hipotesis terbukti, klaim-klaim tersebut
berdiri di atas data.
