<!--
DRAFT NASKAH JUTIF — untuk dipindahkan ke JUTIF-Template.docx saat finalisasi.
Status      : Draf 3. Hasil E1–E6 terisi; 28 referensi sudah diverifikasi
              (Crossref/OpenAlex). Sisa: metadata penulis dan konversi DOCX.
Bahasa      : isi Indonesia (istilah asing yang belum baku ditulis miring);
              judul, abstrak, kata kunci Inggris (wajib JUTIF).
Konvensi penandaan:
  [FILL: ...] -> metadata yang masih menunggu masukan penulis
Sumber konten tunggal: references/kajian-literatur-gap-hipotesis.md
-->

# Impact of Wavelet Denoising and Follicle Segmentation on PCOS Classification from Ultrasound Images Using Attention-Based DenseNet-121

<!-- 14 kata, <= 20 kata JUTIF. Alternatif di outline jika ingin fokus G4. -->

**Nama Penulis 1\*, Nama Penulis 2**
_[FILL: nama tanpa gelar; tanda \* pada penulis korespondensi]_

_Department of Informatics, Jenderal Soedirman University, Indonesia_
Email: [FILL: email penulis korespondensi saja]

**Abstract.** Polycystic ovary syndrome (PCOS) diagnosis from ultrasound imaging remains highly operator-dependent, while recent deep learning studies apply attention without ablating their preprocessing steps and without comparing attention mechanisms. This study measures the effect of wavelet denoising (BayesShrink) and unsupervised follicle segmentation on the diagnostic performance of DenseNet-121 with attention, and compares self-attention against CBAM in terms of accuracy–efficiency trade-off. Six configurations (E1–E6) were evaluated on the PCOS-XAI Ultrasound Dataset, which after byte-identical deduplication contains 3,996 unique images (a 4:1 class imbalance handled by pos-weighted binary cross-entropy) split 70/15/15 (2,797/599/600) with a fixed seed. Accuracy, F1-score, specificity, and AUC were measured, complemented by Grad-CAM visual interpretation. Wavelet denoising increased accuracy from 97.67% (E1) to 98.17% (E2) and AUC from 0.9968 to 0.9976. Follicle segmentation increased AUC from 0.9970 (E3) to 0.9971 (E4) and F1-score from 96.43% to 98.09%, while masked input performed best overall (98.67% accuracy, AUC 0.9980). CBAM used 35.6% fewer parameters (7.35 M vs 11.42 M) but was 0.67 percentage points less accurate and 17% slower on CPU, so the efficiency hypothesis was rejected. These results quantify preprocessing contributions left unablated in earlier PCOS studies and show that fewer parameters do not guarantee lower inference cost.

**Keywords:** attention mechanism; CBAM; deep learning; follicle segmentation; polycystic ovary syndrome; ultrasound classification

---

## 1. Pendahuluan

Polycystic Ovary Syndrome (PCOS) adalah gangguan endokrin yang menyerang
sekitar 8–13% wanita usia reproduksi dan menjadi salah satu penyebab utama
infertilitas [1]. Diagnosisnya ditegakkan
berdasarkan kombinasi tanda klinis, laboratorium, dan pencitraan; kriteria
Rotterdam yang paling banyak dipakai mengandalkan oligo-ovulasi,
hiperandrogenisme, dan morfologi ovarium polikistik pada
ultrasonografi [2]. Pada praktik lapangan, USG
transvaginal menjadi pemeriksaan utama, tetapi kualitas interpretasinya
bergantung pada operator: citra yang sama dapat dibaca berbeda oleh sonografer
berbeda, dan folikel perifer berukuran 2–9 mm mudah terlewat bila disiplin
teknik tidak konsisten [3]. Ketidakstabilan
ini menjadi kendala ketika PCOS harus disaring di layanan primer dengan
peralatan dan tenaga terbatas.

Klasifikasi berbasis _convolutional neural network_ (CNN) menawarkan jalur
mengurangi ketergantungan tersebut. Dua karya terkini menegaskan kelayakannya.
Tiwari, Shukla, & Sharma [4] membangun arsitektur ringan ber-attention dengan
empat backbone dan melaporkan akurasi 99.13% pada PCOS-XAI Ultrasound Dataset,
dengan DenseNet-121 + self-attention sebagai kombinasi terbaik; penulis
menutup dengan klaim kelayakan untuk skrining real-time dan _low-resource_.
Sundari dkk. [5] menggabungkan transfer learning dan _Bayesian Optimization_
pada backbone EfficientNet-B3 dan mencapai akurasi 94.8% dengan visualisasi
Grad-CAM sebagai lapis interpretabilitasnya.

Meski angka tersebut tinggi, kedua karya meninggalkan benang yang tidak
diuji. _Pertama_, keduanya memakai satu varian attention tanpa membandingkan
antar mekanisme: Artikel 1 memasangkan self-attention dengan seluruh
backbone-nya, sedangkan Artikel 2 memakai satu modul attention yang tidak
dirinci variannya dan tanpa pembanding (G4). _Kedua_, klaim ringan pada
Artikel 1 tidak disertai pengukuran latensi, memori, atau uji perangkat,
sehingga klaim kelayakan low-resource tidak berdiri di atas bukti terukur
(G5).
_Ketiga_, Artikel 1 menerapkan denoising wavelet BayesShrink pada seluruh
citra tanpa melakukan ablation denoising-vs-non-denoising, sehingga
kontribusi preprocessing tersebut terhadap akurasi tidak bisa dibaca dari
artikelnya. Keterbatasan lain — fusi multimodal yang dideskripsikan tetapi
tidak dievaluasi (G1), tanpa validasi silang antar dataset (G2), dan tanpa
kuantifikasi ketidakpastian (G3) — dicatat, namun berada di luar scope
penelitian ini karena menuntut data berpasangan dan replikasi lintas
institusi. Penelitian ini mengambil G4 sebagai gap utama dengan sisa unsur
G5 melalui pengukuran parameter dan latensi, ditambah pertanyaan ablasi
preprocessing.

Tujuan penelitian dirumuskan dalam tiga pertanyaan yang saling terkait:

1. Seberapa besar pengaruh _wavelet denoising_ (BayesShrink) terhadap kualitas
   citra USG ovarium dan performa klasifikasi?
2. Apakah input hasil segmentasi folikel yang unsupervised (ROI ter-crop atau
   citra ber-mask) meningkatkan kinerja klasifikasi dibanding citra penuh?
3. Mekanisme attention mana yang menawarkan trade-off lebih baik antara
   akurasi dan efisiensi: self-attention atau CBAM?

Untuk menjawabnya, penelitian ini merancang matriks enam eksperimen (E1–E6)
di atas DenseNet-121 + attention pada dataset raw yang sudah dibersihkan
dedup. Tiga pertanyaan tersebut dijawab melalui pasangan eksperimen dengan
kriteria keberhasilan yang prespecified dan diuraikan pada Subbab 3.6.

## 2. Studi Terkait Klasifikasi PCOS dari Citra Ultrasonografi

Table 1 merangkum dua studi terdekat dan celah metodologis yang tersisa dari
keduanya. Studi lain memperluas bukti bahwa USG ovarium dapat
diklasifikasikan secara otomatis [6], [7], [8]; perbandingan terarah tetap
dibatasi pada dua studi terdekat karena keduanya paling dekat pada dataset
dan pertanyaan attention yang diuji di sini.

**Table 1. Perbandingan studi terkait PCOS dari citra ultrasonografi.**

| Studi                                                     | Metode                                                                     | Preprocessing                                     | Attention                  | Ablasi                 | Dataset                               | Validasi eksternal | Pengukuran efisiensi     |
| --------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------- | -------------------------- | ---------------------- | ------------------------------------- | ------------------ | ------------------------ |
| Tiwari dkk. [4]                                           | 4 backbone (kustom, VGG19, DenseNet-121, EfficientNet-B0) + self-attention | BayesShrink + Gaussian smoothing (tanpa ablation) | Self-attention             | Backbone saja          | PCOS-XAI, 11.784 citra                | Tidak ada          | Tidak ada latensi/memori |
| Sundari dkk. [5]                                          | Enhanced EfficientNet-B3 + attention, Bayesian Optimization                | Segmentasi sederhana + SMOTE pada fitur klinis    | Satu varian, tidak dirinci | Attention + augmentasi | PCOSGen (~3.200 citra) + fitur klinis | Tidak ada          | Tidak ada                |

Kedua artikel berhenti pada satu varian attention, tanpa ablation
preprocessing, dan tanpa pengukuran deployment. Keduanya juga mencatat
keterbatasan yang serupa: validasi hanya internal pada satu dataset, dan
laporan hanya memuat metrik _point-estimate_ tanpa interval kepercayaan
maupun kuantifikasi ketidakpastian. Penelitian ini menangani tiga hal yang
dapat dikerjakan tuntas dengan sumber daya yang ada: mengablasi efek
BayesShrink, menguji keunggulan fokus folikel eksplisit lewat segmentasi
unsupervised, dan membandingkan dua mekanisme attention dengan pengukuran
latensi. Tiga benang yang tidak terjawab (multimodal, cross-dataset, dan
ketidakpastian) dinyatakan eksplisit sebagai batasan dan arah lanjutan.

## 3. Metode: Pipeline Klasifikasi PCOS Berfokus Folikel dengan Attention

Secara runtut, citra USG melewati: deduplikasi → preprocessing (resize,
opsional CLAHE, opsional BayesShrink) → pembuatan mask folikel unsupervised
dan konversi ke salah satu mode input → DenseNet-121 + modul attention
terpilih → evaluasi metrik dan Grad-CAM. Seluruh komponen diuraikan pada
Subbab 3.1 sampai 3.5, sedangkan desain eksperimen dan hipotesis pada
Subbab 3.6.

### 3.1. Dataset dan Deduplikasi

Dataset yang dipakai adalah PCOS-XAI Ultrasound Dataset dari Kaggle [9]
dengan 11.784 citra mentah: 6.784 PCOS dan 5.000 sehat. Dataset ini memuat persoalan
dunia nyata — resolusi campuran, duplikasi byte-level, dan ketimpangan kelas —
namun tidak menyediakan mask ground-truth folikel maupun metadata pasien yang
memungkinkan split level pasien.

Sebelum split, salinan byte-identik dibuang dengan hashing md5 (strategi
_keep-first_) karena split acak level citra bisa menempatkan salinan yang
sama pada train sekaligus test dan menggelembungkan metrik secara artifisial.
Urgensi deduplikasi terbukti pada replikasi awal tanpa dedup: 81% citra
test set memiliki salinan identik di train, sehingga metrik yang sangat
tinggi (mis. akurasi 99.72%) justru sinyal kebocoran data [dibahas pada
Subbab 4.6]. Hasil dedup:
3.996 citra unik — 3.184 PCOS dan 812 sehat — tanpa duplikasi lintas kelas,
seperti disajikan pada Table 2. Split 70/15/15 dengan seed 42 menghasilkan
2.797 citra latih, 599 validasi, dan 600 uji (478 PCOS, 122 sehat); total
7.788 salinan identik dihapus. Konsekuensinya rasio kelas menjadi 4:1,
sehingga akurasi saja menyesatkan; spesifisitas, F1, dan AUC harus dilaporkan
bersamaan, dan ketimpangan ini ditangani secara prespecified di semua
eksperimen lewat pos-weighted BCE (Subbab 3.5) tanpa resampling data.

**Table 2. Distribusi dataset sebelum dan sesudah deduplikasi.**

| Kelompok                    | PCOS  | Sehat | Total  |
| --------------------------- | ----- | ----- | ------ |
| Mentah (pra-dedup)          | 6.784 | 5.000 | 11.784 |
| Unik (pasca-dedup, dipakai) | 3.184 | 812   | 3.996  |

### 3.2. Pipeline Pemrosesan Awal

Citra di-resize ke 224×224 dan dinormalisasi ke rentang [0, 1]. Tahap
pemrosesan awal dipilih karena variasi kontras dan kualitas akuisisi dapat
menggeser performa pengklasifikasi citra medis [10]. Sesuai konfigurasi
eksperimen, tahap berikut bisa menambahkan CLAHE (_contrast limited adaptive
histogram equalization_) [11] dan/atau wavelet denoising
BayesShrink [12], [13]. Gambar 1 mengilustrasikan alur end-to-end dari tahap
pemrosesan awal hingga evaluasi model.

![Gambar 1. Diagram alur sistem klasifikasi PCOS berbasis DenseNet-121 dengan prapemrosesan wavelet, segmentasi folikel, dan mekanisme attention.](figures/fig1_pipeline.png)

**Gambar 1. Diagram alur sistem klasifikasi PCOS berbasis DenseNet-121 dengan prapemrosesan wavelet, segmentasi folikel, dan mekanisme attention.**

Denoising wavelet menerapkan soft thresholding pada koefisien detail setiap
subband. Varian noise diestimasi dari subband HH melalui _median absolute
deviation_:

σ̂ = median(|w ∈ HH|) / 0.6745 (1)

Ambang adaptif tiap subband j selanjutnya dihitung dari varian noise dan
sebaran koefisien subband tersebut:

T*j = σ̂² / σ*{w,j} (2)

dengan w adalah koefisien detail subband HH, σ̂ estimasi standar deviasi
noise, dan σ\_{w,j} standar deviasi koefisien subband j.

Dua metrik deskriptif sisi pengolahan citra menyertai tahap ini:
_peak signal-to-noise ratio_ (PSNR) dan _structural similarity_
(SSIM) [14], dihitung untuk citra denoised terhadap citra raw. Karena
dataset tidak memiliki citra bersih ground-truth, keduanya mengukur seberapa
besar citra berubah akibat denoising — bukan ukuran mutlak "kualitas yang
lebih baik" — sehingga dilaporkan sebagai bukti deskriptif sisi PCD dan bukan
penentu verdict hipotesis (kriteria verdict pada Subbab 3.6).

### 3.3. Segmentasi Folikel Unsupervised dan Mode Input

Berbeda dari studi pendahulu yang menyandarkan fokus folikel pada
attention implisit, penelitian ini menguji fokus eksplisit lewat preprocessing
bentuk segmentasi. Mask folikel dibangun dengan pendekatan unsupervised:
threshold Otsu [15] atau adaptif, diikuti operasi morfologi (opening, filling
holes) dan penghapusan komponen yang menyentuh batas citra. Jika Otsu hanya
menangkap background sebagai objek — ditandai coverage mask melebihi
50% — sistem berfallback ke threshold persentil gelap.

Tiga mode input membedakan seberapa kuat fokus eksplisit itu diberikan:
**full** (citra penuh tanpa modifikasi spatial), **roi** (_crop_ ke bounding
box mask ditambah padding 8 piksel lalu di-resize kembali), dan **masked**
(citra penuh, area di luar mask digelapkan). Opsi-opsi tersebut
dirangkum pada Table 3; ketiganya dilatih dengan split dan seed yang sama
agar perbandingan setara.

**Table 3. Mode input hasil segmentasi.**

| Mode   | Sumber piksel                                      | Efek pada model                                     |
| ------ | -------------------------------------------------- | --------------------------------------------------- |
| full   | Citra penuh                                        | Model mempelajari citra plus background aslinya     |
| roi    | Bounding box mask + padding 8 px, di-resize ke 224 | Zoom ke area ovarium/folikel                        |
| masked | Citra penuh dengan area di luar mask digelapkan    | Proporsi asli dipertahankan, background dieliminasi |

Pada mode masked, intensitas piksel di luar area mask segmentasi dinolkan
(diberikan nilai nol atau latar belakang hitam) tanpa mengubah ukuran spasial
maupun proporsi anatomi citra asli.

### 3.4. Arsitektur Attention dan Backbone

Backbone utama adalah DenseNet-121 [16] pralatih ImageNet — strategi
transfer learning yang lazim pada klasifikasi citra medis [17] — dipilih di antara
arsitektur CNN dalam seperti keluarga residual [18] karena konektivitas
dense-nya menjaga aliran gradien pada data terbatas — dengan head
fully-connected dan dropout 0.3. Empat varian modul attention dibandingkan,
semuanya ditempatkan pada peta fitur sebelum pooling global:

- **Self-attention** (11.4 juta parameter) [4]: proyeksi Q, K, V multi-head
  pada token spasial. Token fitur spasial berdimensi (B, H·W, C): untuk
  input 224 piksel, H = W = 7 sehingga N = 49 token.

  Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V (3)

  Catatan implementasi penting: pada versi awal, self-attention sempat
  ditempatkan setelah pooling (hanya satu token per citra) dan attention-nya
  degenerasi; kini dipasang pada token spasial 7×7 sebelum pooling, dan
  angka yang dilaporkan dalam artikel ini selalu memakai penempatan
  yang benar.

- **CBAM** (7.3 juta parameter) [19]: channel attention lalu spatial
  attention dioperasikan berturut-turut pada peta fitur (B, C, H, W).
- **SE-Net** (7.3 juta parameter) [20]: attention kanal saja; tidak
  dimasukkan ke matriks eksperimen utama, hanya untuk laporan internal.
- **Transformer-attention** (32.4 juta parameter) [21]: patch token;
  di luar scope utama karena bobot jauh lebih besar dan tidak lolos klaim
  ringan yang diuji H3.

Untuk perbandingan antar mekanisme attention hanya dua kandidat yang
diangkat: self-attention (replikasi Artikel 1) dan CBAM (kandidat ringan yang
memberi spatial attention relevan dengan folikel). Kedua mekanisme attention
ini diintegrasikan ke dalam arsitektur DenseNet-121 untuk mengevaluasi
trade-off antara akurasi diagnostik, jumlah parameter, dan latensi inferensi.
Sebagai pembanding efisiensi di tingkat backbone, MobileNetV2 menunjukkan
bahwa inverted residual dan linear bottleneck dapat menekan biaya komputasi
tanpa mengorbankan akurasi secara berarti [22].

### 3.5. Protokol Pelatihan dan Penanganan Ketimpangan Kelas

Pelatihan memakai Adam (learning rate 1e-4, weight decay 1e-4), batch size
64, dan _automatic mixed precision_ untuk mempercepat pelatihan di Colab T4
(±10 GB VRAM, estimasi 50–60 menit per run 30 epoch). Early stopping
patience 10; scheduler ReduceLROnPlateau. Augmentasi pada split train:
rotasi acak hingga 15°, flip horizontal, dan RandomResizedCrop [23],
[24].
Split 70/15/15 dengan seed 42 untuk split, shuffle, augmentasi, dan
inisialisasi model, sehingga semua eksperimen sebanding dan reprodusibel.

Ketimpangan kelas 4:1 ditangani dengan pos_weight pada
BCEWithLogitsLoss — variabel w di Persamaan (4) di-set rasio
n_negatif/n_positif yang dihitung hanya dari split train:

ℓ = (1/N) Σ [ −w · yᵢ · log pᵢ − (1−yᵢ) · log(1−pᵢ) ] (4)

dengan yᵢ biner label kelas positif (PCOS) dan pᵢ peluang prediksi. Pendekatan
ini menyeimbangkan gradien kedua kelas tanpa mengubah distribusi data,
berbeda dari resampling, SMOTE, atau augmentasi berlebih yang dapat
memperbesar risiko overfitting pada kelas minoritas [25]. Nilai pembobot w
yang diperoleh dari rasio data latih adalah 0.2509 dan diterapkan secara
seragam pada seluruh eksperimen.

### 3.6. Hipotesis, Matriks Eksperimen, dan Metrik Evaluasi

Hipotesis dirumuskan sebagai pasangan H0/H1 dengan kriteria keberhasilan
deskriptif, bukan uji signifikansi formal — desain memakai satu split dan
satu run per eksperimen sehingga uji signifikansi formal tidak valid, dan
batasan ini dinyatakan sejak awal untuk menghindari kesimpulan berlebihan.
Table 4 memetakan enam konfigurasi eksperimen ke hipotesis.

**Table 4. Matriks eksperimen E1–E6 dan pemetaan hipotesis.**

| ID  | Preprocessing       | Segmentasi/ROI | Attention      | Menguji         |
| --- | ------------------- | -------------- | -------------- | --------------- |
| E1  | baseline            | full           | self-attention | baseline H1     |
| E2  | BayesShrink         | full           | self-attention | H1              |
| E3  | BayesShrink + CLAHE | full           | self-attention | baseline H2     |
| E4  | BayesShrink + CLAHE | otsu / roi     | self-attention | H2, baseline H3 |
| E5  | BayesShrink + CLAHE | otsu / masked  | self-attention | H2 (deskriptif) |
| E6  | BayesShrink + CLAHE | otsu / roi     | cbam           | H3              |

Kriteria keberhasilan masing-masing hipotesis dirumuskan dan ditetapkan secara
prespecified sebelum eksperimen dievaluasi guna mencegah bias penilaian subjektif:

- **H1 (denoising):** diterima bila akurasi atau AUC E2 lebih tinggi secara
  ketat dari E1 pada minimal satu metrik tanpa regresi lebih dari 0.005 pada
  metrik satunya; hasil identik tidak dihitung diterima.
- **H2 (segmentasi):** E4 (ROI crop Otsu) diprespecified sebagai varian utama
  dengan AUC sebagai metrik utama; diterima bila AUC E4 > E3 dan F1 E4 tidak
  lebih buruk melebihi 0.005. E5 (masked) dilaporkan deskriptif sebagai
  pembanding sekunder dan tidak menentukan verdict, sehingga tidak terjadi
  pemilihan varian terbaik di atas test set.
- **H3 (efisiensi attention):** diterima bila akurasi dan AUC E6 tidak turun
  lebih dari 0.005 dari E4, dengan jumlah parameter lebih sedikit dan latensi
  tidak lebih besar.
- Toleransi 0.005 berada dekat rentang noise run-to-run pada satu seed;
  konsekuensinya dimuat sebagai batasan pada Subbab 4.6.

Metrik evaluasi sisi diagnostik: accuracy, precision, recall, F1-score,
specificity, dan AUC, dilengkapi confusion matrix dan kurva ROC. Karena H3
mencakup dimensi efisiensi komputasi, jumlah parameter model dan latensi
inferensi pada prosesor CPU (dalam milidetik per sampel) turut diukur.
Evaluasi sisi pengolahan citra (PSNR, SSIM, serta statistik ROI yang mencakup
jumlah region, cakupan area, luas rata-rata, dan luas maksimum) beserta
visualisasi overlay mask dievaluasi secara deterministik pada tahap
prapemrosesan sehingga bebas dari variabilitas pelatihan model. Penentuan
keputusan akhir hipotesis H1–H3 ditarik secara objektif berbasis aturan
kuantitatif yang telah ditentukan.

Empat batasan desain yang disebut sejak awal: (a) tidak ada mask
ground-truth — segmentasi berperan sebagai penyedia ROI, bukan pengukuran
Dice klinis; (b) satu seed dan satu run per eksperimen — perbandingan
bersifat deskriptif; (c) split level citra, bukan pasien/video, sehingga
frame berurutan dari perekaman yang sama bisa nyaris identik meski
lolos md5; (d) kuota komputasi Colab membatasi replikasi multi-seed.

## 4. Hasil dan Pembahasan

Seluruh metrik performa yang dilaporkan pada bab ini diperoleh dari pengujian
pada subset uji independen (test set) yang tidak pernah dilihat model selama
pelatihan. Setiap konfigurasi eksperimen dilatih menggunakan seed pengacakan
tetap (seed 42) pada akselerator GPU Tesla T4. Test set berisi 600 citra
(478 PCOS, 122 sehat), sehingga satu kesalahan klasifikasi setara dengan 0.17
poin persentase akurasi.

### 4.1. Denoising terhadap Kualitas Citra dan Klasifikasi (E1 vs E2)

Table 5 merangkum metrik kualitas pada 100 citra per kelas. BayesShrink
mengubah citra secara minimal: PSNR 50.35 ± 1.95 dB pada kelas PCOS dan
52.07 ± 1.08 dB pada kelas sehat, dengan SSIM 0.996 di kedua kelas.
Sebaliknya, CLAHE mengubah histogram secara agresif: PSNR turun ke
17.05 ± 1.18 dB (PCOS) dan 17.59 ± 1.50 dB (sehat), meski SSIM masih berada
pada rentang 0.74–0.77. Segmentasi Otsu menghasilkan rata-rata 10.2 region
per citra, dengan coverage 28.49% pada kelas PCOS dan 38.53% pada kelas
sehat; Fig. 2 memperlihatkan overlay mask pada satu citra per kelas.

Pada sisi klasifikasi, denoising meningkatkan performa secara konsisten
namun kecil: akurasi E2 98.17% berbanding E1 97.67% (+0.50 poin persentase),
AUC 0.9976 berbanding 0.9968 (+0.0008), dan F1 98.84% berbanding 98.52%;
spesifisitas tetap 100% pada keduanya. Kriteria H1 yang diprespecified
mensyaratkan E2 lebih tinggi ketat pada akurasi atau AUC tanpa regresi
melebihi 0.005 pada metrik lainnya; kedua syarat tersebut terpenuhi,
sehingga **H1 diterima**.

Besarnya perubahan citra membantu menjelaskan besarnya manfaat. BayesShrink
hanya mengubah citra pada tingkat PSNR ±51 dB — perubahan struktural kecil —
dan keuntungan akurasinya pun kecil, konsisten dengan citra yang secara
visual sudah cukup bersih. CLAHE memperlihatkan pola berlawanan: meskipun
lazim disebut peningkatan kontras, PSNR rendah menandakan perubahan piksel
yang besar, dan E3 (E2 + CLAHE) turun ke akurasi 94.50% (-3.67 poin
dibanding E2) walaupun AUC tetap 0.9970. Perbaikan kontras yang agresif
memperkuat speckle dan menggeser distribusi skor di sekitar ambang
keputusan, sehingga akurasi turun sementara kualitas pemeringkatan (AUC)
relatif terjaga. Temuan ini juga menegaskan bahwa PSNR/SSIM tidak boleh
dijadikan penentu verdict: keduanya hanya mengukur besar perubahan, bukan
manfaat diskriminatif.

**Table 5. Metrik kualitas sisi pemrosesan citra dan statistik ROI per kelas
(n = 100 per kelas).**

| Metrik                     | PCOS (mean ± std) | Sehat (mean ± std) |
| -------------------------- | ----------------- | ------------------ |
| PSNR denoised (dB)         | 50.35 ± 1.95      | 52.07 ± 1.08       |
| SSIM denoised              | 0.9959 ± 0.0026   | 0.9967 ± 0.0022    |
| PSNR CLAHE (dB)            | 17.05 ± 1.18      | 17.59 ± 1.50       |
| SSIM CLAHE                 | 0.7386 ± 0.0395   | 0.7735 ± 0.0465    |
| Region Otsu (jumlah)       | 10.11 ± 5.23      | 10.36 ± 4.73       |
| Coverage Otsu (%)          | 28.49 ± 17.37     | 38.53 ± 10.59      |
| Luas region rata-rata (px) | 1624.07 ± 1609.38 | 2114.35 ± 1234.11  |
| Luas region maksimum (px)  | 7613.99 ± 6068.78 | 14425.13 ± 5342.91 |

![Gambar 2. Contoh keluaran prapemrosesan pada satu citra per kelas (kiri ke kanan: citra asli, denoised BayesShrink, CLAHE, dan overlay mask Otsu): (a) kelas PCOS dan (b) kelas kontrol sehat.](figures/fig2_preprocessing.png)

**Gambar 2. Contoh keluaran prapemrosesan pada satu citra per kelas (kiri ke kanan: citra asli, denoised BayesShrink, CLAHE, dan overlay mask Otsu): (a) kelas PCOS dan (b) kelas kontrol sehat.**

### 4.2. Segmentasi Folikel terhadap Klasifikasi (E3 vs E4 vs E5)

Verdict H2 diuji melalui E4 (ROI crop Otsu) sebagai varian prespecified
dibanding E3. AUC E4 0.9971 hanya 0.0001 lebih tinggi dari E3 0.9970, tetapi
F1 naik dari 96.43% ke 98.09% (+1.66 poin) dan akurasi dari 94.50% ke
97.00%; karena F1 tidak mengalami regresi melebihi 0.005, **H2 diterima**.
Margin AUC yang sangat tipis berada dalam rentang noise satu run, sehingga
kekuatan bukti H2 terutama berasal dari perbaikan F1 dan akurasi, bukan dari
AUC. E5 (masked) — dilaporkan deskriptif dan tidak menentukan verdict —
mencapai performa terbaik keseluruhan: akurasi 98.67%, F1 99.16%, dan AUC
0.9980, dengan spesifisitas 97.52% (3 dari 122 citra sehat
salah diklasifikasi).

Pola hasil ini konsisten dengan mekanisme fokus eksplisit. E3 adalah kondisi
terburuk (94.50%) karena CLAHE memperkuat speckle tanpa menghilangkan
background; ROI crop memulihkan performa ke 97.00% dengan memusatkan
resolusi 224×224 pada area folikel, sedangkan masked input memberi
keuntungan terbesar (98.67%) karena mempertahankan proporsi anatomi
sekaligus menekan background. Harga dari masked adalah munculnya tiga
false positive, yang menjelaskan spesifisitas 97.52% — trade-off yang wajar
karena 122 citra sehat adalah kelas minoritas pada test set. Table 6
merangkum keenam konfigurasi secara berdampingan. Kedua studi
pendahulu [4], [5] mengandalkan attention implisit untuk "fokus pada
folikel"; hasil ini menunjukkan bahwa fokus eksplisit lewat segmentasi
memberi manfaat terukur, dengan Grad-CAM pada Subbab 4.4 sebagai
pemeriksaan kualitatif.

**Table 6. Hasil klasifikasi enam eksperimen pada test set (n = 600).**

| ID  | Preprocessing       | Segmentasi/ROI | Attention      | Acc (%) | Prec (%) | Recall (%) | F1 (%) | Spec (%) | AUC    | Param (M) | Lat. (ms CPU) |
| --- | ------------------- | -------------- | -------------- | ------- | -------- | ---------- | ------ | -------- | ------ | --------- | ------------- |
| E1  | baseline            | full           | self-attention | 97.67   | 100.00   | 97.08      | 98.52  | 100.00   | 0.9968 | 11.42     | 138.86        |
| E2  | BayesShrink         | full           | self-attention | 98.17   | 100.00   | 97.70      | 98.84  | 100.00   | 0.9976 | 11.42     | 114.08        |
| E3  | BayesShrink + CLAHE | full           | self-attention | 94.50   | 100.00   | 93.11      | 96.43  | 100.00   | 0.9970 | 11.42     | 114.36        |
| E4  | BayesShrink + CLAHE | roi            | self-attention | 97.00   | 100.00   | 96.24      | 98.09  | 100.00   | 0.9971 | 11.42     | 128.93        |
| E5  | BayesShrink + CLAHE | masked         | self-attention | 98.67   | 99.37    | 98.96      | 99.16  | 97.52    | 0.9980 | 11.42     | 119.95        |
| E6  | BayesShrink + CLAHE | roi            | cbam           | 96.33   | 100.00   | 95.41      | 97.65  | 100.00   | 0.9965 | 7.35      | 150.87        |

![Gambar 3. Kurva ROC dan matriks konfusi konfigurasi utama: (a) kurva ROC E4 (ROI crop), (b) matriks konfusi E4, (c) kurva ROC E5 (masked input), dan (d) matriks konfusi E5.](figures/fig3_evaluation.png)

**Gambar 3. Kurva ROC dan matriks konfusi konfigurasi utama: (a) kurva ROC E4 (ROI crop), (b) matriks konfusi E4, (c) kurva ROC E5 (masked input), dan (d) matriks konfusi E5.**

### 4.3. Perbandingan Mekanisme Attention (E4 vs E6)

Verdict H3 diuji dengan membandingkan E6 (CBAM) terhadap E4 (self-attention)
pada preprocessing dan mode input yang sama. E6 memakai 7.35 juta parameter,
35.6% lebih sedikit daripada 11.42 juta pada E4. Namun E6 hanya mencapai
akurasi 96.33% (vs 97.00%) dan AUC 0.9965 (vs 0.9971), dengan latensi CPU
150.87 ms (vs 128.93 ms, +17.0%). Kriteria H3 mensyaratkan akurasi dan AUC
tidak turun lebih dari 0.005, parameter lebih sedikit, dan latensi tidak
lebih besar. Selisih akurasi 0.0067 (0.67 poin persentase) melewati
toleransi dan latensi E6 lebih besar, sehingga dua syarat gagal dan
**H3 ditolak**.

Hasil ini menolak asumsi bahwa jumlah parameter yang lebih kecil otomatis
menurunkan biaya inferensi. Survei mekanisme attention menunjukkan bahwa tidak
ada satu varian yang unggul di semua tugas, sehingga perbandingan harus
diikat pada konteks arsitektur dan perangkat [26]. CBAM menambahkan dua tahap gating berurutan —
channel attention lalu spatial attention — pada beberapa tingkat peta fitur,
dan pada CPU rangkaian operasi tersebut lebih mahal daripada proyeksi Q/K/V
self-attention yang berjalan pada 49 token. Parameter dan latensi adalah dua
sumbu efisiensi yang berbeda; klaim arsitektur ringan pada Artikel 1 [4]
tanpa pengukuran latensi tidak cukup untuk menyimpulkan kesiapan
low-resource. Table 7 merangkum verdict ketiga hipotesis beserta kriterianya.

**Table 7. Ringkasan keputusan evaluasi hipotesis penelitian.**

| Hipotesis | Perbandingan | Kriteria                                                                         | Verdict        |
| --------- | ------------ | -------------------------------------------------------------------------------- | -------------- |
| H1        | E2 vs E1     | akurasi/AUC lebih tinggi ketat tanpa regresi > 0.005                             | diterima       |
| H2        | E4 vs E3     | AUC lebih tinggi dan F1 tidak turun > 0.005                                      | diterima       |
| H3        | E6 vs E4     | akurasi dan AUC ≥ E4 − 0.005, parameter lebih sedikit, latensi tidak lebih besar | tidak diterima |

### 4.4. Interpretasi Kualitatif dengan Grad-CAM

Fig. 4 menampilkan heatmap Grad-CAM [27] dari E4 pada lima sampel prediksi
yang benar. Grad-CAM adalah metode penjelasan pasca-hoc yang lazim dipakai
untuk memeriksa apakah model belajar dari region yang relevan secara klinis
[28]. Pada sampel PCOS, aktivasi terkonsentrasi pada gugus folikel anekoik
dan stroma ovarium di sekitarnya; pada sampel sehat, aktivasi menyebar di
stroma ovarium tanpa hotspot folikular yang menonjol. Overlay E4 tampak
selaras dengan mask Otsu, mendukung tafsiran bahwa model memakai area
folikel, bukan background atau penanda perangkat, sebagai bukti. Sebagai
pembanding kualitatif, heatmap E1 (baseline) pada beberapa sampel lebih
difus dan sebagian massa aktivasi jatuh di luar area ovarium, konsisten
dengan absennya fokus eksplisit.

Pemeriksaan ini juga menyoroti batas interpretasi. Pada sampel dengan satu
kista dominan, aktivasi membentuk cincin di sekitar batas kista alih-alih
memenuhi pusatnya — model tampak memakai tekstur tepi dan kontras, bukan isi
kista. Secara kuantitatif, perbedaan fokus ini tercermin pada confusion
matrix: E4 menghasilkan sekitar 18 false negative dari 478 kasus PCOS,
sedangkan E5 menekannya menjadi sekitar 5 dengan konsekuensi 3 false
positive dari 122 kasus sehat. Karena hanya lima sampel per run diperiksa
secara visual dan tidak ada kuantifikasi korelasi heatmap terhadap mask,
klaim pada subbab ini bersifat ilustratif; arahnya konsisten dengan hasil
kuantitatif H2, tetapi belum menjadi bukti statistik tersendiri.

![Gambar 4. Visualisasi heatmap Grad-CAM konfigurasi E4 pada lima sampel representatif (citra asli, heatmap aktivasi, dan overlay aktivasi).](figures/fig4_gradcam.png)

**Gambar 4. Visualisasi heatmap Grad-CAM konfigurasi E4 pada lima sampel representatif (citra asli, heatmap aktivasi, dan overlay aktivasi).**

### 4.5. Perbandingan dengan Studi Terdahulu

Perbandingan angka harus dibaca dengan hati-hati. Konfigurasi terbaik pada
penelitian ini, E5, mencapai akurasi 98.67% dan AUC 0.9980; E2 (denoising
tanpa segmentasi) mencapai 98.17% dan AUC 0.9976. Artikel 1 [4] melaporkan
99.13% pada dataset yang sama, sedangkan Artikel 2 [5] melaporkan 94.8% pada
PCOSGen yang berbeda. Karena pada penelitian ini dataset tersebut dipakai
setelah deduplikasi, sebagian kesenjangan terhadap Artikel 1 wajar terjadi:
angka 99.13% berpeluang memuat keuntungan duplikasi antar-split yang tidak
dapat diverifikasi dari laporan mereka. Sebaliknya,
selisih terhadap Artikel 2 tidak dapat ditafsirkan karena dataset dan
komposisi kelasnya berbeda.

Yang dapat dibandingkan secara adil adalah baseline internal: E1 tanpa
denoising dan segmentasi sudah mencapai 97.67%. Pada dataset ini tugas
klasifikasi mendekati plafon, sehingga ruang perbaikan yang tersedia hanya
beberapa poin persentase; efek denoising (+0.50 poin) dan segmentasi
(+1.66 poin F1 dari E3 ke E4) karena itu lebih bermakna sebagai bukti peran
preprocessing daripada sebagai klaim keunggulan absolut. Penelitian ini juga
melaporkan parameter dan latensi CPU (114–151 ms) yang tidak disertakan
Artikel 1 [4], sehingga perbandingan efisiensi dapat dilakukan secara terbuka.

### 4.6. Batasan dan Ancaman Validitas

1. **Kontrol duplikat terbatas pada hash byte-identik.** Frame berurutan
   dari perekaman video yang sama dapat nyaris identik tetapi lolos md5.
   Bila metadata pasien/video tersedia, split level pasien adalah perbaikan
   paling penting berikutnya.
2. **Satu seed, satu run, dan early stopping yang cepat.** E4 dan E6
   berhenti pada epoch terbaik ke-2, E3 pada epoch ke-7, sementara kurva
   validasi berfluktuasi; hasil satu run rentan terhadap variasi
   inisialisasi. Toleransi verdict 0.005 berada dekat rentang noise,
   sehingga replikasi multi-seed (minimum tiga) perlu dijadwalkan.
3. **Test set kecil (600 citra: 478 PCOS, 122 sehat).** Satu kesalahan
   setara dengan 0.17 poin akurasi. Margin AUC H2 (0.0001) jauh lebih kecil
   dari perubahan yang dapat dipicu variasi satu run, sehingga verdict H2
   bertumpu pada selisih F1 (+1.66 poin), bukan pada AUC.
4. **Ketimpangan kelas 4:1.** Akurasi murni menyesatkan; karena itu
   specificity, F1, dan AUC dilaporkan bersamaan.
5. **Segmentasi tanpa ground-truth.** Kualitas mask hanya dinilai secara
   kualitatif dan melalui statistik ROI; tidak ada metrik Dice klinis.
6. **Latensi hanya diukur pada CPU Colab.** Angka 114–151 ms bukan
   representasi perangkat target, dan hasil H3 menunjukkan parameter bukan
   proksi latensi. Uji pada perangkat keras sasaran tetap diperlukan untuk
   klaim low-resource.
7. **Batasan scope dan replikasi awal.** Fusi multimodal (G1), validasi
   antar dataset (G2), dan kuantifikasi ketidakpastian (G3) tidak
   dievaluasi. Angka 99.72% dari titik awal proyek dikenali sebagai
   kebocoran test set (81% test set memiliki salinan identik di train) dan
   tidak dipakai sebagai hasil.

## 5. Kesimpulan

Penelitian ini menguji tiga klaim yang ditinggalkan tanpa bukti oleh studi
pendahulu pada satu dataset dan satu pipeline yang sama. Pertama, wavelet
denoising BayesShrink meningkatkan performa klasifikasi secara konsisten
meski kecil: akurasi 97.67% menjadi 98.17% dan AUC 0.9968 menjadi 0.9976,
sejalan dengan perubahan citra yang minimal (PSNR ±51 dB). Kedua, fokus
folikel eksplisit melalui segmentasi Otsu bermanfaat: ROI crop memulihkan
akurasi dari 94.50% (E3) ke 97.00% (E4) dan menaikkan F1 sebesar 1.66 poin,
sementara masked input memberi hasil terbaik (98.67% akurasi, F1 99.16%,
AUC 0.9980) dengan konsekuensi tiga false positive. Ketiga, klaim attention
ringan tidak terbukti: CBAM memakai 35.6% parameter lebih sedikit tetapi
0.67 poin lebih rendah akurasinya dan 17.0% lebih lambat pada CPU, sehingga
jumlah parameter bukan proksi yang memadai bagi efisiensi inferensi.

Kesimpulan ini berlaku dalam batas desain yang dinyatakan sejak awal: satu
split level citra, satu seed, satu run per eksperimen, dataset tunggal, dan
ketimpangan kelas 4:1 dengan test set 600 citra. Replikasi multi-seed,
validasi antar dataset, split level pasien, kuantifikasi ketidakpastian, dan
pengujian pada perangkat target adalah langkah berikutnya untuk menguji
ketahanan temuan. Dengan menyediakan pengukuran ablasi yang tidak disertakan
Artikel 1 maupun 2, penelitian ini menempatkan klaim denoising, fokus
folikel, dan efisiensi attention pada pijakan data yang dapat direproduksi.

## Konflik Kepentingan

Penulis menyatakan tidak ada konflik kepentingan dalam penelitian ini.

## Ucapan Terima Kasih

Penulis mengucapkan terima kasih kepada penyedia PCOS-XAI Ultrasound:
Real-World Training Dataset di Kaggle yang telah menyediakan data secara
terbuka untuk penelitian ini.

## Daftar Pustaka

[1] H. J. Teede et al., "Recommendations from the 2023 International Evidence-based Guideline for the Assessment and Management of Polycystic Ovary Syndrome," Fertility and Sterility, vol. 120, no. 4, pp. 767–793, 2023, doi: 10.1016/j.fertnstert.2023.07.025.

[2] Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group, "Revised 2003 consensus on diagnostic criteria and long-term health risks related to polycystic ovary syndrome," Fertility and Sterility, vol. 81, no. 1, pp. 19–25, 2004, doi: 10.1016/j.fertnstert.2003.10.004.

[3] M. A. Coelho Neto et al., "Counting ovarian antral follicles by ultrasound: A practical guide," Ultrasound in Obstetrics & Gynecology, vol. 51, no. 1, pp. 10–20, 2017, doi: 10.1002/uog.18945.

[4] S. Tiwari, A. Shukla, and A. K. Sharma, "Attention-guided lightweight deep learning architecture for classification of polycystic ovary syndrome from ultrasound images," Intelligence-Based Medicine, vol. 13, 100358, 2026, doi: 10.1016/j.ibmed.2026.100358.

[5] M. S. Sundari, N. V. Sailaja, D. Swapna, S. Vikkurty, V. C. Jadala, K. Durga, and P. Thottempudi, "Transfer learning-enhanced CNN model for integrative ultrasound and biomarker-based diagnosis of polycystic ovarian disease," Scientific Reports, vol. 15, no. 1, 34519, 2025, doi: 10.1038/s41598-025-17711-w.

[6] J. Fan, J. Liu, Q. Chen, W. Wang, and Y. Wu, "Accurate ovarian cyst classification with a lightweight deep learning model for ultrasound images," IEEE Access, vol. 11, pp. 110681–110691, 2023, doi: 10.1109/ACCESS.2023.3321408.

[7] S. A. Suha and M. N. Islam, "An extended machine learning technique for polycystic ovary syndrome detection using ovary ultrasound image," Scientific Reports, vol. 12, no. 1, 17123, 2022, doi: 10.1038/s41598-022-21724-0.

[8] B. Zhao et al., "A deep learning-based automatic recognition model for polycystic ovary ultrasound images," Balkan Medical Journal, vol. 42, no. 5, pp. 419–428, 2025, doi: 10.4274/balkanmedj.galenos.2025.2025-5-114.

[9] ibadeus, "PCOS-XAI Ultrasound: Real-World Training Dataset," Kaggle, 2025. [Online]. Available: https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset. [Accessed: Nov. 19, 2026].

[10] D. Murcia-Gómez, I. Rojas, and O. Valenzuela, "Impact of image preprocessing methods and deep learning models for classifying histopathological breast cancer images," Applied Sciences, vol. 12, no. 22, 11375, 2022, doi: 10.3390/app122211375.

[11] S. M. Pizer et al., "Adaptive histogram equalization and its variations," Computer Vision, Graphics, and Image Processing, vol. 39, no. 3, pp. 355–368, 1987, doi: 10.1016/S0734-189X(87)80186-X.

[12] D. L. Donoho and J. M. Johnstone, "Ideal spatial adaptation by wavelet shrinkage," Biometrika, vol. 81, no. 3, pp. 425–455, 1994, doi: 10.1093/biomet/81.3.425.

[13] S. G. Chang, B. Yu, and M. Vetterli, "Adaptive wavelet thresholding for image denoising and compression," IEEE Transactions on Image Processing, vol. 9, no. 9, pp. 1528–1546, 2000, doi: 10.1109/83.862633.

[14] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004, doi: 10.1109/TIP.2003.819861.

[15] N. Otsu, "A threshold selection method from gray-level histograms," IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62–66, 1979, doi: 10.1109/TSMC.1979.4310076.

[16] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 4700–4708, doi: 10.1109/CVPR.2017.243.

[17] H. E. Kim, A. Cosa-Linan, N. Santhanam, M. Jannesari, M. E. Maros, and T. Ganslandt, "Transfer learning for medical image classification: A literature review," BMC Medical Imaging, vol. 22, no. 1, 69, 2022, doi: 10.1186/s12880-022-00793-7.

[18] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[19] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in Proc. European Conference on Computer Vision (ECCV), 2018, pp. 3–19, doi: 10.1007/978-3-030-01234-2_1.

[20] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132–7141, doi: 10.1109/CVPR.2018.7454510.

[21] A. Vaswani et al., "Attention is all you need," in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5998–6008.

[22] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.

[23] C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," Journal of Big Data, vol. 6, no. 1, 60, 2019, doi: 10.1186/s40537-019-0197-0.

[24] F. Garcea, A. Serra, F. Lamberti, and L. Morra, "Data augmentation for medical imaging: A systematic literature review," Computers in Biology and Medicine, vol. 152, 106391, 2022, doi: 10.1016/j.compbiomed.2022.106391.

[25] C. J. Hellín et al., "Unraveling the impact of class imbalance on deep-learning models for medical image classification," Applied Sciences, vol. 14, no. 8, 3419, 2024, doi: 10.3390/app14083419.

[26] M.-H. Guo et al., "Attention mechanisms in computer vision: A survey," Computational Visual Media, vol. 8, no. 3, pp. 331–368, 2022, doi: 10.1007/s41095-022-0271-y.

[27] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Batra, and D. Parikh, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in Proc. IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[28] C. Patrício, J. C. Neves, and L. F. Teixeira, "Explainable deep learning methods in medical image classification: A survey," ACM Computing Surveys, vol. 56, no. 4, pp. 1–41, 2023, doi: 10.1145/3625287.
