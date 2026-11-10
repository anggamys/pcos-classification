<!--
DRAFT NASKAH JUTIF — untuk dipindahkan ke JUTIF-Template.docx saat finalisasi.
Status      : Draf 1. Bagian hasil menunggu matriks E1–E6 selesai di Colab.
Bahasa      : isi Indonesia (istilah asing yang belum baku ditulis miring);
              judul, abstrak, kata kunci Inggris (wajib JUTIF).
Konvensi penandaan:
  [FILL: ...] -> tempat menunggu data eksperimen (results.csv / summary.json)
  [REF: ...]  -> slot referensi yang HARUS dicari dan diverifikasi
                 (jangan mengarang entri; target total >= 25)
Sumber konten tunggal: references/kajian-literatur-gap-hipotesis.md
-->

# Impact of Wavelet Denoising and Follicle Segmentation on PCOS Classification from Ultrasound Images Using Attention-Based DenseNet-121

<!-- 14 kata, <= 20 kata JUTIF. Alternatif di outline jika ingin fokus G4. -->

**Nama Penulis 1\*, Nama Penulis 2**
_[FILL: nama tanpa gelar; tanda \* pada penulis korespondensi]_

_Department of Informatics, Jenderal Soedirman University, Indonesia_
Email: [FILL: email penulis korespondensi saja]

**Abstrak.** Diagnosis polycystic ovary syndrome (PCOS) dari ultrasonografi transvaginal sangat bergantung pada pengalaman operator, sementara model deep learning terkini menerapkan attention tanpa mengablasi langkah preprocessing dan tanpa membandingkan mekanisme attention antar varian. Penelitian ini mengukur pengaruh wavelet denoising (BayesShrink) dan segmentasi folikel unsupervised terhadap performa diagnostik DenseNet-121 + attention, serta membandingkan self-attention dengan CBAM dari sisi trade-off akurasi–efisiensi. Studi ablasi enam konfigurasi (E1–E6) dilakukan pada PCOS-XAI Ultrasound Dataset yang pasca-deduplikasi berisi 3.996 citra unik (ketimpangan kelas 4:1 ditangani pos-weighted binary cross-entropy), dengan split 70/15/15 dan seed tetap. Evaluasi memakai akurasi, F1-score, spesifisitas, dan AUC, dilengkapi interpretasi visual Grad-CAM. [FILL: 1 kalimat hasil utama E1–E6 + verdict H1–H3]. [FILL: 1 kalimat temuan/implikasi: nilai preprocessing eksplisit dan pemilihan mekanisme attention yang efisien untuk skrining low-resource].

**Keywords:** attention mechanism; CBAM; deep learning; follicle segmentation; polycystic ovary syndrome; ultrasound classification

---

_(Dari sini isi naskah berbahasa Indonesia. Saat migrasi ke DOCX: Times New
Roman 11 pt, spasi 1,15, satu kolom; abstrak spasi 1.)_

## 1. Pendahuluan

Polycystic Ovary Syndrome (PCOS) adalah gangguan endokrin yang menyerang
sekitar 8–13% wanita usia reproduksi dan menjadi salah satu penyebab utama
infertilitas [REF: epidemiologi PCOS terbaru, 2021+]. Diagnosisnya ditegakkan
berdasarkan kombinasi tanda klinis, laboratorium, dan pencitraan; kriteria
Rotterdam yang paling banyak dipakai mengandalkan oligo-ovulasi,
hiperandrogenisme, dan morfologi ovarium polikistik pada
ultrasonografi [REF: kriteria Rotterdam]. Pada praktik lapangan, USG
transvaginal menjadi pemeriksaan utama, tetapi kualitas interpretasinya
bergantung pada operator: citra yang sama dapat dibaca berbeda oleh sonografer
berbeda, dan folikel perifer berukuran 2–9 mm mudah terlewat bila disiplin
teknik tidak konsisten [REF: variabilitas antar-operator USG]. Ketidakstabilan
ini menjadi kendala ketika PCOS harus disaring di layanan primer dengan
peralatan dan tenaga terbatas.

Klasifikasi berbasis _convolutional neural network_ (CNN) menawarkan jalur
mengurangi ketergantungan tersebut. Dua karya terkini menegaskan kelayakannya.
Tiwari, Shukla, & Sharma [1] membangun arsitektur ringan ber-attention dengan
empat backbone dan melaporkan akurasi 99,13% pada PCOS-XAI Ultrasound Dataset,
dengan DenseNet-121 + self-attention sebagai kombinasi terbaik; penulis
menutup dengan klaim kelayakan untuk skrining real-time dan _low-resource_.
Sundari dkk. [2] menggabungkan transfer learning dan _Bayesian Optimization_
pada backbone EfficientNet-B3 dan mencapai akurasi 94,8% dengan visualisasi
Grad-CAM sebagai lapis interpretabilitasnya.

Meski angka tersebut tinggi, kedua karya meninggalkan benang yang tidak
diuji. _Pertama_, keduanya memakai satu varian attention tanpa membandingkan
antar mekanisme: Artikel 1 memasangkan self-attention dengan seluruh
backbone-nya, sedangkan Artikel 2 memakai satu modul attention yang tidak
dirinci variannya dan tanpa pembanding [G4]. _Kedua_, klaim ringan pada
Artikel 1 tidak disertai pengukuran latensi, memori, atau uji perangkat,
sehingga klaim kelayakan low-resource tidak berdiri di atas bukti terukur
[G5].
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
keduanya. Baris tambahan dari studi pustaka lain akan dilengkapi setelah
pencarian referensi blok pendukung selesai.

**Table 1. Perbandingan studi terkait PCOS dari citra ultrasonografi
[FILL: tambah baris dari referensi blok pendukung].**

| Studi                                                     | Metode                                                                     | Preprocessing                                     | Attention                  | Ablasi                 | Dataset                               | Validasi eksternal | Pengukuran efisiensi     |
| --------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------- | -------------------------- | ---------------------- | ------------------------------------- | ------------------ | ------------------------ |
| Tiwari dkk. [1]                                           | 4 backbone (kustom, VGG19, DenseNet-121, EfficientNet-B0) + self-attention | BayesShrink + Gaussian smoothing (tanpa ablation) | Self-attention             | Backbone saja          | PCOS-XAI, 11.784 citra                | Tidak ada          | Tidak ada latensi/memori |
| Sundari dkk. [2]                                          | Enhanced EfficientNet-B3 + attention, Bayesian Optimization                | Segmentasi sederhana + SMOTE pada fitur klinis    | Satu varian, tidak dirinci | Attention + augmentasi | PCOSGen (~3.200 citra) + fitur klinis | Tidak ada          | Tidak ada                |
| [REF: CNN klasifikasi USG ovarium 2022–2025]              | –                                                                          | –                                                 | –                          | –                      | –                                     | –                  | –                        |
| [REF: transfer learning klasifikasi ginekologi 2022–2025] | –                                                                          | –                                                 | –                          | –                      | –                                     | –                  | –                        |

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

Dataset yang dipakai adalah PCOS-XAI Ultrasound Dataset dari Kaggle dengan
11.784 citra mentah: 6.784 PCOS dan 5.000 sehat. Dataset ini memuat persoalan
dunia nyata — resolusi campuran, duplikasi byte-level, dan ketimpangan kelas —
namun tidak menyediakan mask ground-truth folikel maupun metadata pasien yang
memungkinkan split level pasien.

Sebelum split, salinan byte-identik dibuang dengan hashing md5 (strategi
_keep-first_) karena split acak level citra bisa menempatkan salinan yang
sama pada train sekaligus test dan menggelembungkan metrik secara artifisial.
Urgensi deduplikasi terbukti pada replikasi awal tanpa dedup: 81% citra
test set memiliki salinan identik di train, sehingga metrik yang sangat
tinggi (mis. akurasi 99,72%) justru sinyal kebocoran data [dibahas pada
Subbab 4.6]. Hasil dedup:
3.996 citra unik — 3.184 PCOS dan 812 sehat — tanpa duplikasi lintas kelas,
seperti disajikan pada Table 2. Konsekuensinya rasio kelas menjadi 4:1,
sehingga akurasi saja menyesatkan; spesifisitas, F1, dan AUC harus dilaporkan
bersamaan, dan ketimpangan ini ditangani secara prespecified di semua
eksperimen lewat pos-weighted BCE (Subbab 3.5) tanpa resampling data.

**Table 2. Distribusi dataset sebelum dan sesudah deduplikasi.**

| Kelompok                    | PCOS  | Sehat | Total  |
| --------------------------- | ----- | ----- | ------ |
| Mentah (pra-dedup)          | 6.784 | 5.000 | 11.784 |
| Unik (pasca-dedup, dipakai) | 3.184 | 812   | 3.996  |

### 3.2. Pipeline Pemrosesan Awal

Citra di-resize ke 224×224 dan dinormalisasi ke [0,1]. Sesuai konfigurasi
eksperimen, tahap berikut bisa menambahkan CLAHE (_contrast limited adaptive
histogram equalization_) [REF: Pizer dkk., CLAHE] dan/atau wavelet denoising
BayesShrink [9], [10]. Fig. 1 mengilustrasikan alur end-to-end sampai evaluasi.

_(FIGUR 1: diagram pipeline end-to-end. Harus dibuat sebagai gambar berbahasa
Inggris, min. 300 dpi: resize/normalize → CLAHE (optional) → BayesShrink
wavelet denoising (optional) → follicle segmentation (Otsu/adaptive) → input
mode (full/ROI/masked) → DenseNet-121 + attention → Grad-CAM.)_

Denoising wavelet menerapkan soft thresholding pada koefisien detail setiap
subband. Varian noise diestimasi dari subband HH melalui _median absolute
deviation_:

σ̂ = median(|w ∈ HH|) / 0.6745 (1)

Ambang adaptif tiap subband j selanjutnya dihitung dari varian noise dan
sebaran koefisien subband tersebut:

T*j = σ̂² / σ*{w,j} (2)

dengan w adalah koefisien detail subband HH, σ̂ estimasi standar deviasi
noise, dan σ\_{w,j} standar deviasi koefisien subband j. Pada finalisasi,
kedua persamaan ditulis ulang lewat equation editor di DOCX (bukan gambar)
dengan nomor rata kanan dan dirujuk di teks menurut nomornya; bentuk
persamaan juga disamakan dengan implementasi di `src/preprocessing/wavelet.py`.

Dua metrik deskriptif sisi pengolahan citra menyertai tahap ini:
_peak signal-to-noise ratio_ (PSNR) dan _structural similarity_
(SSIM) [11], dihitung untuk citra denoised terhadap citra raw. Karena
dataset tidak memiliki citra bersih ground-truth, keduanya mengukur seberapa
besar citra berubah akibat denoising — bukan ukuran mutlak "kualitas yang
lebih baik" — sehingga dilaporkan sebagai bukti deskriptif sisi PCD dan bukan
penentu verdict hipotesis (kriteria verdict pada Subbab 3.6).

### 3.3. Segmentasi Folikel Unsupervised dan Mode Input

Berbeda dari studi pendahulu yang menyandarkan fokus folikel pada
attention implisit, penelitian ini menguji fokus eksplisit lewat preprocessing
bentuk segmentasi. Mask folikel dibangun dengan pendekatan unsupervised:
threshold Otsu [3] atau adaptif, diikuti operasi morfologi (opening, filling
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

_Catatan final: cek ulang perilaku masked di kode (`--input-mode masked`)
sebelum menulis klaim baris tabel ini di DOCX._

### 3.4. Arsitektur Attention dan Backbone

Backbone utama adalah DenseNet-121 [4] pralatih ImageNet dengan head
fully-connected dan dropout 0,3. Empat varian modul attention dibandingkan,
semuanya ditempatkan pada peta fitur sebelum pooling global:

- **Self-attention** (11,4 juta parameter) [1]: proyeksi Q, K, V multi-head
  pada token spasial. Token fitur spasial berdimensi (B, H·W, C): untuk
  input 224 piksel, H = W = 7 sehingga N = 49 token.

  Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V (3) -- equation editor

  Catatan implementasi penting: pada versi awal, self-attention sempat
  ditempatkan setelah pooling (hanya satu token per citra) dan attention-nya
  degenerasi; kini dipasang pada token spasial 7×7 sebelum pooling, dan
  angka yang dilaporkan dalam tugas akhir ini selalu memakai penempatan
  yang benar.

- **CBAM** (7,3 juta parameter) [5]: channel attention lalu spatial
  attention dioperasikan berturut-turut pada peta fitur (B, C, H, W).
- **SE-Net** (7,3 juta parameter) [6]: attention kanal saja; tidak
  dimasukkan ke matriks eksperimen utama, hanya untuk laporan internal.
- **Transformer-attention** (32,4 juta parameter) [7]: patch token;
  di luar scope utama karena bobot jauh lebih besar dan tidak lolos klaim
  ringan yang diuji H3.

Untuk perbandingan antar mekanisme attention hanya dua kandidat yang
diangkat: self-attention (replikasi Artikel 1) dan CBAM (kandidat ringan yang
memberi spatial attention relevan dengan folikel). Pilihan ini
diaktifkan lewat argumen `--attention`; jumlah parameter dan latensi
dicatat ke `experiments/results/results.csv`.

### 3.5. Protokol Pelatihan dan Penanganan Ketimpangan Kelas

Pelatihan memakai Adam (learning rate 1e-4, weight decay 1e-4), batch size
64, dan _automatic mixed precision_ untuk mempercepat pelatihan di Colab T4
(±10 GB VRAM, estimasi 50–60 menit per run 30 epoch). Early stopping
patience 10; scheduler ReduceLROnPlateau. Augmentasi pada split train:
rotasi acak hingga 15°, flip horizontal, dan RandomResizedCrop [13].
Split 70/15/15 dengan seed 42 untuk split, shuffle, augmentasi, dan
inisialisasi model, sehingga semua eksperimen sebanding dan reprodusibel.

Ketimpangan kelas 4:1 ditangani dengan pos_weight pada
BCEWithLogitsLoss — variabel w di Persamaan (4) di-set rasio
n_negatif/n_positif yang dihitung hanya dari split train:

ℓ = (1/N) Σ [ −w · yᵢ · log pᵢ − (1−yᵢ) · log(1−pᵢ) ] (4) -- equation editor

dengan yᵢ biner label kelas positif (PCOS) dan pᵢ peluang prediksi. Pendekatan
ini menyeimbangkan gradien kedua kelas tanpa mengubah distribusi data,
berbeda dari resampling
atau SMOTE; nilai w yang dipakai terekam pada
`run_summary.json` tiap run dan dapat dinonaktifkan dengan
`--no-pos-weight` untuk kebutuhan kontrol.

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

Kriteria keberhasilan masing-masing hipotesis, diprespecified di
`experiments/hypotheses.py` sehingga verdict dievaluasi otomatis dengan kode
yang sama (bukan penilaian manual penulis):

- **H1 (denoising):** diterima bila akurasi atau AUC E2 lebih tinggi secara
  ketat dari E1 pada minimal satu metrik tanpa regresi lebih dari 0,005 pada
  metrik satunya; hasil identik tidak dihitung diterima.
- **H2 (segmentasi):** E4 (ROI crop Otsu) diprespecified sebagai varian utama
  dengan AUC sebagai metrik utama; diterima bila AUC E4 > E3 dan F1 E4 tidak
  lebih buruk melebihi 0,005. E5 (masked) dilaporkan deskriptif sebagai
  pembanding sekunder dan tidak menentukan verdict, sehingga tidak terjadi
  pemilihan varian terbaik di atas test set.
- **H3 (efisiensi attention):** diterima bila akurasi/AUC E6 ≥ E4 − 0,5 poin
  dengan jumlah parameter lebih sedikit dan latensi tidak lebih besar.
- Toleransi 0,005 berada dekat rentang noise run-to-run pada satu seed;
  konsekuensinya dimuat sebagai batasan pada Subbab 4.6.

Metrik evaluasi sisi diagnostik: accuracy, precision, recall, F1-score,
specificity, dan AUC, dilengkapi confusion matrix dan kurva ROC. Karena H3
mencakup sisi efisiensi, jumlah parameter dan latensi inferensi CPU
(`infer_ms_cpu`) juga direkam. Bukti sisi pengolahan citra (PSNR/SSIM,
statistik ROI: jumlah region, coverage, luas rata-rata dan maksimum) dan
overlay mask dihasilkan tanpa pelatihan oleh `evaluate_quality.py`, jadi bebas
dari variabilitas pelatihan. Verdict H1–H3 direkam pada
`matrix_summary.json`, sehingga setiap klaim pada Bab 4 dapat dilacak ke
file yang dihasilkan pipeline, bukan dihitung manual.

Empat batasan desain yang disebut sejak awal: (a) tidak ada mask
ground-truth — segmentasi berperan sebagai penyedia ROI, bukan pengukuran
Dice klinis; (b) satu seed dan satu run per eksperimen — perbandingan
bersifat deskriptif; (c) split level citra, bukan pasien/video, sehingga
frame berurutan dari perekaman yang sama bisa nyaris identik meski
lolos md5; (d) kuota komputasi Colab membatasi replikasi multi-seed.

## 4. Hasil dan Pembahasan

> **BLOCKER: seluruh angka pada bagian ini menunggu
> `experiments/results/results.csv` (dan `experiments/quality/summary.json`).
> Teks di bawah adalah kerangka pelaporan, tidak boleh dikirim sebelum angka
> final diisi.**

### 4.1. Denoising terhadap Kualitas Citra dan Klasifikasi (E1 vs E2)

Sisi pengolahan citra, efek BayesShrink diukur dari perubahan noise dan
struktur citra: PSNR/SSIM raw → denoised per kelas disajikan pada Table 5,
dengan contoh visual pada Fig. 2. Verdict H1: [FILL: verdict dari
`matrix_summary.json` berdasarkan kriteria Subbab 3.6].

[FILL: 2–3 paragraf analisis: jika E2 unggul, hubungkan dengan reduksi
noise USG terhadap signal-to-noise citra; jika tidak, jelaskan bahwa
BayesShrink dapat menghasilkan perubahan citra besar tanpa manfaat
diskriminatif; tegaskan PSNR/SSIM hanya deskriptif dan verdict dibaca
bersama akurasi/AUC.]

**Table 5. Metrik kualitas sisi pemrosesan citra (raw → denoised) dan
statistik ROI per kelas [isi dari `experiments/quality/summary.json`].**

| Metrik                              | PCOS (mean ± std) | Sehat (mean ± std) |
| ----------------------------------- | ----------------- | ------------------ |
| PSNR (dB)                           | [FILL]            | [FILL]             |
| SSIM                                | [FILL]            | [FILL]             |
| Jumlah region ROI                   | [FILL]            | [FILL]             |
| Coverage mask (%)                   | [FILL]            | [FILL]             |
| Luas region rata-rata/maksimum (px) | [FILL]            | [FILL]             |

### 4.2. Segmentasi Folikel terhadap Klasifikasi (E3 vs E4 vs E5)

Verdict H2: [FILL: E4 vs E3 di AUC dengan dipertahankan toleransi F1; plus
laporan deskriptif E5].

[FILL: 2–3 paragraf: ROI memberi zoom ke area folikel dengan padding 8 px;
masked mempertahankan proporsi asli namun menggelapkan background; bandingkan
E4 dan E5 sebagai catatan sekunder — hanya E4 yang menentukan verdict;
studi pendahulu [1], [2] mengandalkan fokus folikel lewat attention implisit
tanpa menguji fokus eksplisit, dan di sinilah kebaruan H2 berdiri; tutup
dengan pengaitan ke interpretasi Grad-CAM pada Subbab 4.4.]

**Table 6. Hasil klasifikasi enam eksperimen [isi dari `results.csv`].**

| ID  | Preprocessing       | Segmentasi/ROI | Attention      | Acc    | Prec   | Recall | F1     | Spec   | AUC    | Param (M) | Lat. (ms CPU) |
| --- | ------------------- | -------------- | -------------- | ------ | ------ | ------ | ------ | ------ | ------ | --------- | ------------- |
| E1  | baseline            | full           | self-attention | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 11.4      | [FILL]        |
| E2  | BayesShrink         | full           | self-attention | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 11.4      | [FILL]        |
| E3  | BayesShrink + CLAHE | full           | self-attention | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 11.4      | [FILL]        |
| E4  | BayesShrink + CLAHE | roi            | self-attention | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 11.4      | [FILL]        |
| E5  | BayesShrink + CLAHE | masked         | self-attention | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 11.4      | [FILL]        |
| E6  | BayesShrink + CLAHE | roi            | cbam           | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 7.3       | [FILL]        |

### 4.3. Perbandingan Mekanisme Attention (E4 vs E6)

Verdict H3: [FILL]. [FILL: 2–3 paragraf: bandingkan akurasi/AUC dengan
parameter 11,4 juta (self-attention) vs 7,3 juta (CBAM) serta latensi CPU;
jika setara, posisikan CBAM sebagai pilihan lebih efisien; jika tidak
setara, bahas trade-off murni parameter-latensi; karena hanya satu seed,
hindari over-claim].

### 4.4. Interpretasi Kualitatif dengan Grad-CAM

Fig. 4 menampilkan heatmap Grad-CAM [8] pada sampel model utama
[FILL: eksperimen yang dipilih, mis. E4]. [FILL: 2–3 paragraf: apakah area
aktivasi menumpuk pada folikel (konsisten dengan H2) atau menyebar ke
background; kaitkan dengan kasus false negative — pada konteks skrining, FN
lebih berbahaya daripada FP; catat bahwa analisis ini kualitatif, bukan
kuantifikasi korelasi heatmap vs mask].

### 4.5. Perbandingan dengan Studi Terdahulu

Angka pada dataset ini tidak dapat dibandingkan langsung dengan angka
Artikel 1 [1] (99,13%) dan Artikel 2 [2] (94,8%): split berbeda, protokol
evaluasi tidak sebanding, dan pembeda terbesar adalah deduplikasi yang
digunakan di sini. Hasil replikasi lama 99,72% dikenali sebagai kebocoran
test set (salinan identik di train) dan tidak dilaporkan sebagai hasil
valid [dibahas pada Subbab 4.6]. [FILL: 1–2 paragraf: konfigurasi dasar
yang sama dengan Artikel 1 (DenseNet-121 + self-attention, PCOS-XAI,
BayesShrink) memungkinkan pembacaan perbandingan; klaim utama bukan "lebih
akurat", melainkan bahwa penataan preprocessing dan pemilihan attention
kini berdiri di atas data, ditambah laporan parameter dan latensi yang
tidak disertakan Artikel 1.]

### 4.6. Batasan dan Ancaman Validitas

1. **Kontrol duplikat terbatas pada hash byte-identik.** Frame berurutan
   perekaman video yang sama bisa nyaris identik namun lolos md5. Bila
   metadata pasien/video tersedia, split level pasien adalah perbaikan
   paling penting berikutnya.
2. **Satu seed dan satu run per eksperimen.** Toleransi verdict 0,005
   berada dekat rentang noise run-to-run. Replikasi multi-seed (min 3)
   sebaiknya dijadwalkan bila kuota Colab memungkinkan.
3. **Ketimpangan kelas 4:1.** Akurasi murni menyesatkan; karena itu
   specificity, F1, dan AUC dilaporkan bersamaan dan verdict H2
   disematkan pada AUC.
4. **Segmentasi tanpa ground-truth.** Kualitas mask hanya bisa dinilai
   kualitatif dan dari statistik ROI; tidak ada metrik Dice klinis.
5. **Batasan scope G1–G3.** Fusi multimodal, validasi antar dataset, dan
   kuantifikasi ketidakpastian (MC Dropout/kalibrasi) tidak dievaluasi dan
   disarankan sebagai arah lanjutan.
6. **Replikasi awal tidak valid.** Angka 99,72% dari titik awal proyek
   dilaporkan sebagai bukti kebocoran (81% test set punya salinan identik
   di train), bukan sebagai hasil valid, dan angka tersebut tidak
   dipakai sebagai komponen hasil.

## 5. Kesimpulan

> [FILL: SIMPAN SEBAGAI PARAGRAF, BUKAN DAFTAR. Kaitan: JUTIF menyatakan
>
> > Conclusion tak boleh mengulang abstrak. Setelah angka masuk, tulis ulang
> > bagian ini.]

Penelitian ini menguji tiga klaim yang ditinggalkan tanpa bukti oleh studi
pendahulu: wavelet denoising membantu performa, fokus folikel eksplisit
membantu, dan attention ringan dapat mempertahankan akurasi. Melalui studi
ablasi enam konfigurasi dengan kriteria prespecified pada dataset yang sama,
setiap klaim diberi jawaban yang dapat dilacak ke artefak pipeline.
[FILL: jawaban tiga rumusan masalah + verdict H1–H3 dengan metrik utamanya
(diterima/ditolak beserta angka pendukung)]. [FILL: klaim yang dibatasi —
verdict deskriptif pada satu split dan satu seed, dataset tunggal,
ketimpangan kelas 4:1, tanpa metadata pasien].

Untuk penelitian selanjutnya, prioritas yang nyata adalah
(a) replikasi multi-seed untuk mengukur variasi run-to-run, (b) validasi
antar dataset (mis. UBC-OV atau PCOSGen) sebagai uji generalisasi,
(c) split level pasien bila metadata tersedia, (d) kuantifikasi
ketidakpastian prediksi melalui MC Dropout atau kalibrasi, dan (e) uji
deployment pada perangkat nyata untuk menjawab klaim low-resource secara
terukur. Dengan mengisi pengukuran preprocessing dan attention yang tidak
disertakan Artikel 1 maupun 2, penelitian ini menopang klaim-klaim
tersebut agar dapat berdiri di atas data.

## Konflik Kepentingan

Penulis menyatakan tidak ada konflik kepentingan dalam penelitian ini.

## Ucapan Terima Kasih

[FILL: isi bila ada pemberi dana, penyedia dataset, atau pihak yang
membantu; kosongkan bila tidak ada.]

## Daftar Pustaka

Entri [1]–[13] ditulis dari metadata baku dan tetap perlu dicek ulang
(DOI via CrossRef) pada finalisasi. Slot [REF] pada teks di atas menjadi
entri baru yang harus dicari dan diverifikasi (jangan mengarang).

<!-- Perbaiki: nomornya tetap urut kemunculan; bila teks revisi [REF] masuk,
renumber ulang. Saat ini nomor [9]–[13]/[REF-slot] sama diguna.
JUTIF: >=25 referensi; >=80% jurnal/prosiding 5 tahun terakhir;
buku <=20%; self-citation <=3; SEMUA harus dikutip di teks. -->

[1] S. Tiwari, A. Shukla, and A. K. Sharma, "Attention-guided lightweight deep learning architecture for classification of polycystic ovary syndrome from ultrasound images," Intelligence-Based Medicine, vol. 13, 100358, 2026, doi: 10.1016/j.ibmed.2026.100358.

[2] [FILL: daftar semua penulis sesuai metadata artikel; bila lebih dari 7, tulis 6 penulis pertama lalu et al.] "Transfer Learning-Enhanced CNN Model for Integrative Ultrasound and Biomarker-Based Diagnosis of Polycystic Ovarian Disease," Scientific Reports, vol. 15, 34519, 2025, doi: 10.1038/s41598-025-17711-w.

[3] N. Otsu, "A threshold selection method from gray-level histograms," IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62–66, 1979, doi: 10.1109/TSMC.1979.4310076.

[4] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 4700–4708, doi: 10.1109/CVPR.2017.243.

[5] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in Proc. European Conference on Computer Vision (ECCV), 2018, pp. 3–19, doi: 10.1007/978-3-030-01234-2_1.

[6] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 7132–7141, doi: 10.1109/CVPR.2018.7454510.

[7] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5998–6008.

[8] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Batra, and D. Parikh, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in Proc. IEEE International Conference on Computer Vision, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[9] D. L. Donoho and J. M. Johnstone, "Ideal spatial adaptation by wavelet shrinkage," Biometrika, vol. 81, no. 3, pp. 425–455, 1994, doi: 10.1093/biomet/81.3.425.

[10] S. G. Chang, B. Yu, and M. Vetterli, "Adaptive wavelet thresholding for image denoising and compression," IEEE Transactions on Image Processing, vol. 9, no. 9, pp. 1528–1546, 2000, doi: 10.1109/83.862633.

[11] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004, doi: 10.1109/TIP.2003.819861.

[12] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[13] C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," Journal of Big Data, vol. 6, no. 1, 60, 2019, doi: 10.1186/s40537-019-0197-0.

[14] [FILL: penulis/organisasi sesuai halaman Kaggle], "PCOS-XAI Ultrasound Dataset," Kaggle. [FILL: URL dan tanggal akses; tulis sebagai entri web IEEE].

### Slot referensi yang masih terbuka (HARUS dicari & diverifikasi; jangan mengarang)

- [REF] epidemiologi PCOS (angka 8–13%, publikasi 2021+, preferensi jurnal medis)
- [REF] kriteria Rotterdam (konsensus 2003/2004 atau versi terbaru)
- [REF] variabilitas antar-operator USG (jurnal medis 2020+)
- [REF] CLAHE asli (Pizer dkk. 1987) — klasik, boleh dipakai dengan catatan
- [REF] 2–4 CNN untuk klasifikasi citra USG medis/ginekologi 2022–2025
- [REF] preprocessing penting pada klasifikasi citra medis (2020+)
- [REF] augmentasi citra pada dataset medis (2021+, bukan survey 2019)
- [REF] DenseNet untuk ultrasound (2021+) — 1–2 karya
- [REF] transfer learning pada citra medis — pilih 1–2 karya
- [REF] 1–2 karya low-resource CAD / edge deployment (2022+)
- [REF] perbandingan attention pada tugas klasifikasi citra (2021+)

Target total: sekitar 25–30 entri (14 terverifikasi + 11 slot).
Perhatikan aturan JUTIF: minimal 25 referensi, semua dikutip di teks,
jurnal/prosiding >= 80%, buku <= 20% (saat ini 0), self-citation <= 3.

<!--
CHECKLIST JUTIF untuk migrasi DOCX:
[ ] Times New Roman 11 pt, margin kiri 2,5 / atas 2,5 / kanan 2 / bawah 2,5 cm, A4, spasi 1,15, satu kolom
[ ] Judul <= 20 kata Inggris
[ ] Penulis + tanda * korespondensi; afiliasi Inggris; email korespondensi
[ ] Abstrak 150–250 kata tanpa sitasi; kata kunci 3–6 alfabetis
[ ] Persamaan (1)–(4) ditulis di equation editor, nomor rata kanan, dirujuk di teks
[ ] Caption tabel di atas, caption gambar di bawah; semua dirujuk dengan nomor
[ ] Semua figur >= 300 dpi, teks Inggris, desimal titik
[ ] Minimal 25 referensi, semua dikutip, IEEE konsisten
[ ] Panjang naskah >= 4000 kata (di luar abstrak, info penulis, referensi)
[ ] Tabel/gambar tidak terpotong halaman; posisinya atas/bawah halaman.
-->
