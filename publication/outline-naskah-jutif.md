# Outline Naskah JUTIF — PCOS Classification

Target: JUTIF Vol. 7 No. 2 (Februari 2026) atau terbitan berikutnya (rolling submission).
Sumber konten: `references/kajian-literatur-gap-hipotesis.md` (sumber tunggal narasi),
`README.md` (teknis), `experiments/results/results.csv` (hasil — MENUNGGU TRAINING).

## Keputusan pra-penulisan (perlu dikunci lebih dulu)

| Keputusan | Opsi | Status |
| --- | --- | --- |
| Bahasa isi | Indonesia (judul/abstrak/kata kunci Inggris) atau full Inggris | Belum dipilih |
| Cara tulis | Edit langsung `JUTIF-Template.docx` | Sesuai saran panduan |
| Bagian Result | Menunggu hasil E1–E6 dari Colab | BLOCKER — bagian hasil baru kerangka |
| Framing replikasi Artikel 1 | Dilaporkan sebagai re-run independen yang gagal valid (karena dedup + placement fix), bukan replikasi sukses; angka lama (99,72%) TIDAK dipakai sebagai hasil valid karena test set lama terkontaminasi 81% duplikat | Sudah diputuskan di README |
| Sudut evaluasi | Analisis Citra Medis (performa diagnostik + interpretabilitas); PSNR/SSIM dari sisi PCD dilaporkan sebagai bukti deskriptif H1, bukan cerita utama | Sesuai README proyek |

## Judul (≤ 20 kata, Inggris, tanpa nama studi kasus)

Kandidat 1 (14 kata, adaptasi judul kerja):

> **Impact of Wavelet Denoising and Follicle Segmentation on PCOS Classification
> from Ultrasound Images Using Attention-Based DenseNet-121**

Kandidat 2 (15 kata, menonjolkan perbandingan attention = gap G4):

> **Follicle-Focused Preprocessing and Attention Mechanism Comparison for PCOS
> Ultrasound Classification with Grad-CAM**

Catatan: kata "Grad-CAM" di judul opsional; cukup di abstrak/kata kunci bila ingin judul fokus metode.

## Informasi penulis (placeholder)

- Nama tanpa gelar; `*` pada corresponding author.
- Afiliasi Inggris: Department of Informatics, Jenderal Soedirman University, Indonesia.
- Email hanya corresponding author; nomor telepon/WhatsApp diisi di sistem submission (tidak tampil).

## Abstrak (150–250 kata, Inggris, spasi 1, tanpa sitasi)

Lima gerakan, masing-masing 1–2 kalimat:

1. **Latar belakang**: interpretasi USG PCOS subjektif antar-operator; dua studi
   terdahulu memakai attention tanpa ablasi denoising dan tanpa membandingkan
   mekanisme attention.
2. **Tujuan**: mengukur pengaruh wavelet denoising (BayesShrink) dan segmentasi
   folikel unsupervised terhadap performa DenseNet-121 + Attention, serta
   membandingkan self-attention vs CBAM.
3. **Metode**: ablation study enam konfigurasi (E1–E6) pada PCOS-XAI dataset
   (3.996 citra unik pasca deduplikasi md5; rasio kelas 4:1 ditangani pos_weight
   BCE), split 70/15/15 dengan seed tetap; metrik Accuracy/F1/Specificity/AUC +
   Grad-CAM.
4. **Hasil**: `[FILL — hasil E1–E6 + verdict H1–H3 dari results.csv]`.
5. **Temuan/dampak**: `[FILL — peran pemrosesan citra eksplisit & seleksi attention
   yang efisien untuk skrining low-resource]`.

Kata kunci (3–6, alfabetis, Inggris), draf: Attention Mechanism, CBAM,
Grad-CAM, Ovarian Follicle Segmentation, Polycystic Ovary Syndrome, Ultrasound
Image Classification. (Urutkan alfabetis final.)

## Sistematika bab (judul eksplisit, maksimal 2 level; total ±4.000–5.000 kata di luar abstrak/referensi)

### 1. Introduction (10–20% → ±600–900 kata)

1.1 Konteks: PCOS, peran USG transvaginal, kriteria Rotterdam, subjektivitas
operator. (±2 paragraf)

1.2 Kebutuhan CAD berbasis deep learning + kajian pustaka mini: Artikel 1
(Tiwari dkk.), Artikel 2 (Sundari dkk.), ditambah 2–4 studi lain, disusun
naratif → mengarah ke **gap G4 (tanpa perbandingan mekanisme attention) dan sisa
G5 (klaim lightweight tanpa pengukuran)**. G1–G3 disebut singkat sebagai
keterbatasan/laporkan, bukan scope. (±2–3 paragraf)

1.3 Tujuan eksplisit (3 rumusan masalah dari §3 kajian literatur) dan implikasi
metode secara implisit. (±1 paragraf)

Sumber kalimat: §1–§2 kajian literatur. Jangan duplikasi detail panjang artikel —
pindahkan ke Related Work/Method agar alur hemat kata.

### 2. Related Work (opsional bab terpisah; JUTIF membolehkan bab tambahan)

Judul eksplisit bila dipisah: "Related Works on Ultrasound-Based PCOS
Classification". Isi: komparasi Artikel 1 vs Artikel 2 (+ studi lain) dalam
tabel ringkas (arsitektur, preprocessing, dataset, ablasi, attention,
ketidakpastian, pengukuran efisiensi), ditutup narasi benang yang tinggal.

Pilihan ini menambah jumlah kata; bila total mepet batas, gabungkan ke dalam
Introduction tanpa bab sendiri.

### 3. Proposed Method (20–30% → ±1.200–1.700 kata)

Judul eksplisit, mis.: "Follicle-Focused Deep Learning Pipeline for PCOS
Classification".

3.1 **Dataset and Deduplication** — PCOS-XAI 11.784 citra mentah → dedup md5
keep-first → 3.996 unik (3.184 PCOS, 812 sehat; rasio 4:1). Table 1: distribusi
sebelum/sesudah dedup. Jelaskan kenapa dedup wajib (81% test set lama punya
salinan identik di train).

3.2 **Preprocessing Pipeline** — resize 224 → (CLAHE opsional) → BayesShrink
denoising → segmen opsional. Persamaan wajib via equation editor (bukan
gambar), bernomor kanan atas, dirujuk di teks: (1) threshold adaptif
BayesShrink σ; (2) rumus CLAHE bila sempit. Fig. 1: diagram pipeline
(tulis semua teks dalam Inggris, ±300 dpi).

3.3 **Unsupervised Follicle Segmentation and Input Modes** — Otsu/adaptive +
morfologi + hapus komponen tepi + fallback persentil gelap saat Otsu menangkap
background (coverage > 50%). Definisi 3 mode input: full / ROI crop / masked.
Table 2: ringkas mode input.

3.4 **Attention Mechanisms and Backbone** — DenseNet-121 + tiga keluarga
attention (Self-attention pada token spasial 7×7 sebelum pooling; SE-Net/CBAM
pada peta fitur 2D; Transformer patch-based). Formula scaled dot-product
attention + pos_weight BCE. Table 3: parameter tiap opsi
(Self-Attention 11.4M; SE/CBAM 7.3M; Transformer 32.4M).

3.5 **Training Protocol and Class Imbalance Handling** — Adam (lr 1e-4,
weight decay 1e-4), AMP, early stopping patience 10, ReduceLROnPlateau,
augmentasi (rotasi 15°, flip horizontal, RandomResizedCrop); pos_weight BCE
(rasio n_neg/n_pos dari train split saja, tanpa resampling) agar akurasi tidak
menyesatkan pada rasio 4:1.

3.6 **Hypotheses, Experiment Matrix, and Evaluation Metrics** — Table 4:
matriks E1–E6 (dari `experiments/experiments.yml`) + pemetaan ke hipotesis
H1–H3 dan kriteria kesuksesan deskriptif (toleransi 0,005; satu split sehingga
tanpa uji signifikansi formal — nyatakan dari awal). Metrik: Accuracy,
Precision, Recall, F1, Specificity, AUC, params, latency CPU
(`infer_ms_cpu`); PSNR/SSIM dari `evaluate_quality.py` sebagai bukti deskriptif
PCD. Sertakan 4 batasan: (a) tanpa mask GT → segmentasi bukan pengukuran Dice
klinis; (b) satu seed (42), toleransi verdict ≈ rentang noise run-to-run;
(c) split level citra, berpotensi frame dari video yang sama tak terdeteksi
hash; (d) satu run per eksperimen.

Catatan JUTIF: state of the art masuk di bagian ini — sertakan singkat kenapa
pilihan lain (MBBO, SMOTE, EfficientNet-B3 pada Artikel 2) berada di luar scope
(fokus: wavelet denoising, segmentasi folikel, DenseNet-121 + Attention,
Grad-CAM).

### 4. Results and Discussion (50–65% → ±2.400–3.200 kata)

Semua caption tabel di atas tabel; caption gambar di bawah gambar; semua dirujuk
dengan nomor (bukan "tabel di bawah"). Desimal titik. Teks dalam figur
berbahasa Inggris.

Tabel/gambar rencana:

- Table 5: kualitas sisi pemrosesan citra (`evaluate_quality.py`: PSNR/SSIM
  raw vs denoised; statistik ROI: jumlah region, coverage, luas rata-rata/maks).
- Table 6: hasil klasifikasi E1–E6 (`results.csv`: Accuracy, F1, Specificity,
  AUC + params + latency).
- Table 7: verdict H1–H3 (`accepted`/`not_accepted` dari `matrix_summary.json`).
- Fig. 2: contoh hasil segmentasi (original → mask → ROI crop).
- Fig. 3: kurva ROC + confusion matrix model utama.
- Fig. 4: Grad-CAM overlay untuk argumen fokus folikel.

4.1 **Denoising Effect on Image Quality and Classification (E1 vs E2)** — dari
verdict H1; sisipkan Table 5 dan Fig. 2 bila relevan.

4.2 **Effect of Follicle Segmentation on Classification (E3 vs E4 vs E5)** —
verdict H2; E4 (ROI crop Otsu) adalah varian prespecified dengan AUC sebagai
metrik utama; E5 (masked) pelaporan deskriptif saja (bukan pemilihan varian
terbaik di atas test set).

4.3 **Accuracy–Efficiency Trade-off of Attention Mechanisms (E4 vs E6)** —
verdict H3; bandingkan parameter + `infer_ms_cpu`.

4.4 **Qualitative Interpretation with Grad-CAM** — heatmap menghadapi klaim
kecil bahwa model "fokus folikel, mengabaikan background"; bahas
false negative/false positive (FN paling berbahaya untuk skrining).

4.5 **Comparison with Prior Studies** — sebutkan konteks angka Artikel 1
(99,13%) dan Artikel 2 (94,8%) TANPA klaim mengalahkan; tegaskan beda desain
(dedup, split level citra, satu seed, satu run) membuat perbandingan antar
dataset tidak 1-1.

4.6 **Threats to Validity and Limitations** — frame berurutan dari video yang
sama pasca-dedup bisa nyaris identik; satu-seed pada toleransi 0,005;
G1–G3 di luar scope; replikasi lama tidak valid sebagai justifikasi desain.

BLOCKER: semua angka bagian ini menunggu `experiments/results/results.csv` —
jangan tulis paragraf hasil sebelum angka final masuk.

### 5. Conclusion (paragraf, bukan daftar; jangan mengulang abstrak; ±150–250 kata)

(a) Jawaban atas 3 rumusan masalah + verdictnya; (b) klaim dibatasi: deskriptif,
satu seed, rasio 4:1; (c) arah lanjutan: G2 (validasi antar dataset),
G3 (ketidakpastian), G1 (multimodal), replikasi multi-seed.

### 6. Conflict of Interest

1 kalimat standar.

### 7. Acknowledgement

Opsional; isi bila ada dana/dosen/pihak yang perlu diacungi.

### 8. References — ≥25 entri, IEEE, semua dikutip di teks

Target komposisi: ≥80% jurnal/konferensi dalam 5 tahun terakhir, buku ≤20%,
self-citation ≤3, nama jurnal tidak disingkat, wajib volume+nomor+halaman+DOI.

Rencana blok referensi (dari sekarang 2 sumber inti → target ≥25):

1. Tiwari dkk. 2026, Intelligence-Based Medicine (sudah ada DOI).
2. Sundari dkk. 2025, Scientific Reports (sudah ada DOI).
3–8. PCOS klinis & USG (epidemiologi PCOS, kriteria Rotterdam, ultrasonografi
   transvaginal, subjektivitas antar-operator).
9–14. CNN/transfer learning untuk klasifikasi citra USG medis.
15–19. DenseNet-121 (Huang dkk.), attention (SE-Net Hu dkk., CBAM Woo dkk.,
   self-attention/transformer Vaswani dkk.).
20–22. Grad-CAM (Selvaraju dkk. 2017).
23–24. Wavelet denoising (thresholding Donoho–Johnstone; BayesShrink Chang dkk.)
   dan Otsu (1979) / adaptive threshold.
25+. Arsitektur ringan / CAD low-resource untuk layanan kesehatan primer
   (mendukung narasi G5).

Contoh format target (IEEE, nama jurnal penuh):

```text
[1] S. Tiwari, A. Shukla, and A. K. Sharma, "Attention-guided lightweight deep
learning architecture for classification of polycystic ovary syndrome from
ultrasound images," Intelligence-Based Medicine, vol. 13, 100358, 2026,
doi: 10.1016/j.ibmed.2026.100358.
```

## Rencana artefak gambar/tabel (dari repo)

| Artefak | Sumber repo | Ketergantungan |
| --- | --- | --- |
| Fig. 1 pipeline diagram | dibuat baru | hanya butuh waktu |
| Fig. 2 overlay segmentasi | `experiments/quality/` (figure overlay) | `evaluate_quality.py` (CPU, bisa jalan detik-menit) |
| Fig. 3 ROC/confusion matrix | artefak tiap run eksperimen | E1–E6 selesai (BLOCKER) |
| Fig. 4 Grad-CAM | `checkpoints/<run>/gradcam/` | E1–E6 selesai (BLOCKER) |
| Table 1–7 | `results.csv`, `matrix_summary.json`, quality `summary.json` | E1–E6 selesai untuk tabel 6–7 |

Semua figur: ≥300 dpi, teks Inggris, gaya formal, tanpa judul berlebihan di
dalam gambar (caption yang berfungsi).

## Risiko / pertanyaan terbuka

1. **E1–E6 belum jalan** → bagian Result hanya kerangka; abstrak bagian hasil
   dan conclusion tidak boleh ditulis dulu.
2. **Replikasi Artikel 1 tidak valid** → cukup disebut terang di 4.6 sebagai
   justifikasi desain dedup, bukan sebagai klaim hasil.
3. **Perbandingan antar studi tidak 1-1** (beda dataset/split/seed) → hindari
   klaim mengalahkan; fokus ke metrik yang benar-benar diukur (params,
   `infer_ms_cpu`).
4. **Estimasi jumlah kata**: Introduction 900 + Related Work 400 + Method
   1.600 + Results and Discussion 2.800 + Conclusion 250 ≈ 5.950 — kemungkinan
   lebih dari butuh; bila mepet, gabungkan Related Work ke Introduction.
