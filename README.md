# Pengaruh Wavelet Denoising dan Segmentasi Folikel terhadap Klasifikasi PCOS dengan DenseNet-121 + Attention dan Grad-CAM

Klasifikasi Polycystic Ovary Syndrome (PCOS) dari citra ultrasound: mengukur pengaruh preprocessing wavelet denoising (BayesShrink) dan segmentasi folikel unsupervised terhadap performa DenseNet-121 + Attention, dengan interpretabilitas Grad-CAM.

Topik ini dipakai untuk **dua tugas akhir matakuliah sekaligus**: Pengolahan Citra Digital (PCD) dan Analisis Citra Medis. Satu pipeline dan dataset yang sama, dengan sudut evaluasi berbeda: PCD fokus pada kualitas preprocessing/segmentasi, Analisis Citra Medis fokus pada performa diagnostik dan interpretabilitas.

## Referensi

1. Tiwari, S., Shukla, A., & Sharma, A.K. (2026). _Attention-guided lightweight deep learning architecture for classification of polycystic ovary syndrome from ultrasound images._ Intelligence-Based Medicine, 13, 100358. DOI: [10.1016/j.ibmed.2026.100358](https://doi.org/10.1016/j.ibmed.2026.100358)

2. Sundari, M.S., et al. (2025). _Transfer Learning-Enhanced CNN Model for Integrative Ultrasound and Biomarker-Based Diagnosis of Polycystic Ovarian Disease._ Scientific Reports, 15:34519. DOI: [10.1038/s41598-025-17711-w](https://doi.org/10.1038/s41598-025-17711-w)

## Konteks penelitian

Narasi penelitian, latar, research gap G1-G5, rumusan masalah, hipotesis
H1-H3 beserta kriteria penerimaannya, dan rencana eksperimen didokumentasikan
penuh di
[`references/kajian-literatur-gap-hipotesis.md`](references/kajian-literatur-gap-hipotesis.md).
File itu adalah sumber tunggal agar tidak ada penjelasan ganda; README ini
hanya memuat hal teknis (struktur kode, cara menjalankan eksperimen, hasil).

Catatan perilaku pipeline: jika Otsu menangkap background (coverage > 50%),
segmentasi otomatis fallback ke threshold persentil gelap.

## Struktur Proyek

```
pcos-classification/
├── datasets/
│   ├── infected/              # 6.784 citra PCOS
│   └── noninfected/           # 5.000 citra sehat
├── src/
│   ├── dataset.py             # Custom Dataset & DataLoader (full/roi/masked)
│   ├── preprocessing/         # Tahap PCD: enhancement, denoise, segmentasi, metrik
│   │   ├── wavelet.py         # Wavelet denoising (BayesShrink)
│   │   ├── segmentation.py    # Segmentasi folikel unsupervised (Otsu/adaptive + morfologi)
│   │   └── quality.py         # Enhancement CLAHE, metrik PSNR/SSIM
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Metrics & visualisasi
│   ├── gradcam.py             # Grad-CAM visualisasi (custom)
│   └── models/
│       ├── attention.py       # Attention modules (Self-Attn, SE-Net, CBAM, Transformer)
│       └── densenet121.py     # DenseNet-121 + Attention
├── experiments/
│   ├── run_matrix.py          # Runner matriks eksperimen E1-E6
│   └── evaluate_quality.py    # Evaluasi PSNR/SSIM + statistik ROI (tanpa training)
├── checkpoints/               # Model checkpoints
├── requirements.txt
└── main.py
```

## Fitur

### Attention Mechanisms (Perbandingan)

| Mekanisme          | Parameters | Prinsip Kerja                  |
| ------------------ | ---------- | ------------------------------ |
| **Self-Attention** | 11.4M      | Q, K, V multi-head (Artikel 1) |
| **SE-Net**         | 7.3M       | Channel-wise attention         |
| **CBAM**           | 7.3M       | Channel + Spatial attention    |
| **Transformer**    | 32.4M      | Patch-based attention          |

### Model Arsitektur

- **DenseNet-121 + Attention** - Backbone utama (11.4M params dengan self-attention)

### Advanced Features

- **Grad-CAM** - Visual heatmap untuk interpretasi model (implementasi custom)
- **Mixed Precision (AMP)** - Training float16 untuk percepatan
- **Wavelet Denoising** - preprocessing BayesShrink (opsional)
- **Segmentasi folikel unsupervised** - Otsu/adaptive + morfologi, mode input full/roi/masked

## Instalasi

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Penggunaan

### Basic Training

```bash
# Default (Self-Attention)
python main.py

# Bandingkan attention mechanisms (E1 vs E6 memakai self_attention vs cbam)
python main.py --attention self_attention --epochs 30
python main.py --attention cbam --epochs 30
```

### Advanced Features

```bash
# Preprocessing + segmentasi + ROI
python main.py --denoise --enhance clahe --segment otsu --input-mode roi --epochs 30

# Mode masked
python main.py --denoise --enhance clahe --segment otsu --input-mode masked --epochs 30
```

```bash
# Grad-CAM visualization
python main.py --gradcam --gradcam-samples 5

# Kombinasi lengkap sesuai judul kerja
python main.py --denoise --enhance clahe --segment otsu --input-mode roi --attention cbam --gradcam --epochs 30
```

### Semua Argumen

| Argumen             | Default          | Deskripsi                                               |
| ------------------- | ---------------- | ------------------------------------------------------- |
| `--data-dir`        | `datasets`       | Path dataset directory                                  |
| `--batch-size`      | `64`             | Batch size                                              |
| `--image-size`      | `224`            | Image resize size                                       |
| `--epochs`          | `30`             | Max training epochs                                     |
| `--lr`              | `1e-4`           | Learning rate                                           |
| `--weight-decay`    | `1e-4`           | Weight decay                                            |
| `--patience`        | `10`             | Early stopping patience                                 |
| `--save-dir`        | `checkpoints`    | Checkpoint save directory                               |
| `--denoise`         | `false`          | Enable wavelet denoising                                |
| `--enhance`         | `none`           | Contrast enhancement: none / clahe                      |
| `--segment`         | `none`           | Segmentasi folikel: none / otsu / adaptive              |
| `--input-mode`      | `full`           | Input model: full / roi / masked                        |
| `--seg-pad`         | `8`              | Padding ROI crop (px)                                   |
| `--no-pretrained`   | `false`          | Disable pretrained weights                              |
| `--num-workers`     | `2`              | DataLoader workers                                      |
| `--seed`            | `42`             | Seed split, shuffle, augmentasi, inisialisasi model     |
| `--no-dedupe`       | `false`          | Nonaktifkan pembuangan salinan identik (md5)            |
| `--no-pos-weight`   | `false`          | Nonaktifkan pos_weight BCE (ketimpangan kelas)          |
| `--device`          | `auto`           | auto (GPU bila ada) / cuda (wajib GPU) / cpu            |
| `--attention`       | `self_attention` | Attention: self_attention / se_net / cbam / transformer |
| `--dropout`         | `0.3`            | Dropout rate                                            |
| `--gradcam`         | `false`          | Generate Grad-CAM                                       |
| `--gradcam-samples` | `5`              | Jumlah sampel Grad-CAM                                  |

## Pipeline

### 1. Preprocessing

- Deduplikasi: salinan byte-identik (md5, keep-first) dibuang sebelum split, aktif secara default (`--no-dedupe` untuk mematikan)
- Resize 224×224, normalisasi [0,1]
- Wavelet denoising (BayesShrink) - opsional (`--denoise`)
- Enhancement CLAHE - opsional (`--enhance clahe`)
- Segmentasi folikel unsupervised: Otsu/adaptive + morfologi + hapus komponen tepi (`--segment`), dihitung pada skala kerja tetap 224×224 lalu mask di-upscale ke resolusi asli
- Mode input: citra penuh, ROI crop, atau masked (`--input-mode`); mask kosong dikembalikan sebagai citra asli, bukan citra hitam
- Augmentasi: RandomRotation(15), RandomHorizontalFlip, RandomResizedCrop

### 2. Matriks Eksperimen (Tugas Akhir PCD + Citra Medis)

```bash
# Jalankan 6 eksperimen E1-E6 sesuai experiments/experiments.yml
# (hasil ke experiments/results/results.csv)
python experiments/run_matrix.py --config experiments/experiments.yml

# Sebagian saja, atau timpa data-dir/epochs dari YAML
python experiments/run_matrix.py --config experiments/experiments.yml --only E1,E2
```

| ID  | Preprocessing       | Segmentasi/ROI | Attention      |
| --- | ------------------- | -------------- | -------------- |
| E1  | baseline            | none / full    | self_attention |
| E2  | BayesShrink         | none / full    | self_attention |
| E3  | BayesShrink + CLAHE | none / full    | self_attention |
| E4  | BayesShrink + CLAHE | otsu / roi     | self_attention |
| E5  | BayesShrink + CLAHE | otsu / masked  | self_attention |
| E6  | BayesShrink + CLAHE | otsu / roi     | cbam           |

### 3. Training

- Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
- Loss: BCEWithLogitsLoss dengan pos_weight = rasio n_negatif/n_positif
  split train (menyeimbangkan gradien ketimpangan 4:1 pasca-dedup; dinonaktifkan
  dengan `--no-pos-weight`). Nilai yang dipakai terekam di run_summary.json
- Mixed Precision (AMP) untuk percepatan
- Early stopping: patience=10
- Scheduler: ReduceLROnPlateau

### 4. Evaluasi ganda

- **PCD:** PSNR/SSIM (denoise vs asli), histogram/spektrum, overlay mask, statistik ROI (jumlah region, coverage, luas rata-rata/maksimum). PSNR/SSIM mengukur kesetiaan terhadap citra asli, bukan efektivitas penghilangan noise; nilai rendah pada CLAHE wajar karena perubahannya nonlinear.
- **Medis:** Accuracy, Precision, Recall, F1-Score, Specificity, AUC.
- Confusion matrix, ROC curve, Grad-CAM heatmap untuk interpretasi.

### Menjalankan di Colab (T4)

Aktifkan GPU lebih dulu: **Runtime > Change runtime type > T4 GPU**. Default
`--device auto` langsung memakai CUDA bila tersedia, dan `run_matrix.py`
meneruskan `device: auto` dari YAML ke setiap eksperimen. Di lokal dengan torch
CPU-only, nilai yang sama otomatis turun ke CPU. Untuk memastikan run tidak
diam-diam memakai CPU di Colab, jalankan dengan `--device cuda` (gagal cepat
bila GPU belum aktif).

```bash
# 0. Bukti sisi PCD untuk H1 (tanpa training, cepat, bisa jalan di CPU)
python experiments/evaluate_quality.py --data-dir /content/drive/MyDrive/data-latih/PCOS --samples-per-class 100

# Full matrix E1-E6 (data-dir Colab menimpa nilai YAML)
python experiments/run_matrix.py --config experiments/experiments.yml --data-dir /content/drive/MyDrive/data-latih/PCOS --epochs 30

# Sebagian dulu (disarankan untuk sesi Colab pendek)
python experiments/run_matrix.py --config experiments/experiments.yml --data-dir /content/drive/MyDrive/data-latih/PCOS --epochs 30 --only E1,E2
```

Ganti `device: auto` menjadi `device: cuda` di `experiments/experiments.yml`
bila ingin matrix berhenti dengan pesan jelas saat runtime tidak punya GPU.

Tiap eksperimen menyimpan checkpoint, plot, `stdout.log`, `training.log`, dan ringkasan metrik ke `experiments/results/results.csv`. Saat dijalankan lewat `run_matrix.py`, log training juga tampil live di terminal (stdout dan stderr digabung). Estimasi ±50-60 menit per run 30 epoch di T4.

### Dokumentasi JSON (proses & hasil machine-readable)

- Tiap run menulis `run_summary.json` di folder `--save-dir`-nya: timestamp, git commit, environment (device/GPU/versi torch), config lengkap, ukuran dataset, parameter model, history per-epoch, epoch terbaik, flag early stopping, metrik test, dan daftar artefak.
- `run_matrix.py` menulis `matrix_summary.json` di `out_dir`: seluruh baris eksperimen + verdict deskriptif H1-H3 (`accepted`/`not_accepted`, hanya untuk eksperimen yang datanya tersedia) + salinan tiap `run_summary.json`.
- `evaluate_quality.py` menulis `summary.json` di samping `summary.txt` (mean/std/min/max per metrik, per kelas).

## Hasil

### Replikasi Artikel 1: DenseNet-121 + Self-Attention

> **Tidak valid, tertunda replikasi ulang.** Angka di bawah dihasilkan sebelum
> dua perbaikan: (1) deduplikasi data (81% test set tadinya punya salinan
> identik di train) dan (2) perbaikan penempatan self-attention (dulu
> dijalankan pada satu token hasil pooling sehingga attention-nya degenerasi,
> kini pada token spasial 7×7 sebelum pooling).

| Metrik      | Hasil  | Artikel |
| ----------- | ------ | ------- |
| Accuracy    | 99.72% | 99.13%  |
| Precision   | 100.0% | 100.0%  |
| Recall      | 99.50% | 98.51%  |
| F1-Score    | 99.75% | 100.0%  |
| Specificity | 100.0% | -       |
| AUC         | 100.0% | 99.97%  |

### Matriks Eksperimen E1-E6

Hasil valid dari run Colab T4 pada commit `70e0de3`, 30 epoch maksimum dengan
early stopping patience 10, split test 600 citra. Sumber angka:
`experiments/results/results.csv` dan `experiments/results/matrix_summary.json`.

| ID  | Accuracy | Precision | Recall | F1     | Specificity | AUC    |
| --- | -------- | --------- | ------ | ------ | ----------- | ------ |
| E1  | 98.17%   | 99.58%    | 98.12% | 98.84% | 98.35%      | 99.65% |
| E2  | 94.67%   | 100.0%    | 93.32% | 96.54% | 100.0%      | 99.70% |
| E3  | 96.00%   | 99.78%    | 95.20% | 97.44% | 99.17%      | 99.56% |
| E4  | 95.00%   | 100.0%    | 93.74% | 96.77% | 100.0%      | 99.74% |
| E5  | 99.17%   | 99.58%    | 99.37% | 99.48% | 98.35%      | 99.72% |
| E6  | 96.00%   | 99.78%    | 95.20% | 97.44% | 99.17%      | 99.55% |

Parameter dan latency CPU: E1-E5 self-attention 11.416.961 parameter
(108,2-117,4 ms per forward), E6 CBAM 7.347.683 parameter (110,1 ms).

Verdict deskriptif dari `matrix_summary.json`:

- **H1 tidak diterima.** E2 menurunkan accuracy 3,50 poin dari E1; AUC naik
  0,05 poin, tetapi regresi accuracy melewati toleransi 0,5 poin.
- **H2 tidak diterima untuk varian utama E4.** AUC E4 lebih tinggi 0,18 poin
  dari E3, namun F1 turun 0,67 poin sehingga melewati toleransi 0,5 poin.
  E5 (masked) hanya deskriptif dan mencapai accuracy 99,17%, F1 99,48%.
- **H3 tidak diterima.** Accuracy E6 setara dalam toleransi dan parameter
  turun 35,6%, tetapi latency CPU 110,07 ms sedikit di atas E4 108,73 ms
  sehingga kriteria "latency tidak lebih besar" tidak terpenuhi. Selisih
  1,34 ms berada di bawah rentang pengukuran antar run self-attention
  (108,20-117,41 ms), sehingga klaim latency H3 bersifat indikatif; kriteria
  prespecified tidak diubah setelah melihat hasil.

Bukti sisi PCD dari `experiments/quality/` (n=100 per kelas): PSNR denoise
51,21 dB dan SSIM 0,9963; PSNR CLAHE 17,32 dB dan SSIM 0,7561. Segmentasi Otsu
menghasilkan mask kosong 8/200 (4,0%) dengan coverage 0,335; adaptive coverage
0,315. Coverage Otsu berbeda antar kelas (0,285 infected vs 0,385
noninfected), sedangkan adaptive berlawanan arah (0,352 vs 0,279). Arah yang
tidak konsisten ini menunjukkan deskriptor region belum layak dipakai sebagai
penanda diagnostik; verdict H2 tetap dari metrik model.

Catatan kode: run di atas memakai commit `70e0de3`. Setelah run, guard
`sigma <= 0` ditambahkan pada denoise wavelet untuk 4 citra noninfected dengan
subband HH1 tanpa variasi (lihat `references/tahapan-penelitian.md` bagian 11).
Prediksi citra test yang terdampak tidak berubah, tetapi rerun E2-E6
disarankan bila hasil harus sepadan penuh dengan kode di HEAD.

Dua perbaikan lain setelah run tidak mengubah training: detail verdict H3 kini
mencantumkan angka parameter dan latency, dan Grad-CAM memakai `-logit` untuk
prediksi Normal pada model satu logit. Figure Grad-CAM di
`experiments/results/` masih dari versi lama, sehingga panel Normal dapat
kosong dan perlu diregenerasi dari checkpoint. Checkpoint tidak menyimpan
config atau tipe attention; muat model memakai `run_summary.json` atau
`experiments.yml`.

Audit data setelah run: dedup md5 tidak menangkap salinan hasil resize/crop.
Dengan korelasi 64x64 >= 0,995, 57/600 citra test punya near-duplicate di
train (68 dengan val), 11 di antaranya berlabel berlawanan; 34/812 citra
noninfected unik near-identik dengan citra infected. Tanpa 68 citra tersebut,
accuracy E1/E5/E3-E6 menjadi 98,87/99,44/96,62 dan verdict tidak berubah.
Bounding box ROI juga mencakup lebih dari 95% frame pada 492/600 citra test,
sehingga crop ROI sebagian besar tidak memotong. Rincian di
`references/tahapan-penelitian.md` bagian 11.1 dan 11.2.

### Perbandingan Attention Mechanisms

Perbandingan attention diwakili eksperimen E4 (self-attention) vs E6 (CBAM)
pada tabel matriks E1-E6 di atas; kolom parameter dan latency tercatat di
`experiments/results/results.csv`. Semua mekanisme dijalankan pada peta fitur
2D sebelum pooling: SE/CBAM pada (B, C, H, W); self-attention/transformer
pada token spasial (B, H·W, C) dengan N = H·W (7×7 untuk input 224px).
Detail prinsip kerja tiap mekanisme:

| Mekanisme          | Parameters | Prinsip Kerja                  |
| ------------------ | ---------- | ------------------------------ |
| **Self-Attention** | 11.4M      | Q, K, V multi-head (Artikel 1) |
| **CBAM**           | 7.3M       | Channel + Spatial attention    |

## Dataset

- **PCOS-XAI Ultrasound Dataset** - 11.784 gambar mentah (6.784 PCOS, 5.000 sehat)
- Deduplikasi md5 otomatis sebelum split: 3.996 gambar unik (3.184 PCOS, 812 sehat; rasio kelas 4:1, laporkan specificity/F1/AUC bersama akurasi)
- Source: [Kaggle](https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset)
- Tahapan lengkap dari pengambilan data sampai artefak (termasuk verifikasi data dan checklist Colab): [references/tahapan-penelitian.md](references/tahapan-penelitian.md)
