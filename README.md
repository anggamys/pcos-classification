# Pengaruh Wavelet Denoising dan Segmentasi Folikel terhadap Klasifikasi PCOS dengan DenseNet-121 + Attention dan Grad-CAM

Klasifikasi Polycystic Ovary Syndrome (PCOS) dari citra ultrasound: mengukur pengaruh preprocessing wavelet denoising (BayesShrink) dan segmentasi folikel unsupervised terhadap performa DenseNet-121 + Attention, dengan interpretabilitas Grad-CAM.

Topik ini dipakai untuk **dua tugas akhir matakuliah sekaligus**: Pengolahan Citra Digital (PCD) dan Analisis Citra Medis. Satu pipeline dan dataset yang sama, dengan sudut evaluasi berbeda: PCD fokus pada kualitas preprocessing/segmentasi, Analisis Citra Medis fokus pada performa diagnostik dan interpretabilitas.

## Referensi

1. Tiwari, S., Shukla, A., & Sharma, A.K. (2026). *Attention-guided lightweight deep learning architecture for classification of polycystic ovary syndrome from ultrasound images.* Intelligence-Based Medicine, 13, 100358. DOI: [10.1016/j.ibmed.2026.100358](https://doi.org/10.1016/j.ibmed.2026.100358)

2. Sundari, M.S., et al. (2025). *Transfer Learning-Enhanced CNN Model for Integrative Ultrasound and Biomarker-Based Diagnosis of Polycystic Ovarian Disease.* Scientific Reports, 15:34519. DOI: [10.1038/s41598-025-17711-w](https://doi.org/10.1038/s41598-025-17711-w)

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
│   ├── preprocessing.py       # Wavelet denoising (BayesShrink)
│   ├── segmentation.py        # Segmentasi folikel unsupervised (Otsu/adaptive + morfologi)
│   ├── image_quality.py       # Enhancement CLAHE, metrik PSNR/SSIM
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

| Mekanisme | Parameters | Prinsip Kerja |
|-----------|------------|---------------|
| **Self-Attention** | 11.4M | Q, K, V multi-head (Artikel 1) |
| **SE-Net** | 7.3M | Channel-wise attention |
| **CBAM** | 7.3M | Channel + Spatial attention |
| **Transformer** | 32.4M | Patch-based attention |

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

| Argumen | Default | Deskripsi |
|---|---|---|
| `--data-dir` | `datasets` | Path dataset directory |
| `--batch-size` | `64` | Batch size |
| `--image-size` | `224` | Image resize size |
| `--epochs` | `30` | Max training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--weight-decay` | `1e-4` | Weight decay |
| `--patience` | `10` | Early stopping patience |
| `--save-dir` | `checkpoints` | Checkpoint save directory |
| `--denoise` | `false` | Enable wavelet denoising |
| `--enhance` | `none` | Contrast enhancement: none / clahe |
| `--segment` | `none` | Segmentasi folikel: none / otsu / adaptive |
| `--input-mode` | `full` | Input model: full / roi / masked |
| `--seg-pad` | `8` | Padding ROI crop (px) |
| `--no-pretrained` | `false` | Disable pretrained weights |
| `--num-workers` | `2` | DataLoader workers |
| `--attention` | `self_attention` | Attention: self_attention / se_net / cbam / transformer |
| `--dropout` | `0.3` | Dropout rate |
| `--gradcam` | `false` | Generate Grad-CAM |
| `--gradcam-samples` | `5` | Jumlah sampel Grad-CAM |

## Pipeline

### 1. Preprocessing
- Resize 224×224, normalisasi [0,1]
- Wavelet denoising (BayesShrink) - opsional (`--denoise`)
- Enhancement CLAHE - opsional (`--enhance clahe`)
- Segmentasi folikel unsupervised: Otsu/adaptive + morfologi + hapus komponen tepi (`--segment`)
- Mode input: citra penuh, ROI crop, atau masked (`--input-mode`)
- Augmentasi: RandomRotation(15), RandomHorizontalFlip, RandomResizedCrop

### 2. Matriks Eksperimen (Tugas Akhir PCD + Citra Medis)

```bash
# Jalankan 6 eksperimen E1-E6 (hasil ke experiments/results/results.csv)
python experiments/run_matrix.py --data-dir datasets --epochs 30

# Sebagian saja
python experiments/run_matrix.py --data-dir datasets --epochs 30 --only E1,E2
```

| ID | Preprocessing | Segmentasi/ROI | Attention |
|---|---|---|---|
| E1 | baseline | none / full | self_attention |
| E2 | BayesShrink | none / full | self_attention |
| E3 | BayesShrink + CLAHE | none / full | self_attention |
| E4 | BayesShrink + CLAHE | otsu / roi | self_attention |
| E5 | BayesShrink + CLAHE | otsu / masked | self_attention |
| E6 | BayesShrink + CLAHE | otsu / roi | cbam |

### 3. Training
- Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
- Loss: BCEWithLogitsLoss
- Mixed Precision (AMP) untuk percepatan
- Early stopping: patience=10
- Scheduler: ReduceLROnPlateau

### 4. Evaluasi ganda
- **PCD:** PSNR/SSIM (denoise vs asli), histogram/spektrum, overlay mask, statistik ROI (jumlah region, coverage, luas rata-rata/maksimum).
- **Medis:** Accuracy, Precision, Recall, F1-Score, Specificity, AUC.
- Confusion matrix, ROC curve, Grad-CAM heatmap untuk interpretasi.

### Menjalankan di Colab (T4)

```bash
# 0. Bukti sisi PCD untuk H1 (tanpa training, cepat, bisa jalan di CPU)
python experiments/evaluate_quality.py --data-dir /content/drive/MyDrive/data-latih/PCOS --samples-per-class 100

# Full matrix E1-E6
python experiments/run_matrix.py --data-dir /content/drive/MyDrive/data-latih/PCOS --epochs 30

# Sebagian dulu (disarankan untuk sesi Colab pendek)
python experiments/run_matrix.py --data-dir /content/drive/MyDrive/data-latih/PCOS --epochs 30 --only E1,E2
```

Tiap eksperimen menyimpan checkpoint, plot, `stdout.log`, dan ringkasan metrik ke `experiments/results/results.csv`. Estimasi ±50-60 menit per run 30 epoch di T4.

## Hasil

### Replikasi Artikel 1: DenseNet-121 + Self-Attention
| Metrik | Hasil | Artikel |
|---|---|---|
| Accuracy | 99.72% | 99.13% |
| Precision | 100.0% | 100.0% |
| Recall | 99.50% | 98.51% |
| F1-Score | 99.75% | 100.0% |
| Specificity | 100.0% | - |
| AUC | 100.0% | 99.97% |

### Matriks Eksperimen E1-E6 (diisi dari `experiments/results/results.csv`)
> Status: menunggu training di Colab

| ID | Accuracy | F1 | AUC | Catatan |
|---|---|---|---|---|
| E1 | - | - | - | Baseline |
| E2 | - | - | - | + BayesShrink |
| E3 | - | - | - | + CLAHE |
| E4 | - | - | - | + ROI Otsu |
| E5 | - | - | - | + Masked Otsu |
| E6 | - | - | - | + CBAM |

### Perbandingan Attention Mechanisms
Perbandingan attention diwakili eksperimen E4 (self-attention) vs E6 (CBAM)
pada tabel matriks E1-E6 di atas; kolom parameter dan latency tercatat di
`experiments/results/results.csv`. Detail prinsip kerja tiap mekanisme:

| Mekanisme | Parameters | Prinsip Kerja |
|-----------|------------|---------------|
| **Self-Attention** | 11.4M | Q, K, V multi-head (Artikel 1) |
| **CBAM** | 7.3M | Channel + Spatial attention |

## Dataset

- **PCOS-XAI Ultrasound Dataset** - 11.784 gambar (6.784 PCOS, 5.000 sehat)
- Source: [Kaggle](https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset)
