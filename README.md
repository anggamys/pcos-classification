# PCOS Classification

Klasifikasi Polycystic Ovary Syndrome (PCOS) dari citra ultrasound menggunakan Deep Learning dengan Self-Attention mechanism.

## Referensi

1. Tiwari, S., Shukla, A., & Sharma, A.K. (2026). *Attention-guided lightweight deep learning architecture for classification of polycystic ovary syndrome from ultrasound images.* Intelligence-Based Medicine, 13, 100358. DOI: [10.1016/j.ibmed.2026.100358](https://doi.org/10.1016/j.ibmed.2026.100358)

2. Sundari, M.S., et al. (2025). *Transfer Learning-Enhanced CNN Model for Integrative Ultrasound and Biomarker-Based Diagnosis of Polycystic Ovarian Disease.* Scientific Reports, 15:34519. DOI: [10.1038/s41598-025-17711-w](https://doi.org/10.1038/s41598-025-17711-w)

## Struktur Proyek

```
pcos-classification/
├── datasets/
│   ├── infected/              # 6.784 citra PCOS
│   └── noninfected/           # 5.000 citra sehat
├── src/
│   ├── dataset.py             # Custom Dataset & DataLoader
│   ├── preprocessing.py       # Wavelet denoising (BayesShrink)
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Metrics & visualisasi
│   ├── optimize.py            # Bayesian Optimization (Optuna)
│   ├── gradcam.py             # Grad-CAM visualisasi
│   ├── smote.py               # SMOTE untuk class imbalance
│   └── models/
│       ├── attention.py       # Self-Attention module
│       ├── densenet121.py     # DenseNet-121 + Attention
│       └── efficientnet_b3.py # EfficientNet-B3 + Attention
├── checkpoints/               # Model checkpoints
├── requirements.txt
└── main.py
```

## Fitur

### Model Arsitektur
- **DenseNet-121 + Attention** - 11.4M parameters (default)
- **EfficientNet-B3 + Attention** - Alternatif backbone
- **Self-Attention** - Multi-head attention (8 heads)

### Advanced Features
- **Bayesian Optimization** - Auto-tune hyperparameter dengan Optuna
- **Grad-CAM** - Visual heatmap untuk interpretasi model
- **SMOTE** - Oversampling untuk class imbalance
- **Mixed Precision (AMP)** - Training float16 untuk percepatan
- **Wavelet Denoising** - preprocessing BayesShrink (opsional)

## Instalasi

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Penggunaan

### Basic Training
```bash
# Default (DenseNet-121)
python main.py

# EfficientNet-B3
python main.py --model efficientnet_b3

# Custom training
python main.py --epochs 50 --lr 5e-5 --batch-size 64 --denoise
```

### Advanced Features
```bash
# Bayesian Optimization
python main.py --optimize --n-trials 20

# Grad-CAM visualization
python main.py --gradcam --gradcam-samples 5

# SMOTE untuk class imbalance
python main.py --smote

# Semua fitur combined
python main.py --model efficientnet_b3 --optimize --gradcam --smote --epochs 50
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
| `--no-pretrained` | `false` | Disable pretrained weights |
| `--num-workers` | `2` | DataLoader workers |
| `--model` | `densenet121` | Model: densenet121 / efficientnet_b3 |
| `--dropout` | `0.3` | Dropout rate |
| `--optimize` | `false` | Jalankan Bayesian Optimization |
| `--n-trials` | `20` | Jumlah Optuna trials |
| `--gradcam` | `false` | Generate Grad-CAM |
| `--gradcam-samples` | `5` | Jumlah sampel Grad-CAM |
| `--smote` | `false` | Apply SMOTE |

## Pipeline

### 1. Preprocessing
- Resize 224×224, normalisasi [0,1]
- Wavelet denoising (BayesShrink) - opsional
- Augmentasi: RandomRotation(15), RandomHorizontalFlip, RandomResizedCrop

### 2. Training
- Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
- Loss: BCEWithLogitsLoss
- Mixed Precision (AMP) untuk percepatan
- Early stopping: patience=10
- Scheduler: ReduceLROnPlateau

### 3. Evaluasi
- Accuracy, Precision, Recall, F1-Score, Specificity, AUC
- Confusion matrix, ROC curve
- Grad-CAM heatmap untuk interpretasi

## Hasil

### DenseNet-121 + Attention (Replikasi)
| Metrik | Hasil | Artikel |
|---|---|---|
| Accuracy | 99.72% | 99.13% |
| Precision | 100.0% | 100.0% |
| Recall | 99.50% | 98.51% |
| F1-Score | 99.75% | 100.0% |
| Specificity | 100.0% | - |
| AUC | 100.0% | 99.97% |

### EfficientNet-B3 + Attention (Artikel 2)
| Metrik | Artikel |
|---|---|
| Accuracy | 94.8% |
| Precision | 94.0% |
| Recall | 93.2% |
| F1-Score | 93.6% |
| AUC | 0.97 |

## Dataset

- **PCOS-XAI Ultrasound Dataset** - 11.784 gambar (6.784 PCOS, 5.000 sehat)
- Source: [Kaggle](https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset)
