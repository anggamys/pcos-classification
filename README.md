# PCOS Classification

Klasifikasi Polycystic Ovary Syndrome (PCOS) dari citra ultrasound menggunakan DenseNet-121 dengan Self-Attention mechanism.

Duplikasi dari artikel:
> Tiwari, S., Shukla, A., & Sharma, A.K. (2026). *Attention-guided lightweight deep learning architecture for classification of polycystic ovary syndrome from ultrasound images.* Intelligence-Based Medicine, 13, 100358. DOI: [10.1016/j.ibmed.2026.100358](https://doi.org/10.1016/j.ibmed.2026.100358)

## Struktur Proyek

```
pcos-classification/
├── datasets/
│   ├── infected/          # 6.784 citra PCOS
│   └── noninfected/       # 5.000 citra sehat
├── src/
│   ├── dataset.py         # Custom Dataset & DataLoader
│   ├── preprocessing.py   # Wavelet denoising (BayesShrink)
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Metrics & visualisasi
│   └── models/
│       ├── attention.py   # Self-Attention module
│       └── densenet121.py # DenseNet-121 + Attention
├── checkpoints/           # Model checkpoints
├── requirements.txt
└── main.py
```

## Pipeline

### 1. Preprocessing
- Resize 224 x 224, normalisasi [0, 1]
- Wavelet denoising (BayesShrink) - opsional
- Augmentasi: RandomRotation(15), RandomHorizontalFlip, RandomResizedCrop

### 2. Arsitektur
- **Backbone**: DenseNet-121 pretrained (ImageNet)
- **Self-Attention**: Multi-head attention (8 heads) pada feature maps
- **Classifier**: FC(1024, 256) -> ReLU -> Dropout(0.3) -> FC(256, 1) -> Sigmoid

### 3. Training
- Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
- Loss: BCEWithLogitsLoss
- Early stopping: patience=10
- Scheduler: ReduceLROnPlateau

### 4. Evaluasi
- Accuracy, Precision, Recall, F1-Score, Specificity, AUC
- Confusion matrix, ROC curve, Precision-Recall curve

## Instalasi

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Penggunaan

### CLI

```bash
# Default settings
python main.py

# Custom training
python main.py --epochs 50 --lr 5e-5 --batch-size 64 --denoise

# Resume with specific dataset
python main.py --data-dir /path/to/data --save-dir my_checkpoints --patience 5
```

### Semua Argumen

| Argumen | Default | Deskripsi |
|---|---|---|
| `--data-dir` | `datasets` | Path dataset directory |
| `--batch-size` | `32` | Batch size |
| `--image-size` | `224` | Image resize size |
| `--epochs` | `30` | Max training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--weight-decay` | `1e-4` | Weight decay |
| `--patience` | `10` | Early stopping patience |
| `--save-dir` | `checkpoints` | Checkpoint save directory |
| `--denoise` | `false` | Enable wavelet denoising |
| `--no-pretrained` | `false` | Disable pretrained weights |
| `--num-workers` | `2` | DataLoader workers |

### Python API

```python
from src.dataset import create_dataloaders
from src.models.densenet121 import DenseNet121Attention
from src.train import train
from src.evaluate import evaluate_model, compute_metrics

train_loader, val_loader, test_loader = create_dataloaders('datasets', batch_size=32)
model = DenseNet121Attention(num_classes=1, pretrained=True)

config = {'epochs': 30, 'lr': 1e-4, 'weight_decay': 1e-4, 'patience': 10, 'save_dir': 'checkpoints'}
history = train(model, train_loader, val_loader, config)
```

## Hasil (Artikel)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| DenseNet-121 + Attention | 99.13% | 1.000 | 0.9851 | 1.0000 | 0.9997 |

## Referensi

- Dataset: [PCOS-XAI Ultrasound Dataset](https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset)
