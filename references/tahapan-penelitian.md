# Tahapan Penelitian dan Pengambilan Data

Dokumen ini mencatat urutan kerja dari pengambilan data sampai artefak hasil,
supaya prosesnya bisa diulang dan dipertanggungjawabkan di laporan. Narasi
riset (masalah, gap, hipotesis) ada di
[kajian-literatur-gap-hipotesis.md](kajian-literatur-gap-hipotesis.md);
dokumen ini hanya soal prosedur.

| Tahap               | Keluaran                                                 |
| ------------------- | -------------------------------------------------------- |
| 1. Pengambilan data | 11.784 citra mentah (6.784 PCOS, 5.000 sehat)            |
| 2. Verifikasi data  | jumlah file dan struktur direktori terkonfirmasi         |
| 3. Deduplikasi      | 3.996 citra unik, 7.788 salinan dibuang                  |
| 4. Split            | train/val/test = 2.797/599/600                           |
| 5. Praproses        | citra siap model per konfigurasi eksperimen              |
| 6. Training         | 6 model (E1-E6) dengan protokol seragam                  |
| 7. Evaluasi         | metrik medis, Grad-CAM, metrik kualitas PCD              |
| 8. Dokumentasi      | `results.csv`, `matrix_summary.json`, `run_summary.json` |

## 1. Sumber dan pengambilan data

Dataset: PCOS-XAI Ultrasound Dataset
(<https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset>),
11.784 citra USG: 6.784 kelas PCOS (infected) dan 5.000 kelas sehat
(noninfected).

Ambil datanya sekali, lalu simpan di luar repo (dataset tidak masuk git):

```bash
kaggle datasets download -d ibadeus/pcos-xai-ultrasound-dataset
unzip pcos-xai-ultrasound-dataset.zip -d datasets/PCOS
```

Di Colab, taruh di Google Drive agar tidak perlu unduh ulang tiap sesi:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Path yang dipakai di seluruh perintah dokumen ini:

```bash
DATA_DIR=/content/drive/MyDrive/data-latih/PCOS   # Colab
DATA_DIR=datasets/PCOS                            # lokal
```

## 2. Verifikasi data sebelum eksperimen

Loader mensyaratkan struktur dan ekstensi persis berikut. Nama kelas dan
huruf kecilnya berpengaruh, begitu juga `.jpg` (format lain tidak dibaca):

```
PCOS/
  infected/*.jpg       label 1
  noninfected/*.jpg    label 0
```

Jalankan pemeriksaan ini sebelum training. Angka yang diharapkan: 6.784,
5.000, dan total 11.784.

```bash
find "$DATA_DIR" -maxdepth 2 -name "*.jpg" | wc -l
ls "$DATA_DIR/infected" | wc -l
ls "$DATA_DIR/noninfected" | wc -l
```

Kalau jumlahnya tidak cocok atau direktori tidak ditemukan, jangan lanjut.
Guard di `src/dataset.py` akan menggagalkan run dengan `FileNotFoundError`
yang menyebut path dan kelas yang bermasalah, dan pesan itu ikut tersimpan di
`stdout.log`/`training.log`. Kegagalan seperti ini normalnya berarti Drive
belum di-mount, nama folder berbeda, atau isi file ada di subfolder lain.

## 3. Deduplikasi

Split acak level citra tanpa dedup bisa menaruh salinan byte-identik di train
dan test sekaligus, sehingga metrik menggelembung. Karena itu salinan
byte-identik dibuang lebih dulu dengan hash md5, keep-first, dijalankan
otomatis sebelum split (`--no-dedupe` hanya untuk kontrol, bukan untuk hasil
utama).

Hasil pada dataset ini:

| Kondisi                 | Jumlah                                  |
| ----------------------- | --------------------------------------- |
| Citra mentah            | 11.784                                  |
| Salinan identik dibuang | 7.788                                   |
| Citra unik              | 3.996 (3.184 infected, 812 noninfected) |

Dedup juga menurunkan duplikat lintas kelas. Bila proses ini membuat sebuah
kelas hilang seluruhnya, dataset ditolak dengan `ValueError` alih-alih
melanjutkan training yang tidak bermakna.

Konsekuensi yang harus dilaporkan: rasio kelas menjadi sekitar 4:1, sehingga
akurasi saja menyesatkan dan specificity, F1, serta AUC wajib dilaporkan
bersamaan.

## 4. Split

Split 70/15/15 dengan seed 42, sekali untuk semua eksperimen:

| Subset   | Jumlah |
| -------- | ------ |
| Train    | 2.797  |
| Validasi | 599    |
| Test     | 600    |

Seed yang sama juga mengatur shuffle, augmentasi, dan inisialisasi model.
Karena itu seluruh eksperimen memakai sampel train/val/test yang identik dan
perbedaan hasil hanya berasal dari perlakuan yang diuji.

## 5. Praproses

Urutan pipeline dan titik variasinya:

1. Baca citra dan konversi RGB.
2. Wavelet denoising BayesShrink (`--denoise`): dekomposisi `db4` level 3,
   sigma noise dari median koefisien HH1, soft threshold adaptif per subband,
   rekonstruksi di-crop ke ukuran asli.
3. Enhancement CLAHE (`--enhance clahe`), opsional.
4. Segmentasi folikel unsupervised (`--segment otsu` atau `adaptive`):
   threshold, morfologi, hapus komponen tepi, fallback persentil bila region
   kosong.
5. Mode input (`--input-mode full|roi|masked`): citra penuh, crop ROI dengan
   padding, atau citra dengan latar disamarkan.
6. Resize 224x224 dan normalisasi ImageNet.

Praproses deterministik di-cache per citra selama satu run, sedangkan
augmentasi training (rotasi 15 derajat, flip horizontal, random resized crop)
tetap diacak per epoch.

## 6. Training

Protokol sama untuk semua eksperimen (satu sumber nilai di
[experiments/experiments.yml](../experiments/experiments.yml)):

| Komponen       | Nilai                                               |
| -------------- | --------------------------------------------------- |
| Model          | DenseNet-121 + attention (self-attention atau CBAM) |
| Optimizer      | Adam, lr 1e-4, weight decay 1e-4                    |
| Loss           | BCEWithLogitsLoss dengan pos_weight                 |
| Batch size     | 64                                                  |
| Epoch maksimum | 30                                                  |
| Early stopping | patience 10 pada val_loss                           |
| Scheduler      | ReduceLROnPlateau                                   |
| Presisi        | AMP float16 pada GPU                                |
| Seed           | 42                                                  |

`pos_weight` dihitung dari label split train saja, yaitu n_negatif/n_positif.
Pada split ini nilainya 0,2509 (train: 2.236 infected, 561 noninfected),
sehingga kontribusi gradien kelas mayoritas diturunkan tanpa resampling data
dan tanpa kebocoran val/test. Nilai yang benar-benar dipakai tersimpan di
checkpoint dan `run_summary.json`; `--no-pos-weight` tersedia untuk kontrol.

## 7. Evaluasi

Dua sudut evaluasi dari run yang sama:

- **PCD (kualitas praproses):** PSNR/SSIM denoise terhadap citra asli,
  PSNR/SSIM CLAHE, histogram dan spektrum, overlay mask, statistik region
  (jumlah, coverage, luas rata-rata dan maksimum). Dihitung tanpa training:

  ```bash
  python experiments/evaluate_quality.py --data-dir "$DATA_DIR" \
    --samples-per-class 100
  ```

- **Analisis Citra Medis (performa diagnostik):** accuracy, precision,
  recall, F1, specificity, AUC, average precision, confusion matrix, ROC,
  dan Grad-CAM. Verdict H1-H3 dihitung dari metrik ini dengan toleransi
  0,005 dan aturan yang sudah ditetapkan sebelum eksperimen.

## 8. Artefak dan ketelusuran

Tiap eksperimen menulis ke `experiments/results/<ID>/`:

| Berkas             | Isi                                                                                                 |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| `stdout.log`       | seluruh keluaran konsol, termasuk stderr dan traceback bila gagal                                   |
| `training.log`     | salinan konsol dari sisi `main.py` (tee)                                                            |
| `run_summary.json` | waktu, commit git, environment, config, pos_weight, ukuran dataset, history, metrik, daftar artefak |
| `best_model.pth`   | bobot terbaik, optimizer state, pos_weight                                                          |
| `*.png`            | training history, confusion matrix, ROC, Grad-CAM                                                   |

Level matriks di `experiments/results/`:

- `results.csv`: satu baris per eksperimen (konfigurasi, metrik, parameter,
  latency, pos_weight, returncode).
- `matrix_summary.json`: seluruh baris, verdict H1-H3, dan salinan tiap
  `run_summary.json`.

Saat dijalankan lewat `run_matrix.py`, keluaran child process juga tampil
live di terminal (child dijalankan dengan `-u`, stdout dan stderr digabung),
jadi training bisa dipantau tanpa menunggu run selesai.

## 9. Urutan eksekusi

Lokal (CPU cukup untuk cek pipeline, bukan untuk hasil):

```bash
python -m unittest tests.test_failure_paths tests.test_pos_weight tests.test_streaming
python main.py --data-dir "$DATA_DIR" --epochs 1 --batch-size 4 --num-workers 0
```

Colab T4 (urutan yang dipakai untuk hasil):

```bash
# 1. Bukti kualitas praproses (cepat, tanpa training)
python experiments/evaluate_quality.py --data-dir "$DATA_DIR" \
  --samples-per-class 100

# 2. Matriks penuh E1-E6
python experiments/run_matrix.py --config experiments/experiments.yml \
  --data-dir "$DATA_DIR" --epochs 30

# 3. Bila sesi terbatas, kerjakan bertahap lalu gabungkan hasilnya
python experiments/run_matrix.py --config experiments/experiments.yml \
  --data-dir "$DATA_DIR" --epochs 30 --only E1,E2
```

Bila hanya sebagian eksperimen yang dijalankan ulang, `results.csv` dan
`matrix_summary.json` akan ditimpa dengan subset tersebut. Simpan atau
gabungkan berkas dari sesi sebelumnya bila ingin riwayat lengkap.

## 10. Checklist sebelum run

- [ ] Drive sudah di-mount (Colab) dan `echo $DATA_DIR` menunjuk path benar.
- [ ] Tiga perintah verifikasi di bagian 2 menghasilkan 6.784, 5.000, 11.784.
- [ ] Working tree bersih dan commit sudah di-push, agar hash commit di
      `run_summary.json` menunjuk kode yang benar-benar dipakai.
- [ ] Kuota Colab cukup untuk seluruh eksperimen (sekitar 50-60 menit per run
      30 epoch di T4).
- [ ] Folder `experiments/results` dari run lama dipindahkan bila ingin
      membandingkan, karena nama folder per eksperimen akan dipakai ulang.

## 11. Catatan replikasi

Hasil lama (accuracy 99,72%) tidak sah dan tidak boleh dikutip: dibuat sebelum
deduplikasi dan sebelum perbaikan penempatan attention. Checkpoint lama juga
tidak kompatibel dengan arsitektur sekarang. Semua angka E1-E6 harus berasal
dari run baru setelah commit pipeline yang berlaku.
