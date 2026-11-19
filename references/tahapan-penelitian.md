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
2. Enhancement CLAHE (`--enhance clahe`), opsional.
3. Wavelet denoising BayesShrink (`--denoise`): dekomposisi `db4` level 3,
   sigma noise dari median koefisien HH1, soft threshold adaptif per subband,
   rekonstruksi di-crop ke ukuran asli. Urutan CLAHE lalu denoise dipakai
   konsisten oleh training dan evaluasi kualitas.
4. Segmentasi folikel unsupervised (`--segment otsu` atau `adaptive`):
   threshold, morfologi, hapus komponen tepi, fallback persentil bila cakupan
   melebihi 0,5. Mask selalu dihitung pada skala kerja tetap 224x224 agar
   radius morfologi dan `min_area` setara antar citra, lalu di-upscale
   nearest-neighbor ke resolusi asli.
5. Mode input (`--input-mode full|roi|masked`): citra penuh, crop ROI dengan
   padding, atau citra dengan latar disamarkan. Bila mask kosong, ROI dan
   masked sama-sama memakai citra asli; mode masked tidak boleh menghasilkan
   citra hitam.
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
| Device         | auto: CUDA bila tersedia, fallback CPU di lokal     |
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
- [ ] Runtime Colab memakai T4 GPU; default `device: auto` memilih CUDA, dan
      `device: cuda` dipakai bila ingin gagal cepat saat GPU tidak aktif.
- [ ] Tiga perintah verifikasi di bagian 2 menghasilkan 6.784, 5.000, 11.784.
- [ ] Working tree bersih dan commit sudah di-push, agar hash commit di
      `run_summary.json` menunjuk kode yang benar-benar dipakai.
- [ ] Kuota Colab cukup untuk seluruh eksperimen (sekitar 50-60 menit per run
      30 epoch di T4).
- [ ] Folder `experiments/results` dari run lama dipindahkan bila ingin
      membandingkan, karena nama folder per eksperimen akan dipakai ulang.

## 11. Catatan replikasi

Resolusi citra berbeda antar kelas: median infected 279x266 (0,073 megapiksel),
noninfected 848x630 (0,491 megapiksel). Segmentasi karena itu dihitung pada
skala kerja tetap 224x224 agar radius morfologi dan ambang luas setara, lalu
mask di-upscale ke resolusi asli. Perbedaan resolusi tetap dilaporkan sebagai
ancaman validitas karena dapat menjadi isyarat palsu bagi model.

Hasil lama (accuracy 99,72%) tidak sah dan tidak boleh dikutip: dibuat sebelum
deduplikasi dan sebelum perbaikan penempatan attention. Checkpoint lama juga
tidak kompatibel dengan arsitektur sekarang. Semua angka E1-E6 pada bab hasil
berasal dari run baru pada commit `70e0de3` (Colab T4) dan tersimpan di
`experiments/results/`. Run ulang pada konfigurasi yang sama menghasilkan angka
berbeda; protokol agregasi multi-seed ada di bagian 11.3.

Setelah run tersebut, ditemukan peringatan `pywt` pada 4 citra noninfected yang
subband HH1-nya tidak memiliki variasi (sigma noise = 0). Soft threshold
dengan nilai 0 menghasilkan NaN untuk koefisien nol, lalu NaN diam-diam
menjadi 0 saat cast ke uint8. Guard `sigma <= 0` ditambahkan agar kanal
dikembalikan apa adanya. Keempat citra berada di train (3), val (1), dan test
(1). Pada citra test yang terdampak, probabilitas prediksi E2-E6 berubah kurang
dari 0,0005 dan kelas prediksi tetap sama, sehingga verdict tidak berubah.
Rerun E2-E6 tetap disarankan bila hasil harus sepenuhnya sepadan dengan kode
di HEAD.

### 11.1. Audit near-duplicate dan label noise

Dedup md5 hanya menghapus salinan byte-identik. Untuk mengukur sisa duplikasi,
seluruh 3.996 citra unik di-thumbnail ke 64x64 grayscale, dinormalisasi, lalu
dihitung korelasi kosinus antar split. Ambang 1,0 berarti identik dan 0,995
dipakai sebagai batas near-duplicate; pasangan teratas diverifikasi visual.

- 57/600 citra test punya near-duplicate di train (68 bila val dihitung).
- 11 citra (12 dengan val) near-identik dengan citra berlabel berlawanan.
- 34/812 citra noninfected unik (4,2%) near-identik dengan citra infected,
  semuanya berkorelasi >= 0,9997 dan berpola `Image_*.jpg` lawan `image*.jpg`.
- Setelah 68 citra test tersebut dikeluarkan, accuracy E1 98,87%, E5 99,44%,
  E3/E6 96,62%. Verdict H1-H3 tidak berubah.
- Citra near-duplicate berlabel sama diprediksi benar 56/56 oleh E1, E3, E5,
  dan E6; pasangan berlabel konflik menjadi sumber error.

Sumber duplikasi juga tidak seimbang: noninfected 5.000 file mentah menjadi
812 unik (84% duplikat), infected 6.784 menjadi 3.184 unik. Tidak ada grup
md5 lintas kelas, sehingga rasio 80/20 berasal dari dataset sumber, bukan dari
urutan keep-first.

### 11.2. Audit ROI, mask, dan Grad-CAM

- Bounding box gabungan region Otsu mencakup lebih dari 95% frame pada
  492/600 citra test (median 100%). Hanya 3 citra yang crop-nya di bawah 5%
  frame. Mode ROI karena itu sebagian besar identik dengan citra penuh.
- Mask test kosong 24/600 dan coverage di bawah 5% pada 44/600. E5 benar 44/44
  pada subset coverage rendah; kelima error E5 berada pada coverage normal.
  `segment_otsu` hanya punya fallback untuk coverage terlalu besar
  (`max_coverage`), belum ada batas coverage minimum.
- Grad-CAM lama selalu memakai logit PCOS karena model hanya punya satu
  keluaran. Prediksi Normal menghasilkan gradien negatif yang dipotong ReLU:
  3/12 prediksi Normal pada 60 citra test pertama E1 menghasilkan CAM kosong.
  Kode sekarang memakai -logit untuk prediksi Normal; figure lama perlu
  diregenerasi dari checkpoint.
- `best_model.pth` tidak menyimpan config, tipe attention, atau pemetaan kelas,
  hanya `epoch`, `model_state_dict`, `optimizer_state_dict`, `pos_weight`,
  `val_acc`, dan `val_loss`. Rekonstruksi model tetap memerlukan
  `run_summary.json` atau `experiments.yml`.
- Epoch di checkpoint 0-indexed (E1 `epoch=1` untuk `best_epoch=2`), berbeda
  dari log yang 1-indexed.
- `results.csv` membulatkan metrik ke 4 desimal; nilainya konsisten dengan
  `run_summary.json` setelah pembulatan. Verdict H2 dihitung dari nilai bulat
  itu, tetapi margin 0,0017 masih jauh di atas galat pembulatan.

### 11.3. Protokol multi-seed dan agregasi

Tiga run matriks pada konfigurasi, data, dan seed yang sama (16, 18, dan
19 November 2026) menghasilkan angka yang berbeda: accuracy E2 bergerak
94,67 sampai 98,17 dan E3 bergerak 92,33 sampai 96,00, sementara toleransi
verdict 0,5 poin. Perbedaan itu membalik verdict H1 dan H2 antar run. Karena
itu hasil satu run tidak dipakai untuk verdict.

Aturan agregasi berikut ditetapkan sebelum run seed 43 dan 44, tanpa mengubah
kriteria H1-H3:

1. Setiap seed menjalankan keenam eksperimen dengan split, epoch, dan early
   stopping yang sama; hanya seed RNG yang berbeda.
2. Metrik agregat adalah rata-rata aritmetika antar seed. Simpangan baku
   sampel (n-1) dilaporkan sebagai dispersi, bukan interval kepercayaan.
3. Verdict dihitung dengan kriteria H1-H3 yang sama pada baris rata-rata.
4. Status akhir accepted hanya bila verdict agregat accepted dan minimal
   mayoritas seed (`n // 2 + 1`) juga accepted. Aturan ini mencegah satu run
   yang kebetulan lolos menyelamatkan klaim. E5 tetap deskriptif dan varian
   utama H2 tetap E4.
5. Semua seed yang dipakai harus dilaporkan. Seed tidak boleh dipilih
   berdasarkan hasilnya.

Perintah dan layout keluaran:

    python experiments/run_matrix.py --seeds 42,43,44 --only E1,E2
    python experiments/run_matrix.py --aggregate-only

Hasil per seed ada di `experiments/results/seed-<n>/E*/`, ringkasan per seed
di `seed-<n>/results.csv`, dan agregat di `aggregate.csv` serta
`matrix_summary.json`. Agregasi memakai semua folder `seed-*` yang ada,
sehingga seed dapat dijalankan bertahap lalu digabung tanpa training ulang.

Latency diukur sekali per matriks. Model self-attention dan CBAM dibangun
dalam satu proses, lalu diukur bergantian 5 putaran x 20 forward; angka yang
dipakai adalah median, bukan satu pengukuran. Hasilnya disimpan di
`latency.json` dan dipakai bersama semua seed agar H3 tidak membandingkan
sesi pengukuran yang berbeda.

`cudnn.benchmark` dimatikan sejak protokol ini karena autotuning memilih
algoritma berbeda antar run pada seed yang sama. Determinisme penuh tidak
dijamin oleh flag itu; variasi antar seed tetap diukur, bukan diasumsikan nol.
