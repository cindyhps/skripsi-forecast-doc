# Dokumentasi Proyek: Prediksi Parameter Kualitas Air dengan LSTM

> Proyek ini membangun model LSTM untuk memprediksi tiga parameter kualitas air — **Suhu**, **pH**, dan **TDS** — menggunakan data time-series sensor dengan interval 30 menit.

---

## Daftar Isi

1. [Gambaran Umum Proyek](#1-gambaran-umum-proyek)
2. [Struktur File & Urutan Penggunaan](#2-struktur-file--urutan-penggunaan)
3. [Requirements & Persiapan Lingkungan](#3-requirements--persiapan-lingkungan)
4. [File: `data_processing_stage.ipynb`](#4-file-data_processing_stageipynb)
5. [File: `alpha_b.ipynb` — Model Alpha (Multivariat)](#5-file-alpha_bipynb--model-alpha-multivariat)
6. [File: `beta_b.ipynb` — Model Beta (Univariat + Hyperparameter Search)](#6-file-beta_bipynb--model-beta-univariat--hyperparameter-search)
7. [Arsitektur Model LSTM](#7-arsitektur-model-lstm)
8. [Struktur Direktori yang Dihasilkan](#8-struktur-direktori-yang-dihasilkan)
9. [Metrik Evaluasi](#9-metrik-evaluasi)
10. [Catatan Penting](#10-catatan-penting)

---

## 1. Gambaran Umum Proyek

Proyek ini memprediksi nilai **3 langkah ke depan** (t+1, t+2, t+3) untuk setiap parameter kualitas air dari data historis 144 timestep (setara 3 hari dengan interval 30 menit).

| Parameter | Satuan | Keterangan |
|-----------|--------|------------|
| Suhu (`temperature`) | °C | Temperatur air |
| pH (`ph`) | — | Tingkat keasaman air |
| TDS (`tds`) | ppm | Total Dissolved Solids, maks 1000 ppm |

**Alur kerja keseluruhan:**

```
Raw CSV Sensor
     ↓
data_processing_stage.ipynb  →  dataset_b/sequences/combined_sequences_corrected.csv
     ↓
alpha_b.ipynb  →  Model multivariat (1 model untuk semua parameter sekaligus)
beta_b.ipynb   →  Model univariat (1 model per parameter + hyperparameter search)
```

---

## 2. Struktur File & Urutan Penggunaan

Ketiga notebook **harus dijalankan berurutan**:

```
1. data_processing_stage.ipynb   ← Jalankan PERTAMA
2. alpha_b.ipynb                 ← Jalankan KEDUA  (opsional, model alternatif)
3. beta_b.ipynb                  ← Jalankan KETIGA (model utama dengan tuning)
```

> **Penting:** `alpha_b.ipynb` dan `beta_b.ipynb` keduanya bergantung pada output dari `data_processing_stage.ipynb`. Pastikan preprocessing selesai sepenuhnya sebelum menjalankan salah satu dari keduanya.

---

## 3. Requirements & Persiapan Lingkungan

### Versi Python

```
Python 3.12
```

### Instalasi Dependencies

Buat virtual environment (disarankan) lalu install semua library berikut:

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# atau
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

Buat file `requirements.txt` dengan isi berikut:

```txt
numpy>=1.26.0
pandas>=2.1.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.11.0
scikit-learn>=1.3.0
tensorflow>=2.15.0
joblib>=1.3.0
plotly>=5.18.0
kaleido>=0.2.1
h5py>=3.10.0
```

> **Catatan `kaleido`:** Diperlukan oleh Plotly untuk menyimpan visualisasi sebagai file gambar (`.png`). Jika tidak diinstall, `fig.write_image()` akan error.

### Verifikasi Instalasi

Jalankan cell pertama di notebook mana pun untuk memverifikasi:

```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")  # Harus >= 2.15.0
```

### Dataset Input yang Dibutuhkan

Sebelum menjalankan notebook, siapkan file berikut:

```
dataset/
└── dataset_unprocessed.csv     ← File sensor mentah (WAJIB ada)
```

Format kolom CSV yang diharapkan:

| Kolom CSV Asli | Keterangan |
|----------------|------------|
| `waktu` | Timestamp format `DD/MM/YYYY HH:MM` |
| `water_temp_c` | Suhu air (°C) |
| `ph_value` | Nilai pH |
| `tds_ppm` | Nilai TDS (ppm) |

---

## 4. File: `data_processing_stage.ipynb`

### Tujuan

Mengolah data sensor mentah menjadi sequence siap-latih untuk model LSTM. Ini adalah **langkah fondasi** seluruh pipeline.

### Konfigurasi Utama

```python
DATA_POINTS_TOTAL  = 4320   # Total data yang digunakan (90 hari × 48 titik/hari)
WINDOW_SIZE        = 144    # Panjang input sequence (3 hari lookback)
FORECAST_HORIZON   = 3      # Prediksi 3 langkah ke depan (t+1, t+2, t+3)
STEP_SIZE          = 1      # Sliding window bergeser 1 timestep
TRAIN_RATIO        = 0.70   # 70% training
VAL_RATIO          = 0.20   # 20% validasi
TEST_RATIO         = 0.10   # 10% testing
```

### Tahapan Preprocessing

**Step 1 — Load & Rename Kolom**
Membaca `dataset/dataset_unprocessed.csv`, mengganti nama kolom ke format standar (`time`, `temperature`, `ph`, `tds`), mengkonversi tipe data numerik.

**Step 2 — Filter Interval 30 Menit**
Hanya menyimpan baris dengan menit ke-0 dan ke-30 (`MINUTE_FILTER = [0, 30]`) untuk memastikan konsistensi interval.

**Step 3 — Handle Missing Values**
Interpolasi time-based untuk mengisi nilai yang hilang, dilanjutkan forward-fill dan backward-fill untuk edge case.

**Step 4 — Penanganan TDS Tidak Valid**
Nilai TDS negatif di-clip ke 0, nilai TDS = 0 (noise sensor) diganti dengan interpolasi linear.

**Step 5 — IQR Outlier Handling**
Deteksi outlier menggunakan metode IQR (Q1 − 1.5×IQR hingga Q3 + 1.5×IQR), nilai di luar batas di-cap ke batas tersebut.

**Step 6 — Batas TDS Maksimum 1000 ppm**
Nilai TDS di atas 1000 ppm di-cap dan di-rescale ulang.

**Step 7 — Min-Max Scaling**
Normalisasi semua fitur ke rentang [0, 1] menggunakan `MinMaxScaler`. Scaler di-fit **hanya pada data training** untuk mencegah data leakage.

**Step 8 — Sliding Window Sequence Generation**
Membuat sequence input-output:
- Input `X`: shape `(N, 144, 3)` — 144 timestep, 3 fitur
- Target `y`: shape `(N, 3, 3)` — 3 timestep ke depan, 3 fitur

**Step 9 — Splitting & Saving**
Membagi data secara kronologis (tidak acak) dan menyimpan ke format NPZ + CSV.

### Output yang Dihasilkan

```
dataset/
├── dataset_cleaned_raw.csv
└── sequences/
    ├── combined_sequences_corrected.csv    ← Input utama untuk alpha_b & beta_b
    ├── X_train.npy / y_train.npy
    ├── X_val.npy   / y_val.npy
    ├── X_test.npy  / y_test.npy
    ├── sequences.h5
    ├── sequence_timeline.csv
    └── metadata.json

dataset_b/
├── scaled_data.npz             ← Data scaled tersimpan
├── scaler_X.save               ← Scaler untuk fitur input
└── scaler_y.save               ← Scaler untuk target

models/
├── scalers.pkl
└── scaler_info.pkl

dataset/split_data/
├── data_splits_with_labels.csv
└── split_metadata.json
```

---

## 5. File: `alpha_b.ipynb` — Model Alpha (Multivariat)

### Tujuan

Melatih **satu model LSTM tunggal** yang memprediksi ketiga parameter (Suhu, pH, TDS) **sekaligus** dalam satu output multivariat. Ini adalah pendekatan model terpadu dengan arsitektur 2 layer LSTM.

### Cara Menjalankan

1. Pastikan `data_processing_stage.ipynb` sudah selesai dijalankan
2. Buka `alpha_b.ipynb`
3. Jalankan semua cell dari atas ke bawah

### Konfigurasi Model

```python
LEARNING_RATE = 0.0005
BATCH_SIZE    = 32
EPOCHS        = 100000   # EarlyStopping akan berhenti lebih awal
```

Callbacks yang digunakan:
- `ReduceLROnPlateau` — menurunkan learning rate 50% jika `val_loss` tidak membaik selama 5 epoch
- `ModelCheckpoint` — menyimpan bobot terbaik otomatis ke `models_alpha/alpha_b.keras`
- `EarlyStopping` — berhenti training jika `val_loss` tidak membaik selama 20 epoch

### Output yang Dihasilkan

```
models_alpha/
├── alpha_b.keras               ← Checkpoint bobot terbaik selama training
├── alpha_b_final.keras         ← Model final tersimpan manual
├── history_b.pkl               ← History training (Python pickle)
└── history_b.csv               ← History training (CSV)

results/alpha_b/
├── test_predictions_B.png      ← Prediksi vs aktual (sample acak)
├── test_prediction_combined_b.png  ← Prediksi vs aktual (full test set)
├── error_distribution_B.png    ← Distribusi error residual
└── prediksi_144_kedepan_full_B.png ← Forecast 144 langkah ke depan
```

---

## 6. File: `beta_b.ipynb` — Model Beta (Univariat + Hyperparameter Search)

### Tujuan

Melatih **tiga model LSTM terpisah** — satu per parameter (Suhu, pH, TDS) — dengan **pencarian hyperparameter exhaustive** untuk menemukan kombinasi terbaik. Pendekatan ini memberikan fleksibilitas lebih tinggi karena setiap parameter dapat memiliki arsitektur optimal sendiri.

### Cara Menjalankan

1. Pastikan `data_processing_stage.ipynb` sudah selesai dijalankan
2. Buka `beta_b.ipynb`
3. Jalankan semua cell dari atas ke bawah

> **Peringatan waktu:** Notebook ini menjalankan **30 kombinasi hyperparameter** × 3 parameter = **90 training run**. Estimasi waktu sangat bergantung pada hardware yang digunakan.

### Grid Hyperparameter Search

```python
units_list   = [16, 32, 64, 128, 256]   # Jumlah unit LSTM
dropout_list = [0.1, 0.2]               # Dropout rate
lr_list      = [0.001, 0.0005, 0.0001]  # Learning rate

# Total kombinasi: 5 × 2 × 3 = 30 kombinasi per parameter
```

### Konfigurasi Training

```python
BATCH_SIZE = 32
EPOCHS     = 100000   # Dibatasi EarlyStopping
```

Callbacks per run:
- `ReduceLROnPlateau` — patience=5, factor=0.5
- `EarlyStopping` — patience=10, restore best weights

### Pemilihan Model Terbaik

Untuk setiap parameter, model dengan **validation loss (MSE) terendah** di antara 30 kombinasi yang dipilih sebagai model terbaik.

### Output yang Dihasilkan

```
models_beta/
├── beta_b_suhu_final.keras             ← Model terbaik untuk Suhu
├── beta_b_ph_final.keras               ← Model terbaik untuk pH
├── beta_b_tds_final.keras              ← Model terbaik untuk TDS
├── dataset_b_suhu_best.keras           ← Checkpoint terbaik selama search
├── dataset_b_ph_best.keras
├── dataset_b_tds_best.keras
├── history_b_suhu.pkl / .csv           ← History model terbaik Suhu
├── history_b_ph.pkl / .csv             ← History model terbaik pH
├── history_b_tds.pkl / .csv            ← History model terbaik TDS
├── metrics_test_b.csv                  ← Tabel metrik evaluasi test set
├── train_val_loss_curves_b.png         ← Kurva loss training vs validasi
├── pred_vs_actual_test_b.png           ← Prediksi vs aktual test set
├── error_distribution_b.png           ← Distribusi error residual
└── future_forecast_144_b.png          ← Forecast 144 langkah ke depan
```

---

## 7. Arsitektur Model LSTM

### Model Alpha — Arsitektur 2 Layer (Multivariat)

```
Input: (batch, 144 timesteps, 3 fitur)
         ↓
   LSTM(256 units, return_sequences=True)
         ↓
   Dropout(0.2)
         ↓
   LSTM(128 units, return_sequences=False)
         ↓
   Dropout(0.2)
         ↓
   Dense(9)         ← 3 langkah × 3 parameter = 9 output
         ↓
   Reshape(3, 3)
         ↓
Output: (batch, 3 langkah, 3 parameter)
```

**Loss function:** MSE  
**Optimizer:** Adam (lr=0.0005)  
**Metrik:** RMSE, MAE, MAPE, SMAPE, R²

---

### Model Beta — Arsitektur 1 Layer (Univariat, per parameter)

```
Input: (batch, window_size timesteps, 1 fitur)
         ↓
   LSTM(units)        ← units dipilih dari {16, 32, 64, 128, 256}
         ↓
   Dropout(rate)      ← rate dipilih dari {0.1, 0.2}
         ↓
   Dense(3)           ← prediksi t+1, t+2, t+3 untuk 1 parameter
         ↓
Output: (batch, 3 langkah)
```

**Loss function:** MSE  
**Optimizer:** Adam (lr dipilih dari {0.001, 0.0005, 0.0001})  
**Metrik:** RMSE, MAE, MAPE, SMAPE, R²

---

### Perbandingan Alpha vs Beta

| Aspek | Alpha | Beta |
|-------|-------|------|
| Pendekatan | Multivariat (1 model) | Univariat (3 model) |
| Jumlah model | 1 | 3 (per parameter) |
| Input | 3 fitur sekaligus | 1 fitur per model |
| Hyperparameter | Fixed | Grid search (30 kombinasi) |
| Kompleksitas training | Lebih cepat | Lebih lama (90 run) |
| Fleksibilitas | Rendah | Tinggi |
| Lapisan LSTM | 2 (256 → 128) | 1 (dicari otomatis) |

---

## 8. Struktur Direktori yang Dihasilkan

Setelah menjalankan semua notebook, struktur folder akan terlihat seperti berikut:

```
project_root/
├── dataset/
│   ├── dataset_unprocessed.csv         ← INPUT (harus ada)
│   ├── dataset_cleaned_raw.csv
│   ├── sequences/
│   │   ├── combined_sequences_corrected.csv
│   │   ├── X_train.npy, y_train.npy
│   │   ├── X_val.npy, y_val.npy
│   │   ├── X_test.npy, y_test.npy
│   │   └── metadata.json
│   └── split_data/
│       ├── data_splits_with_labels.csv
│       └── split_metadata.json
│
├── dataset_b/
│   ├── scaled_data.npz
│   ├── scaler_X.save
│   └── scaler_y.save
│
├── models/
│   ├── scalers.pkl
│   └── scaler_info.pkl
│
├── models_alpha/
│   ├── alpha_b_final.keras
│   └── history_b.csv
│
├── models_beta/
│   ├── beta_b_suhu_final.keras
│   ├── beta_b_ph_final.keras
│   ├── beta_b_tds_final.keras
│   └── metrics_test_b.csv
│
├── results/alpha_b/
│   └── *.png
│
├── visualizations/
│   └── *.png
│
├── data_processing_stage.ipynb
├── alpha_b.ipynb
└── beta_b.ipynb
```

---

## 9. Metrik Evaluasi

Kedua model menggunakan 5 metrik evaluasi berikut:

| Metrik | Nama Lengkap | Keterangan |
|--------|-------------|------------|
| **RMSE** | Root Mean Squared Error | Sensitif terhadap error besar |
| **MAE** | Mean Absolute Error | Error rata-rata dalam satuan asli |
| **MAPE** | Mean Absolute Percentage Error | Error dalam persentase |
| **SMAPE** | Symmetric MAPE | MAPE versi simetris, lebih stabil |
| **R²** | Coefficient of Determination | Seberapa baik model menjelaskan variansi data (0–1, makin tinggi makin baik) |

Hasil evaluasi disimpan ke `models_beta/metrics_test_b.csv` (Beta) dan dicetak di console (Alpha).

---

## 10. Catatan Penting

**Urutan eksekusi wajib dipatuhi.** `alpha_b.ipynb` dan `beta_b.ipynb` membaca file dari `dataset_b/` yang hanya ada setelah `data_processing_stage.ipynb` selesai.

**Waktu training Beta sangat panjang.** Dengan 90 kombinasi training dan `EPOCHS = 100000` (dibatasi EarlyStopping), estimasi waktu bisa beberapa jam tergantung hardware. Disarankan menggunakan GPU.

**Aktifkan GPU jika tersedia.** TensorFlow secara otomatis menggunakan GPU jika CUDA tersedia. Verifikasi dengan:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Data sensor TDS.** Pipeline memiliki penanganan khusus untuk nilai TDS = 0 (dianggap noise sensor) dan nilai negatif. Pastikan sensor TDS sudah terkalibrasi dengan benar sebelum menggunakan dataset baru.

**Scaler hanya di-fit pada data training.** Ini adalah praktik standar untuk mencegah data leakage — `scaler_X` dan `scaler_y` hanya belajar dari split training, lalu diterapkan ke validasi dan test.

**File model tersimpan dalam format `.keras`** (format baru TensorFlow 2.x). Untuk memuat model:
```python
from tensorflow.keras.models import load_model
model = load_model('models_beta/beta_b_suhu_final.keras')
```

---

*Dokumentasi ini dibuat berdasarkan analisis kode dari `data_processing_stage.ipynb`, `alpha_b.ipynb`, dan `beta_b.ipynb`.*
