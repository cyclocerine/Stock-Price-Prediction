![License](https://img.shields.io/badge/license-MIT-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
# 📈 StockPricePrediction

Aplikasi prediksi harga saham yang menggunakan machine learning untuk memprediksi harga saham dan mendukung backtesting strategi trading.

## Fitur
- Prediksi harga saham menggunakan berbagai model deep learning (CNN-LSTM, BiLSTM, Transformer, Ensemble)
- Backtesting strategi trading
- Optimisasi parameter strategi
- Simulasi trading berdasarkan hasil prediksi
- Reinforcement Learning dengan PPO Agent untuk optimasi strategi trading
- Antarmuka grafis (GUI) untuk kemudahan penggunaan
- Perhitungan dan visualisasi berbagai indikator teknikal

## Struktur Proyek
Berikut adalah struktur proyek 

```
Stock-Price-Prediction/
├── README.md                  # Dokumentasi proyek
├── requirements.txt           # Dependencies proyek
├── LICENSE                    # Lisensi proyek
├── CODE_OF_CONDUCT.md         # Kode etik
├── CONTRIBUTING.md            # Panduan kontribusi
├── src/                       # Direktori utama kode sumber
│   ├── __init__.py            # File inisialisasi package
│   ├── data/                  # Modul terkait data
│   │   ├── __init__.py
│   │   ├── preprocessor.py    # Preprocessing data
│   │   └── indicators.py      # Indikator teknikal
│   ├── models/                # Modul terkait model
│   │   ├── __init__.py
│   │   ├── builder.py         # Pembuat model (CNN-LSTM, BiLSTM, Transformer, Ensemble)
│   │   ├── tuner.py           # Hyperparameter tuning
│   │   └── predictor.py       # Prediktor saham
│   ├── trading/               # Modul terkait trading
│   │   ├── __init__.py
│   │   ├── strategies.py      # Strategi trading
│   │   ├── backtest.py        # Backtesting
│   │   ├── optimizer.py       # Optimisasi strategi
│   │   └── ppo_agent.py       # Reinforcement Learning dengan PPO 
│   ├── gui/                   # Modul terkait GUI
│   │   ├── __init__.py
│   │   ├── app.py             # Aplikasi utama
│   │   ├── prediction_tab.py  # Tab prediksi
│   │   ├── backtest_tab.py    # Tab backtesting
│   │   ├── optimizer_tab.py   # Tab optimisasi
│   │   ├── forecast_tab.py    # Tab trading forecast
│   │   └── styles.py          # Konfigurasi style GUI
│   └── utils/                 # Utilitas
│       ├── __init__.py
│       ├── visualization.py   # Visualisasi data dan hasil
│       └── common.py          # Fungsi umum
├── tests/                     # Direktori untuk pengujian
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_trading.py
│   └── test_utils.py
├── examples/                  # Contoh penggunaan
│   ├── basic_prediction.py
│   ├── backtest_example.py
│   └── strategy_optimization.py
├── scripts/                   # Script untuk menjalankan aplikasi
│   ├── run_app.py             # Menjalankan aplikasi GUI
│   └── run_cli.py             # Menjalankan versi command line
└── assets/                    # Aset statis (gambar, logo, dll)
    └── screenshot.png
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Penggunaan

### GUI
```bash
python scripts/run_app.py
```

### Command Line
```bash
python scripts/run_cli.py --ticker ADRO.JK --start_date 2020-01-01 --end_date 2023-01-01 --model ensemble --lookback 60 --forecast_days 30
```

### Parameter CLI

---

| Argumen        | Keterangan                            | Contoh             |
|----------------|----------------------------------------|--------------------|
| `--ticker`     | Simbol saham/komoditas                 | `AAPL`, `ADRO.JK`  |
| `--start_date` | Tanggal awal data historis (YYYY-MM-DD)| `2020-01-01`       |
| `--end_date`   | Tanggal akhir data historis            | `2024-12-31`       |
| `--model`      | Jenis model yang digunakan             | `cnn_lstm`, `bilstm`, `transformer`, `ensemble` |
| `--lookback`   | Jumlah hari historis untuk input model | `60`               |
| `--forecast`   | Jumlah hari ke depan untuk prediksi    | `20`               |
| `--tune`       | Aktifkan hyperparameter tuning         | `True` / `False`   |
| `--rl`         | Gunakan Reinforcement Learning         | `True` / `False`   |
| `--episodes`   | Jumlah episode untuk training RL       | `100`              |

---

## Model yang Didukung

### CNN-LSTM
Kombinasi Convolutional Neural Network dan Long Short-Term Memory yang efektif untuk data deret waktu. Lapisan CNN mengidentifikasi pola lokal, sedangkan LSTM menangkap ketergantungan jangka panjang.

### BiLSTM
Bidirectional LSTM yang menganalisis data dari kedua arah (maju dan mundur) untuk pemahaman konteks yang lebih baik.

### Transformer
Arsitektur berdasarkan mekanisme attention yang sangat efektif untuk sequence modeling. Menangkap ketergantungan jarak jauh tanpa menggunakan recurrence.

### Ensemble
Kombinasi dari tiga model di atas, menggabungkan kelebihan masing-masing untuk kinerja prediksi yang lebih baik.

## Reinforcement Learning untuk Trading

Proyek ini mengimplementasikan Proximal Policy Optimization (PPO) untuk melatih agen yang mengoptimalkan keputusan trading. PPO Agent belajar secara otomatis kapan harus membeli, menjual, atau menahan saham untuk memaksimalkan keuntungan.

## Contributing
Silakan baca [CONTRIBUTING.md](CONTRIBUTING.md) untuk detail tentang kode etik dan proses pengajuan pull request.
---

## 🧪 Lisensi

Proyek ini menggunakan [MIT License](./LICENSE). Bebas digunakan, tapi kasih kredit ya!

---

## 🤝 Kontribusi
Proyek ini adalah langkah yang menarik dalam dunia prediksi harga saham dengan deep learning. Anda dapat menggunakannya sebagai dasar untuk riset dan pengembangan lebih lanjut.

Apakah Anda menemukan bug? Punya ide untuk peningkatan? Atau hanya ingin berbagi hasil prediksi Anda? Jangan ragu untuk membuka issue di repositori ini atau menghubungi kami lewat email.

Mari kita jadikan alat prediksi saham ini lebih baik bersama-sama! 🚀
Pull Request? Boleh banget. Buka issue? Gaskeun. 🌟

---

## ✨ Dibuat oleh

**Fa'iq Hammam**  
💼 [LinkedIn](https://www.linkedin.com/in/faiq-hammam-mutaqin-9a3733217/) | 💻 IT Engineer | 🧠 Deep Learning Enthusiast
