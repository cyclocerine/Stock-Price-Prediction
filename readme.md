![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)
# 📈 StockPricePrediction

Prediksi harga saham, komoditas, dan instrumen keuangan lainnya menggunakan deep learning (CNN-LSTM, BiLSTM, Transformer, hingga ensemble). Tersedia versi CLI dan GUI interaktif untuk visualisasi dan backtesting.

## 🔥 Fitur Unggulan

- Model deep learning yang fleksibel: CNN-LSTM, BiLSTM, Transformer, Ensemble
- Indikator teknikal: RSI, MACD, ADX, EMA/SMA, Fibonacci, Ichimoku, dan banyak lagi
- Hyperparameter tuning dengan Keras Tuner
- Backtesting strategi trading
- GUI interaktif berbasis PyQt5

---

## 🚀 Instalasi

```bash
git clone https://github.com/cyclocerine/Stock-Price-Prediction.git
cd Stock-Price-Prediction
pip install -r requirements.txt
```

> ⚠️ *Pastikan Python 3.8+ sudah terinstal dan `tensorflow`, `yfinance`, `PyQt5`, `scikit-learn`, `matplotlib`, `keras-tuner` tersedia.*

---

## 🧠 Menjalankan CLI

Untuk prediksi harga menggunakan mode terminal (CLI):

```bash
python main.py --help
```

```bash
python main.py --ticker AAPL --start_date 2020-01-01 --end_date 2024-12-31 --model cnn_lstm --lookback 60 --forecast 20 --tune False
```

### Parameter CLI

| Argumen        | Keterangan                            | Contoh             |
|----------------|----------------------------------------|--------------------|
| `--ticker`     | Simbol saham/komoditas                 | `AAPL`, `ADRO.JK`  |
| `--start_date` | Tanggal awal data historis (YYYY-MM-DD)| `2020-01-01`       |
| `--end_date`   | Tanggal akhir data historis            | `2024-12-31`       |
| `--model`      | Jenis model yang digunakan             | `cnn_lstm`, `bilstm`, `transformer`, `ensemble` |
| `--lookback`   | Jumlah hari historis untuk input model | `60`               |
| `--forecast`   | Jumlah hari ke depan untuk prediksi    | `20`               |
| `--tune`       | Aktifkan hyperparameter tuning         | `True` / `False`   |

---

## 🖼️ Menjalankan GUI

Untuk tampilan antarmuka grafis yang interaktif:

```bash
python main-gui.py
```

### Fitur GUI
- Input mudah untuk ticker, tanggal, dan model
- Visualisasi hasil prediksi & forecast
- Backtesting strategi trading (Trend Following, Mean Reversion, Predictive)
- Optimizer strategi otomatis
- Simpan grafik & hasil evaluasi ke file

---

## 🧪 Lisensi

Proyek ini menggunakan [Apache License 2.0](./LICENSE). Bebas digunakan, tapi kasih kredit ya!

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