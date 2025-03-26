# 📈 Prediksi Harga Saham dengan CNN-LSTM

Selamat datang di proyek prediksi harga saham! 👋 

Apakah Anda pernah bertanya-tanya bagaimana cara memprediksi pergerakan harga saham? Repositori ini hadir untuk membantu Anda menjawab pertanyaan tersebut dengan menggunakan kekuatan deep learning! Di sini, kita menggabungkan dua model canggih - Convolutional Neural Networks (CNN) dan Long Short-Term Memory (LSTM) - untuk menganalisis dan memperkirakan harga saham berdasarkan data historis dan berbagai indikator teknikal.

## 🛠️ Apa Yang Anda Butuhkan

Tidak perlu khawatir tentang setup yang rumit! Cukup pastikan Anda memiliki Python terinstal di komputer Anda, lalu ikuti langkah berikut untuk menginstal semua library yang diperlukan:

```bash
pip install -r requirements.txt
```

Dan voila! Semua yang Anda butuhkan akan terinstal secara otomatis. Mudah, bukan?

## 🚀 Mulai Menggunakan

### Langkah-langkah Sederhana

1. **Unduh kode ke komputer Anda:**

```bash
git clone https://github.com/cyclocerine/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

2. **Sesuaikan dengan kebutuhan Anda:**

   - **Pilih saham favorit Anda** - Ingin melihat bagaimana IHSG akan bergerak? Atau mungkin Anda tertarik dengan Apple? Cukup ubah kode saham:

   ```python
   ticker = '^JKSE'  # Untuk IHSG Indonesia
   # atau
   ticker = 'AAPL'   # Untuk Apple
   # atau saham favorit Anda lainnya!
   ```

   - **Tentukan rentang waktu** - Pilih periode historis yang ingin Anda analisis:

   ```python
   start_date = '2015-01-01'  # Mulai dari awal 2015
   end_date = '2025-03-25'    # Hingga Maret 2025
   ```

   - **Atur parameter prediksi** - Ingin model melihat 60 hari ke belakang untuk memprediksi 20 hari ke depan? Tidak masalah!

   ```python
   lookback = 60       # Model akan belajar dari 60 hari terakhir
   forecast_days = 20  # Dan memprediksi 20 hari ke depan
   ```

3. **Jalankan dan lihat hasilnya:**

```bash
python main.py
```

Kemudian duduklah sambil menikmati kopi ☕ sementara model bekerja keras! Program akan otomatis mengunduh data, melatih model, dan menampilkan hasil prediksi dalam bentuk grafik yang menarik.

## 🧠 Bagaimana Ini Bekerja?

### Arsitektur Model yang Canggih

Model CNN-LSTM kami menggabungkan dua pendekatan powerful dalam deep learning:

- **CNN** - Seperti ahli dalam mendeteksi pola visual, layer CNN mengekstrak pola-pola penting dari data time series
- **LSTM** - Seperti memiliki memori jangka panjang, layer LSTM menangkap hubungan dan tren yang terjadi selama periode waktu tertentu

Kombinasi keduanya menciptakan sistem yang dapat "melihat" pola harga dan "mengingat" tren jangka panjang - persis seperti yang dilakukan trader profesional!

### Fungsi-fungsi Utama yang Bekerja untuk Anda

- 📊 **download_and_preprocess_data** - Mengunduh data saham dan mengkalkulasi berbagai indikator teknikal yang biasa digunakan oleh para analis saham profesional

- 🔍 **create_dataset** - Menyiapkan data dengan cara khusus agar model dapat belajar dengan optimal

- 🏗️ **build_model** - Membangun arsitektur CNN-LSTM yang canggih untuk menangkap kompleksitas pasar saham

- 🔮 **forecast_future** - Menggunakan model terlatih untuk memprediksi harga di masa depan

## ⚙️ Sesuaikan Sesuka Anda

Anda memiliki kendali penuh! Ubah parameter ini sesuai keinginan:

- 🏢 **ticker** - Kode saham yang ingin Anda analisis (AAPL, BBCA.JK, ^JKSE, dll.)
- 📅 **start_date & end_date** - Periode data historis
- 🔙 **lookback** - Seberapa jauh ke belakang model harus "melihat"
- 🔜 **forecast_days** - Seberapa jauh ke depan model harus "meramal"

## 📊 Performa yang Meyakinkan

Model CNN-LSTM ini memiliki keunggulan dibanding model tradisional! Dengan menggabungkan kekuatan CNN untuk deteksi pola dan LSTM untuk memori jangka panjang, model ini mampu menangkap kompleksitas pasar saham yang dipengaruhi berbagai faktor.

Performa model dievaluasi dengan metrik standar industri:
- MSE (Mean Squared Error) - Mengukur rata-rata kesalahan kuadrat
- MAE (Mean Absolute Error) - Mengukur rata-rata kesalahan absolut

Dalam praktiknya, model ini mampu menangkap tren pasar dengan lebih baik dibanding model tradisional seperti ARIMA!

## 🤝 Mari Berkolaborasi

Proyek ini adalah langkah yang menarik dalam dunia prediksi harga saham dengan deep learning. Anda dapat menggunakannya sebagai dasar untuk riset dan pengembangan lebih lanjut.

Apakah Anda menemukan bug? Punya ide untuk peningkatan? Atau hanya ingin berbagi hasil prediksi Anda? Jangan ragu untuk membuka issue di repositori ini atau menghubungi kami lewat email.

Mari kita jadikan alat prediksi saham ini lebih baik bersama-sama! 🚀

## 📜 Lisensi

Proyek ini dilisensikan di bawah lisensi Apache 2.0 - lihat file [LICENSE](./LICENSE) untuk detail.