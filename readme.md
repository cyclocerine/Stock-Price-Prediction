# Prediksi Harga Saham dengan CNN-LSTM

Repositori ini berisi kode untuk memprediksi harga saham menggunakan gabungan model Convolutional Neural Network (CNN) dan Long Short-Term Memory (LSTM). Model ini memanfaatkan berbagai indikator teknikal seperti SMA, RSI, Bollinger Bands, MACD, dan Stochastic Oscillator untuk memperkirakan harga saham di masa depan.

## Persyaratan

Sebelum menjalankan kode, pastikan Anda menginstal semua library yang diperlukan. Anda dapat menginstalnya menggunakan `pip`:

```bash
pip install yfinance numpy pandas matplotlib scikit-learn keras
```

## Cara Penggunaan

### Mengunduh dan Menjalankan Kode

1. Kloning repositori ini ke komputer Anda:

```bash
git clone https://github.com/cyclocerine/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

2. Ubah parameter variabel pada bagian utama kode sesuai kebutuhan Anda:

   - **Kode Saham**: Ganti kode saham yang ingin Anda prediksi pada variabel `ticker`. Misalnya, jika Anda ingin memprediksi indeks saham Indonesia, gunakan `'^JKSE'`. Untuk saham lain, Anda bisa menggantinya dengan kode yang sesuai.

   ```python
   ticker = '^JKSE'  # Ganti dengan kode saham lain, misalnya 'AAPL' untuk Apple
   ```

   - **Tanggal Awal dan Tanggal Akhir**: Ganti tanggal mulai dan tanggal akhir untuk data yang akan diambil menggunakan format YYYY-MM-DD. Misalnya, untuk memulai dari 1 Januari 2015 hingga 25 Maret 2025, gunakan:

   ```python
   start_date = '2015-01-01'
   end_date = '2025-03-25'
   ```

   - **Lookback dan Forecast Days**:

     - **Lookback**: Menentukan jumlah hari historis yang digunakan untuk input model. Misalnya, jika Anda ingin model menggunakan data harga 60 hari terakhir sebagai input, ubah:

     ```python
     lookback = 60
     ```

     - **Forecast Days**: Menentukan jumlah hari ke depan yang ingin Anda prediksi. Misalnya, untuk memprediksi harga selama 20 hari ke depan, ubah:

     ```python
     forecast_days = 20
     ```

3. **Menjalankan Kode**: Setelah melakukan perubahan yang diinginkan, jalankan kode utama:

```bash
python main.py
```

Kode ini akan mengunduh data saham, menghitung indikator teknikal, melatih model, dan menghasilkan prediksi harga untuk periode tertentu.

4. **Melihat Hasil Prediksi**: Setelah kode selesai dijalankan, grafik hasil prediksi akan ditampilkan. Grafik ini menunjukkan perbandingan antara harga aktual, harga prediksi pada data uji, dan perkiraan harga untuk beberapa hari ke depan.

## Penjelasan Kode

### Fungsi-fungsi Utama

- **download_and_preprocess_data**: Mengunduh data saham menggunakan yfinance dan menghitung berbagai indikator teknikal (SMA, RSI, Bollinger Bands, MACD, Stochastic Oscillator).

- **create_dataset**: Membuat dataset dengan menggunakan data yang telah diproses dan mengubahnya menjadi format yang sesuai untuk pelatihan model (menggunakan MinMaxScaler untuk normalisasi).

- **build_model**: Membangun model CNN + LSTM dengan tiga lapis LSTM untuk memproses data sekuensial dan Conv1D untuk ekstraksi fitur.

- **forecast_future**: Melakukan prediksi harga saham ke depan berdasarkan sequence terakhir yang telah diproses.

### Variabel yang Dapat Diubah

- **ticker**: Ganti dengan simbol saham atau indeks yang ingin Anda prediksi (misalnya AAPL untuk Apple, GOOGL untuk Google, atau ^JKSE untuk Jakarta Stock Exchange).

- **start_date** dan **end_date**: Tentukan rentang waktu data historis yang ingin diunduh.

- **lookback**: Jumlah hari historis yang digunakan untuk melatih model.

- **forecast_days**: Jumlah hari ke depan yang akan diprediksi.

## Penutupan

Repositori ini memberikan contoh implementasi model deep learning untuk prediksi harga saham dengan memanfaatkan teknik CNN-LSTM. Anda dapat menyesuaikan kode untuk berbagai saham atau indeks dengan mengganti parameter sesuai kebutuhan.

Jika Anda menemukan masalah atau memiliki pertanyaan, silakan buka issue di repositori ini atau hubungi kontributor utama melalui email.