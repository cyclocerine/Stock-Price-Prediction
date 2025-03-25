import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Input
from keras.optimizers import Adam

# Fungsi untuk mengunduh data dan menghitung indikator teknikal
def download_and_preprocess_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    # Hitung Simple Moving Average (SMA)
    data['SMA'] = data['Close'].rolling(window=14).mean()
    # Hitung RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    data['20_MA'] = data['Close'].rolling(window=20).mean()
    data['Close_rolling_std'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['20_MA'] + 2 * data['Close_rolling_std']
    data['Lower_Band'] = data['20_MA'] - 2 * data['Close_rolling_std']
    # MACD dan Signal Line
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    # Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['Stochastic'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)

    # Hapus baris yang memiliki nilai NaN
    data.dropna(inplace=True)

    # Pilih fitur yang relevan (target adalah 'Close')
    features = ['Close', 'SMA', 'RSI', '20_MA', 'Close_rolling_std',
                'Upper_Band', 'Lower_Band', 'MACD', 'Signal_Line', 'Stochastic']
    data = data[features]

    return data

# Fungsi untuk membuat dataset dengan window lookback
def create_dataset(data, lookback, scaler):
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])  # target adalah harga penutupan (kolom pertama)
    return np.array(X), np.array(y), scaled_data

# Fungsi untuk membangun model CNN + LSTM
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Layer Conv1D untuk ekstraksi fitur
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Tiga lapis LSTM untuk analisis sekuensial
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # BatchNormalization untuk stabilitas model
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Fungsi untuk melakukan prediksi multi-hari secara iteratif (forecasting)
def forecast_future(model, last_sequence, forecast_days, num_features):
    forecasted_scaled = []
    current_sequence = last_sequence.copy()  # bentuk: (lookback, num_features)
    for _ in range(forecast_days):
        # Prediksi satu langkah ke depan
        pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], num_features), verbose=0)
        forecasted_scaled.append(pred[0, 0])
        # Buat baris baru: nilai prediksi untuk 'Close' dan sisanya diisi nol
        new_row = np.zeros(num_features)
        new_row[0] = pred[0, 0]
        # Update current_sequence dengan menghapus baris pertama dan menambahkan new_row di akhir
        current_sequence = np.vstack((current_sequence[1:], new_row))
    return np.array(forecasted_scaled)

# ==== Main Program ====
if __name__ == "__main__":
    # Pengaturan parameter
    ticker = '^JKSE'
    start_date = '2015-01-01'
    end_date = '2025-03-25'
    lookback = 60       # Jumlah hari historis yang digunakan sebagai input
    forecast_days = 20  # Jumlah hari ke depan yang akan di-forecast

    # Unduh dan proses data
    data = download_and_preprocess_data(ticker, start_date, end_date)
    print("Data awal setelah penambahan indikator:")
    print(data.head())

    # Inisialisasi scaler dan buat dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    X, y, scaled_data = create_dataset(data, lookback, scaler)
    print("Bentuk data setelah scaling:", scaled_data.shape)

    # Bagi data menjadi pelatihan dan pengujian (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Bentuk input untuk model (samples, timesteps, features)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Bangun dan latih model
    model = build_model(input_shape)
    model.fit(X_train, y_train, epochs=35, batch_size=32)

    # Prediksi pada data pengujian (single-step prediction)
    predicted_scaled = model.predict(X_test)
    predicted_scaled = predicted_scaled.flatten()

    # Untuk mengembalikan skala harga asli, buat dummy array dengan nilai nol untuk fitur selain 'Close'
    num_features = X_train.shape[2]
    other_feature_count = num_features - 1
    # Proses inverse transform untuk data pengujian
    y_test_full = np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], other_feature_count))))
    real_prices = scaler.inverse_transform(y_test_full)[:, 0]

    pred_full = np.hstack((predicted_scaled.reshape(-1, 1), np.zeros((predicted_scaled.shape[0], other_feature_count))))
    predicted_prices = scaler.inverse_transform(pred_full)[:, 0]

    # Forecasting: prediksi ke depan menggunakan sequence terakhir
    last_sequence = scaled_data[-lookback:]
    forecasted_scaled = forecast_future(model, last_sequence, forecast_days, num_features)
    forecasted_full = np.hstack((forecasted_scaled.reshape(-1, 1), np.zeros((forecast_days, other_feature_count))))
    forecasted_prices = scaler.inverse_transform(forecasted_full)[:, 0]

    # Gabungkan grafik: data pengujian dan forecast ke depan dalam satu plot
    plt.figure(figsize=(12,6))
    # Grafik harga aktual pada data pengujian
    plt.plot(real_prices, color='red', label='Harga Aktual (Test)')
    # Grafik harga prediksi pada data pengujian
    plt.plot(predicted_prices, color='blue', label='Harga Prediksi (Test)')
    # Grafik forecast ke depan (extend x-axis)
    x_forecast = np.arange(len(real_prices), len(real_prices) + forecast_days)
    plt.plot(x_forecast, forecasted_prices, marker='o', linestyle='-', color='green', label='Forecast Harga')

    plt.title('Perbandingan Harga Aktual, Prediksi, dan Harga Perkiraan')
    plt.xlabel('Waktu')
    plt.ylabel('Harga')
    plt.legend()
    plt.show()
