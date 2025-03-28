
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Input
from keras.optimizers import Adam

# Fungsi untuk mengunduh data dan menghitung indikator teknikal
def download_and_preprocess_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            messagebox.showerror("Error", "Data tidak ditemukan! Cek kode saham dan tanggal.")
            return None

        data['SMA'] = data['Close'].rolling(window=14).mean()
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['20_MA'] = data['Close'].rolling(window=20).mean()
        data['Close_rolling_std'] = data['Close'].rolling(window=20).std()
        data['Upper_Band'] = data['20_MA'] + 2 * data['Close_rolling_std']
        data['Lower_Band'] = data['20_MA'] - 2 * data['Close_rolling_std']
        data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['Stochastic'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)

        data.dropna(inplace=True)
        features = ['Close', 'SMA', 'RSI', '20_MA', 'Close_rolling_std', 'Upper_Band',
                    'Lower_Band', 'MACD', 'Signal_Line', 'Stochastic']
        return data[features]
    except Exception as e:
        messagebox.showerror("Error", f"Gagal mengunduh data: {e}")
        return None

# Fungsi untuk membuat dataset
def create_dataset(data, lookback, scaler):
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaled_data

# Fungsi untuk membangun model CNN + LSTM
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Fungsi untuk memprediksi harga saham
def predict():
    ticker = ticker_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    if not ticker or not start_date or not end_date:
        messagebox.showwarning("Peringatan", "Mohon isi semua kolom!")
        return

    global model, scaler, real_prices, predicted_prices

    data = download_and_preprocess_data(ticker, start_date, end_date)
    if data is None:
        return

    scaler = MinMaxScaler(feature_range=(0, 1))
    X, y, scaled_data = create_dataset(data, lookback=60, scaler=scaler)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    predicted_scaled = model.predict(X_test).flatten()

    num_features = X_train.shape[2]
    y_test_full = np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], num_features - 1))))
    real_prices = scaler.inverse_transform(y_test_full)[:, 0]

    pred_full = np.hstack((predicted_scaled.reshape(-1, 1), np.zeros((predicted_scaled.shape[0], num_features - 1))))
    predicted_prices = scaler.inverse_transform(pred_full)[:, 0]

    plot_results()

# Fungsi untuk menampilkan hasil prediksi dalam grafik
def plot_results():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(real_prices, color='red', label='Harga Aktual')
    ax.plot(predicted_prices, color='blue', label='Harga Prediksi')
    ax.set_title("Hasil Prediksi Harga Saham")
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Harga")
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, pady=10)
    canvas.draw()

# Buat GUI
window = tk.Tk()
window.title("Prediksi Saham dengan LSTM + CNN")

# Label dan Input
tk.Label(window, text="Kode Saham:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
ticker_entry = ttk.Entry(window)
ticker_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(window, text="Tanggal Mulai:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
start_date_entry = DateEntry(window, date_pattern='yyyy-mm-dd')
start_date_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(window, text="Tanggal Akhir:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
end_date_entry = DateEntry(window, date_pattern='yyyy-mm-dd')
end_date_entry.grid(row=2, column=1, padx=10, pady=5)

# Tombol Prediksi
predict_button = ttk.Button(window, text="Prediksi", command=predict)
predict_button.grid(row=3, column=0, columnspan=2, pady=10)

# Menjalankan GUI
window.mainloop()
