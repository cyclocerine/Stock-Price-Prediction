# Copyright 2025 Fa'iq Hammam on behalf of cyclocerine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Input, Bidirectional, Concatenate, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, Add, GRU, Flatten, RepeatVector, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import argparse
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import time
import keras_tuner as kt

class TechnicalIndicators:
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """Menghitung Average Directional Index (ADX)"""
        try:
            # Konversi ke numpy array 1D
            high = np.array(high).flatten()
            low = np.array(low).flatten()
            close = np.array(close).flatten()
            
            # True Range
            high_low = high - low
            high_close_prev = np.abs(high - np.roll(close, 1))
            low_close_prev = np.abs(low - np.roll(close, 1))
            tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            
            # Directional Movement
            up_move = high - np.roll(high, 1)
            down_move = np.roll(low, 1) - low
            
            # +DM dan -DM
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothing
            tr_smooth = np.convolve(tr, np.ones(period)/period, mode='valid')
            pos_dm_smooth = np.convolve(pos_dm, np.ones(period)/period, mode='valid')
            neg_dm_smooth = np.convolve(neg_dm, np.ones(period)/period, mode='valid')
            
            # Directional Indicators
            pos_di = 100 * (pos_dm_smooth / tr_smooth)
            neg_di = 100 * (neg_dm_smooth / tr_smooth)
            
            # ADX
            dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
            adx = np.convolve(dx, np.ones(period)/period, mode='valid')
            
            # Padding
            pad_size = len(high) - len(adx)
            adx = np.pad(adx, (pad_size, 0), mode='edge')
            
            return adx
        except Exception as e:
            print(f"Error in calculate_adx: {str(e)}")
            return np.zeros_like(high)

    @staticmethod
    def calculate_rsi(close, period=14):
        """Menghitung Relative Strength Index (RSI)"""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return np.pad(rsi, (period, 0), mode='edge')

    @staticmethod
    def calculate_macd(close, fast=12, slow=26, signal=9):
        """Menghitung MACD"""
        exp1 = pd.Series(close).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = pd.Series(macd).ewm(span=signal, adjust=False).mean()
        return macd, signal_line

class DataPreprocessor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.features = None
        
    def download_data(self):
        """Mengunduh data saham"""
        try:
            print(f"Downloading data for {self.ticker}...")
            # Pastikan format tanggal benar
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            
            # Download data dengan format yang benar
            self.data = yf.download(
                self.ticker,
                start=start,
                end=end,
                progress=False
            )
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
                
            print(f"Downloaded {len(self.data)} data points")
            
            # Pastikan data dalam format yang benar
            self.data = self.data.astype({
                'Open': 'float64',
                'High': 'float64',
                'Low': 'float64',
                'Close': 'float64',
                'Volume': 'float64'
            })
            
            return True
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            return False
            
    def calculate_indicators(self):
        """Menghitung indikator teknikal"""
        try:
            if self.data is None:
                raise ValueError("No data available. Please download data first.")
                
            # Buat DataFrame baru untuk menyimpan indikator
            indicators = pd.DataFrame(index=self.data.index)
            
            # Moving Averages
            indicators['SMA_14'] = self.data['Close'].rolling(window=14, min_periods=1).mean()
            indicators['SMA_50'] = self.data['Close'].rolling(window=50, min_periods=1).mean()
            indicators['SMA_200'] = self.data['Close'].rolling(window=200, min_periods=1).mean()
            
            # Exponential Moving Averages
            indicators['EMA_9'] = self.data['Close'].ewm(span=9, adjust=False, min_periods=1).mean()
            indicators['EMA_21'] = self.data['Close'].ewm(span=21, adjust=False, min_periods=1).mean()
            indicators['EMA_55'] = self.data['Close'].ewm(span=55, adjust=False, min_periods=1).mean()
            
            # RSI
            delta = self.data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            
            rs = avg_gain / avg_loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = self.data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            exp2 = self.data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            indicators['MACD'] = exp1 - exp2
            indicators['Signal_Line'] = indicators['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            indicators['MACD_Histogram'] = indicators['MACD'] - indicators['Signal_Line']
            
            # ADX
            high = self.data['High'].values
            low = self.data['Low'].values
            close = self.data['Close'].values
            indicators['ADX'] = TechnicalIndicators.calculate_adx(high, low, close)
            
            # Volume indicators
            volume_ma = self.data['Volume'].rolling(window=20, min_periods=1).mean()
            indicators['Volume_Ratio'] = self.data['Volume'] / volume_ma.replace(0, np.nan)
            
            # Price momentum
            indicators['Momentum'] = self.data['Close'].pct_change(periods=10)
            indicators['ROC'] = ((self.data['Close'] / self.data['Close'].shift(10)) - 1) * 100
            
            # Gabungkan dengan data asli
            self.data = pd.concat([self.data, indicators], axis=1)
            
            # Hapus NaN values
            self.data.dropna(inplace=True)
            
            # Pilih fitur
            feature_columns = [
                'Close', 'Volume', 'SMA_14', 'SMA_50', 'SMA_200',
                'EMA_9', 'EMA_21', 'EMA_55', 'RSI', 'MACD',
                'Signal_Line', 'MACD_Histogram', 'ADX',
                'Volume_Ratio', 'Momentum', 'ROC'
            ]
            
            # Pastikan semua kolom ada
            available_columns = [col for col in feature_columns if col in self.data.columns]
            self.features = self.data[available_columns]
            
            return True
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return False

    def calculate_obv(self, close, volume):
        """Menghitung On Balance Volume (OBV) secara manual"""
        obv = pd.Series(0.0, index=close.index)
        
        # OBV hari pertama sama dengan volume hari pertama
        obv.iloc[0] = volume.iloc[0]
        
        # Untuk hari-hari berikutnya
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:  # Harga naik
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:  # Harga turun
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:  # Harga tetap
                obv.iloc[i] = obv.iloc[i-1]
            
        return obv

    def calculate_ichimoku(self, high, low, close):
        """Menghitung indikator Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line): (periode 9)
        tenkan_sen_high = high.rolling(window=9).max()
        tenkan_sen_low = low.rolling(window=9).min()
        tenkan_sen = (tenkan_sen_high + tenkan_sen_low) / 2
        
        # Kijun-sen (Base Line): (periode 26)
        kijun_sen_high = high.rolling(window=26).max()
        kijun_sen_low = low.rolling(window=26).min()
        kijun_sen = (kijun_sen_high + kijun_sen_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (periode 52)
        senkou_span_b_high = high.rolling(window=52).max()
        senkou_span_b_low = low.rolling(window=52).min()
        senkou_span_b = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): harga penutupan diplot 26 periode ke belakang
        chikou_span = close.shift(-26)
        
        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })

    def calculate_fibonacci_levels(self, data):
        """Menghitung level-level Fibonacci Retracement untuk periode tertentu"""
        # Ambil high dan low dalam range yang dipilih
        price_max = data['High'].max()
        price_min = data['Low'].min()
        
        # Level-level Fibonacci
        diff = price_max - price_min
        level_0 = price_min  # 0%
        level_23_6 = price_min + 0.236 * diff  # 23.6%
        level_38_2 = price_min + 0.382 * diff  # 38.2% 
        level_50_0 = price_min + 0.5 * diff    # 50%
        level_61_8 = price_min + 0.618 * diff  # 61.8%
        level_78_6 = price_min + 0.786 * diff  # 78.6%
        level_100 = price_max  # 100%
        
        # Buat kolom baru untuk setiap level
        fib_levels = pd.DataFrame(index=data.index)
        fib_levels['fib_0'] = level_0
        fib_levels['fib_23_6'] = level_23_6
        fib_levels['fib_38_2'] = level_38_2
        fib_levels['fib_50_0'] = level_50_0
        fib_levels['fib_61_8'] = level_61_8
        fib_levels['fib_78_6'] = level_78_6
        fib_levels['fib_100'] = level_100
        
        # Hitung jarak relatif harga penutupan terhadap level-level Fibonacci
        fib_levels['close_to_fib_ratio'] = (data['Close'] - level_0) / (level_100 - level_0)
        
        return fib_levels

class ModelBuilder:
    @staticmethod
    def build_cnn_lstm(input_shape):
        """Membangun model CNN-LSTM"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            BatchNormalization(),
            Dropout(0.2),
            
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model

    @staticmethod
    def build_bilstm(input_shape):
        """Membangun model Bidirectional LSTM"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            BatchNormalization(),
            Dropout(0.2),
            
            Bidirectional(LSTM(100, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=False)),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model

    @staticmethod
    def build_transformer(input_shape):
        """Membangun model Transformer"""
        inputs = Input(shape=input_shape)
        
        # Preprocessing
        x = Conv1D(64, 3, activation='relu')(inputs)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        
        # Transformer block
        attention_output = MultiHeadAttention(
            key_dim=256, num_heads=4, dropout=0.1
        )(x, x)
        x = Add()([attention_output, x])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Output processing
        x = GlobalAveragePooling1D()(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(25, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model

    @staticmethod
    def build_ensemble(input_shape):
        """Membangun model Ensemble"""
        inputs = Input(shape=input_shape)
        
        # CNN branch
        cnn = Conv1D(64, 3, activation='relu')(inputs)
        cnn = MaxPooling1D(2)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(0.2)(cnn)
        cnn = Conv1D(128, 3, activation='relu')(cnn)
        cnn = GlobalAveragePooling1D()(cnn)
        
        # LSTM branch
        lstm = Bidirectional(LSTM(100, return_sequences=True))(inputs)
        lstm = Dropout(0.2)(lstm)
        lstm = Bidirectional(LSTM(50, return_sequences=False))(lstm)
        
        # Transformer branch
        trans = Conv1D(64, 3, activation='relu')(inputs)
        trans = MaxPooling1D(2)(trans)
        attention_output = MultiHeadAttention(
            key_dim=256, num_heads=4, dropout=0.1
        )(trans, trans)
        trans = Add()([attention_output, trans])
        trans = LayerNormalization(epsilon=1e-6)(trans)
        trans = GlobalAveragePooling1D()(trans)
        
        # Combine branches
        combined = Concatenate()([cnn, lstm, trans])
        combined = Dense(50, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.2)(combined)
        combined = Dense(25, activation='relu')(combined)
        outputs = Dense(1)(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model

class HyperparameterTuner:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build_cnn_lstm_model(self, hp):
        """Membangun model CNN-LSTM dengan hyperparameter yang dapat di-tuning"""
        model = Sequential()
        
        # Tuning jumlah filter dan kernel size untuk layer CNN pertama
        model.add(Conv1D(
            filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Int('conv1_kernel', min_value=2, max_value=5),
            activation='relu',
            input_shape=self.input_shape
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Tuning jumlah filter dan kernel size untuk layer CNN kedua
        model.add(Conv1D(
            filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=32),
            kernel_size=hp.Int('conv2_kernel', min_value=2, max_value=5),
            activation='relu'
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Tuning jumlah unit LSTM
        model.add(LSTM(
            units=hp.Int('lstm1_units', min_value=50, max_value=200, step=50),
            return_sequences=True
        ))
        model.add(Dropout(hp.Float('dropout3', min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(LSTM(
            units=hp.Int('lstm2_units', min_value=25, max_value=100, step=25),
            return_sequences=False
        ))
        model.add(Dropout(hp.Float('dropout4', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Tuning learning rate
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.add(Dense(25, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
        return model
        
    def build_bilstm_model(self, hp):
        """Membangun model Bidirectional LSTM dengan hyperparameter yang dapat di-tuning"""
        model = Sequential()
        
        # Tuning jumlah filter dan kernel size untuk layer CNN
        model.add(Conv1D(
            filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Int('conv_kernel', min_value=2, max_value=5),
            activation='relu',
            input_shape=self.input_shape
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Tuning jumlah unit BiLSTM
        model.add(Bidirectional(LSTM(
            units=hp.Int('bilstm1_units', min_value=50, max_value=200, step=50),
            return_sequences=True
        )))
        model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(Bidirectional(LSTM(
            units=hp.Int('bilstm2_units', min_value=25, max_value=100, step=25),
            return_sequences=False
        )))
        model.add(Dropout(hp.Float('dropout3', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Tuning learning rate
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.add(Dense(25, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
        return model
        
    def build_transformer_model(self, hp):
        """Membangun model Transformer dengan hyperparameter yang dapat di-tuning"""
        inputs = Input(shape=self.input_shape)
        
        # Tuning parameter preprocessing
        x = Conv1D(
            filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Int('conv1_kernel', min_value=2, max_value=5),
            activation='relu'
        )(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(
            filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=32),
            kernel_size=hp.Int('conv2_kernel', min_value=2, max_value=5),
            activation='relu'
        )(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Tuning parameter transformer
        head_size = hp.Int('head_size', min_value=128, max_value=512, step=128)
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        ff_dim = hp.Int('ff_dim', min_value=2, max_value=8, step=2)
        
        attention_output = MultiHeadAttention(
            key_dim=head_size,
            num_heads=num_heads,
            dropout=hp.Float('attention_dropout', min_value=0.1, max_value=0.5, step=0.1)
        )(x, x)
        x = Add()([attention_output, x])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Tuning learning rate
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(hp.Int('dense1_units', min_value=32, max_value=128, step=32), activation='relu')(x)
        x = Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
        x = Dense(hp.Int('dense2_units', min_value=16, max_value=64, step=16), activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
        return model
        
    def tune_model(self, model_type, X_train, y_train, X_val, y_val, max_trials=10, executions_per_trial=1):
        """Melakukan hyperparameter tuning untuk model yang dipilih"""
        if model_type == 'cnn_lstm':
            tuner = kt.Hyperband(
                self.build_cnn_lstm_model,
                objective='val_loss',
                max_epochs=50,
                factor=3,
                directory='tuning',
                project_name='cnn_lstm_tuning'
            )
        elif model_type == 'bilstm':
            tuner = kt.Hyperband(
                self.build_bilstm_model,
                objective='val_loss',
                max_epochs=50,
                factor=3,
                directory='tuning',
                project_name='bilstm_tuning'
            )
        elif model_type == 'transformer':
            tuner = kt.Hyperband(
                self.build_transformer_model,
                objective='val_loss',
                max_epochs=50,
                factor=3,
                directory='tuning',
                project_name='transformer_tuning'
            )
        else:
            raise ValueError(f"Model type {model_type} not supported for tuning")
            
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # Mulai pencarian hyperparameter
        print(f"\nStarting hyperparameter tuning for {model_type} model...")
        tuner.search(
            X_train, y_train,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Dapatkan model terbaik
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print("\nBest hyperparameters found:")
        for param, value in best_hyperparameters.values.items():
            print(f"{param}: {value}")
            
        return best_model

class StockPredictor:
    def __init__(self, ticker, start_date, end_date, lookback=60, forecast_days=30, model_type='ensemble', tune_hyperparameters=False):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.forecast_days = forecast_days
        self.model_type = model_type
        self.tune_hyperparameters = tune_hyperparameters
        self.preprocessor = DataPreprocessor(ticker, start_date, end_date)
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_data(self):
        """Mempersiapkan data untuk training"""
        if not self.preprocessor.download_data():
            return False
        if not self.preprocessor.calculate_indicators():
            return False
            
        # Normalisasi data
        scaled_data = self.scaler.fit_transform(self.preprocessor.features)
        
        # Buat sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, 0])
            
        self.X = np.array(X)
        self.y = np.array(y)
        return True
        
    def train_model(self):
        """Melatih model"""
        input_shape = (self.X.shape[1], self.X.shape[2])
        
        # Split data untuk validation
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, shuffle=False
        )
        
        if self.tune_hyperparameters and self.model_type != 'ensemble':
            # Lakukan hyperparameter tuning
            tuner = HyperparameterTuner(input_shape)
            self.model = tuner.tune_model(
                self.model_type,
                X_train, y_train,
                X_val, y_val
            )
        else:
            # Gunakan model standar
            if self.model_type == 'cnn_lstm':
                self.model = ModelBuilder.build_cnn_lstm(input_shape)
            elif self.model_type == 'bilstm':
                self.model = ModelBuilder.build_bilstm(input_shape)
            elif self.model_type == 'transformer':
                self.model = ModelBuilder.build_transformer(input_shape)
            else:
                self.model = ModelBuilder.build_ensemble(input_shape)
                
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            ]
            
            # Training
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            return history
        
    def predict(self):
        """Melakukan prediksi"""
        # Prediksi pada data test
        y_pred = self.model.predict(self.X)
        
        # Transformasi kembali ke skala asli
        y_pred_original = self.scaler.inverse_transform(
            np.concatenate([y_pred, np.zeros((len(y_pred), self.X.shape[2]-1))], axis=1)
        )[:, 0]
        
        y_original = self.scaler.inverse_transform(
            np.concatenate([self.y.reshape(-1, 1), np.zeros((len(self.y), self.X.shape[2]-1))], axis=1)
        )[:, 0]
        
        # Forecasting
        last_sequence = self.X[-1]
        forecast = []
        
        for _ in range(self.forecast_days):
            pred = self.model.predict(last_sequence.reshape(1, *last_sequence.shape))
            forecast.append(pred[0, 0])
            
            # Update sequence
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = pred[0]
            
        forecast = np.array(forecast)
        forecast_original = self.scaler.inverse_transform(
            np.concatenate([forecast.reshape(-1, 1), np.zeros((len(forecast), self.X.shape[2]-1))], axis=1)
        )[:, 0]
        
        return y_original, y_pred_original, forecast_original
        
    def evaluate(self, y_true, y_pred):
        """Evaluasi model"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\nModel Evaluation:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
        
    def plot_results(self, y_true, y_pred, forecast):
        """Plot hasil prediksi"""
        plt.figure(figsize=(15, 7))
        
        # Plot data historis
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
        
        # Plot forecast
        forecast_dates = pd.date_range(
            start=self.preprocessor.data.index[-1] + timedelta(days=1),
            periods=self.forecast_days
        )
        plt.plot(
            range(len(y_true), len(y_true) + len(forecast)),
            forecast,
            label='Forecast',
            color='green',
            linestyle='-.'
        )
        
        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.ticker}_prediction.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--ticker', type=str, default='ADRO.JK', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--lookback', type=int, default=60, help='Number of days to look back')
    parser.add_argument('--forecast_days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--model', type=str, default='ensemble', choices=['cnn_lstm', 'bilstm', 'transformer', 'ensemble'], help='Model type')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    predictor = StockPredictor(
        args.ticker,
        args.start_date,
        args.end_date,
        args.lookback,
        args.forecast_days,
        args.model,
        args.tune
    )
    
    if not predictor.prepare_data():
        print("Error preparing data")
        return
        
    print(f"\nTraining {args.model} model...")
    history = predictor.train_model()
    
    print("\nMaking predictions...")
    y_true, y_pred, forecast = predictor.predict()
    
    print("\nEvaluating model...")
    metrics = predictor.evaluate(y_true, y_pred)
    
    print("\nPlotting results...")
    predictor.plot_results(y_true, y_pred, forecast)
    
    print("\nForecast for next days:")
    for i, price in enumerate(forecast, 1):
        print(f"Day {i}: {price:.2f}")

if __name__ == "__main__":
    main()