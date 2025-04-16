import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                           QComboBox, QDateEdit, QSpinBox, QCheckBox, 
                           QProgressBar, QMessageBox, QFileDialog, QTabWidget,
                           QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
                           QGridLayout, QGroupBox, QRadioButton)
from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from main import StockPredictor

class BacktestWorker(QThread):
    finished = pyqtSignal(object, object, object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, predictor, initial_investment, strategy):
        super().__init__()
        self.predictor = predictor
        self.initial_investment = initial_investment
        self.strategy = strategy
        
    def run(self):
        try:
            # Persiapan data
            if not self.predictor.prepare_data():
                self.error.emit("Error preparing data")
                return
            
            self.progress.emit(30)
            
            # Training model
            history = self.predictor.train_model()
            self.progress.emit(60)
            
            # Mendapatkan data untuk backtesting
            y_true, y_pred, _ = self.predictor.predict()
            
            # Jalankan backtesting
            portfolio_values, trades, performance = self.run_backtest(
                y_true, y_pred, self.initial_investment, self.strategy
            )
            
            self.progress.emit(100)
            self.finished.emit(portfolio_values, trades, performance)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def run_backtest(self, actual_prices, predicted_prices, initial_investment, strategy):
        # Inisialisasi portfolio
        cash = initial_investment
        shares = 0
        portfolio_values = []
        trades = []
        
        # Memastikan data memiliki panjang yang sama
        length = min(len(actual_prices), len(predicted_prices))
        actual_prices = actual_prices[:length]
        predicted_prices = predicted_prices[:length]
        
        # Iterasi melalui harga historis
        for i in range(1, length):
            # Hitung nilai portfolio saat ini
            portfolio_value = cash + shares * actual_prices[i]
            portfolio_values.append(portfolio_value)
            
            signal = self.generate_signal(predicted_prices, actual_prices, i, strategy)
            
            # Proses signals
            if signal == 'BUY' and cash > 0:
                # Beli saham sebanyak mungkin dengan uang yang ada
                shares_to_buy = cash / actual_prices[i]
                shares += shares_to_buy
                cash = 0
                trades.append({
                    'day': i,
                    'type': 'BUY',
                    'price': actual_prices[i],
                    'shares': shares_to_buy,
                    'value': shares_to_buy * actual_prices[i]
                })
            elif signal == 'SELL' and shares > 0:
                # Jual semua saham
                cash += shares * actual_prices[i]
                trades.append({
                    'day': i,
                    'type': 'SELL',
                    'price': actual_prices[i],
                    'shares': shares,
                    'value': shares * actual_prices[i]
                })
                shares = 0
        
        # Nilai akhir portfolio
        final_value = cash + shares * actual_prices[-1]
        
        # Menghitung metrik performa
        total_return = (final_value - initial_investment) / initial_investment * 100
        
        # Menghitung drawdown
        peak = portfolio_values[0]
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown = max(drawdown, dd)
        
        # Menghitung Sharpe Ratio (sangat disederhanakan, asumsi risk-free rate = 0)
        daily_returns = [portfolio_values[i]/portfolio_values[i-1]-1 for i in range(1, len(portfolio_values))]
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Menghitung profit/loss per trade
        win_trades = 0
        loss_trades = 0
        for i in range(0, len(trades), 2):
            if i+1 < len(trades):
                buy = trades[i]
                sell = trades[i+1]
                profit = sell['value'] - buy['value']
                if profit > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
        
        win_rate = 0
        if win_trades + loss_trades > 0:
            win_rate = win_trades / (win_trades + loss_trades) * 100
        
        performance = {
            'initial_investment': initial_investment,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return portfolio_values, trades, performance
    
    def generate_signal(self, predicted_prices, actual_prices, index, strategy):
        if strategy == 'Trend Following':
            # Beli jika prediksi menunjukkan tren naik
            if index > 1 and predicted_prices[index] > predicted_prices[index-1]:
                return 'BUY'
            # Jual jika prediksi menunjukkan tren turun
            elif index > 1 and predicted_prices[index] < predicted_prices[index-1]:
                return 'SELL'
        
        elif strategy == 'Mean Reversion':
            # Hitung rata-rata bergerak 5 hari
            if index >= 5:
                sma = sum(actual_prices[index-5:index]) / 5
                # Beli jika harga di bawah rata-rata (oversold)
                if actual_prices[index] < sma * 0.98:
                    return 'BUY'
                # Jual jika harga di atas rata-rata (overbought)
                elif actual_prices[index] > sma * 1.02:
                    return 'SELL'
        
        elif strategy == 'Predictive':
            # Beli jika prediksi menunjukkan harga akan naik
            if predicted_prices[index] > actual_prices[index] * 1.01:
                return 'BUY'
            # Jual jika prediksi menunjukkan harga akan turun
            elif predicted_prices[index] < actual_prices[index] * 0.99:
                return 'SELL'
        
        return 'HOLD'

class WorkerThread(QThread):
    finished = pyqtSignal(object, object, object, object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        
    def run(self):
        try:
            # Persiapan data
            if not self.predictor.prepare_data():
                self.error.emit("Error preparing data")
                return
                
            self.progress.emit(20)
            
            # Training model
            history = self.predictor.train_model()
            self.progress.emit(60)
            
            # Prediksi
            y_true, y_pred, forecast = self.predictor.predict()
            self.progress.emit(80)
            
            # Evaluasi
            metrics = self.predictor.evaluate(y_true, y_pred)
            self.progress.emit(90)
            
            # Plot hasil
            self.predictor.plot_results(y_true, y_pred, forecast)
            self.progress.emit(100)
            
            self.finished.emit(y_true, y_pred, forecast, metrics)
            
        except Exception as e:
            self.error.emit(str(e))

class StrategyOptimizerWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, ticker, start_date, end_date, models, initial_investment, strategies=None,
                param_ranges=None):
        super().__init__()
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.models = models
        self.initial_investment = initial_investment
        self.strategies = strategies or ['Trend Following', 'Mean Reversion', 'Predictive']
        
        # Default parameter ranges jika tidak ada yang diberikan
        self.param_ranges = param_ranges or {
            'Trend Following': {
                'threshold': [0.5, 1.0, 1.5, 2.0],  # Persentase perubahan untuk trigger
            },
            'Mean Reversion': {
                'sma_period': [3, 5, 7, 10, 15],  # Periode SMA
                'threshold': [1.0, 1.5, 2.0, 2.5, 3.0]  # Persentase deviasi dari SMA
            },
            'Predictive': {
                'threshold': [0.5, 1.0, 1.5, 2.0, 2.5]  # Persentase perbedaan prediksi-aktual
            }
        }
    
    def run(self):
        try:
            results = []
            total_combinations = sum(len(self.models) * len(self.strategies) * 
                                    len(self.param_ranges[strategy].get('threshold', [1])) * 
                                    len(self.param_ranges[strategy].get('sma_period', [1]))
                                    for strategy in self.strategies)
            
            completed = 0
            
            for model in self.models:
                # Buat predictor untuk model ini
                predictor = StockPredictor(
                    self.ticker, self.start_date, self.end_date,
                    60, 20, model, False
                )
                
                # Persiapkan data dan latih model
                if not predictor.prepare_data():
                    self.error.emit(f"Error preparing data for model {model}")
                    continue
                
                predictor.train_model()
                
                # Dapatkan data untuk backtesting
                y_true, y_pred, _ = predictor.predict()
                
                # Uji setiap strategi
                for strategy_name in self.strategies:
                    # Dapatkan parameter ranges untuk strategi ini
                    param_dict = self.param_ranges[strategy_name]
                    
                    # Buat semua kombinasi parameter
                    param_combinations = []
                    
                    if strategy_name == 'Trend Following':
                        for threshold in param_dict['threshold']:
                            param_combinations.append({'threshold': threshold})
                    
                    elif strategy_name == 'Mean Reversion':
                        for sma_period in param_dict['sma_period']:
                            for threshold in param_dict['threshold']:
                                param_combinations.append({
                                    'sma_period': sma_period, 
                                    'threshold': threshold
                                })
                    
                    elif strategy_name == 'Predictive':
                        for threshold in param_dict['threshold']:
                            param_combinations.append({'threshold': threshold})
                    
                    # Uji setiap kombinasi parameter
                    for params in param_combinations:
                        # Jalankan backtest dengan strategi dan parameter ini
                        portfolio_values, trades, performance = self.run_backtest_with_params(
                            y_true, y_pred, self.initial_investment, strategy_name, params
                        )
                        
                        # Simpan hasil
                        result = {
                            'model': model,
                            'strategy': strategy_name,
                            'parameters': params,
                            'performance': performance
                        }
                        results.append(result)
                        
                        # Update progress
                        completed += 1
                        progress_percent = int((completed / total_combinations) * 100)
                        self.progress.emit(progress_percent)
            
            # Urutkan hasil berdasarkan total return
            results.sort(key=lambda x: x['performance']['total_return'], reverse=True)
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def run_backtest_with_params(self, actual_prices, predicted_prices, initial_investment, strategy, params):
        # Inisialisasi portfolio
        cash = initial_investment
        shares = 0
        portfolio_values = []
        trades = []
        
        # Memastikan data memiliki panjang yang sama
        length = min(len(actual_prices), len(predicted_prices))
        actual_prices = actual_prices[:length]
        predicted_prices = predicted_prices[:length]
        
        # Iterasi melalui harga historis
        for i in range(1, length):
            # Hitung nilai portfolio saat ini
            portfolio_value = cash + shares * actual_prices[i]
            portfolio_values.append(portfolio_value)
            
            signal = self.generate_signal_with_params(predicted_prices, actual_prices, i, strategy, params)
            
            # Proses signals
            if signal == 'BUY' and cash > 0:
                # Beli saham sebanyak mungkin dengan uang yang ada
                shares_to_buy = cash / actual_prices[i]
                shares += shares_to_buy
                cash = 0
                trades.append({
                    'day': i,
                    'type': 'BUY',
                    'price': actual_prices[i],
                    'shares': shares_to_buy,
                    'value': shares_to_buy * actual_prices[i]
                })
            elif signal == 'SELL' and shares > 0:
                # Jual semua saham
                cash += shares * actual_prices[i]
                trades.append({
                    'day': i,
                    'type': 'SELL',
                    'price': actual_prices[i],
                    'shares': shares,
                    'value': shares * actual_prices[i]
                })
                shares = 0
        
        # Nilai akhir portfolio
        final_value = cash + shares * actual_prices[-1]
        
        # Menghitung metrik performa
        total_return = (final_value - initial_investment) / initial_investment * 100
        
        # Menghitung drawdown
        peak = portfolio_values[0] if portfolio_values else initial_investment
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100 if peak > 0 else 0
            drawdown = max(drawdown, dd)
        
        # Menghitung Sharpe Ratio (sangat disederhanakan, asumsi risk-free rate = 0)
        daily_returns = [portfolio_values[i]/portfolio_values[i-1]-1 for i in range(1, len(portfolio_values))] if len(portfolio_values) > 1 else []
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Menghitung profit/loss per trade
        win_trades = 0
        loss_trades = 0
        for i in range(0, len(trades), 2):
            if i+1 < len(trades):
                buy = trades[i]
                sell = trades[i+1]
                profit = sell['value'] - buy['value']
                if profit > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
        
        win_rate = 0
        if win_trades + loss_trades > 0:
            win_rate = win_trades / (win_trades + loss_trades) * 100
        
        performance = {
            'initial_investment': initial_investment,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return portfolio_values, trades, performance
    
    def generate_signal_with_params(self, predicted_prices, actual_prices, index, strategy, params):
        if strategy == 'Trend Following':
            threshold = params.get('threshold', 1.0)
            # Beli jika prediksi menunjukkan tren naik dengan threshold
            if index > 1 and predicted_prices[index] > predicted_prices[index-1] * (1 + threshold/100):
                return 'BUY'
            # Jual jika prediksi menunjukkan tren turun dengan threshold
            elif index > 1 and predicted_prices[index] < predicted_prices[index-1] * (1 - threshold/100):
                return 'SELL'
        
        elif strategy == 'Mean Reversion':
            sma_period = params.get('sma_period', 5)
            threshold = params.get('threshold', 2.0)
            
            # Pastikan kita memiliki cukup data untuk menghitung SMA
            if index >= sma_period:
                # Hitung SMA periode tertentu
                sma = sum(actual_prices[index-sma_period:index]) / sma_period
                # Beli jika harga di bawah rata-rata (oversold) dengan threshold
                if actual_prices[index] < sma * (1 - threshold/100):
                    return 'BUY'
                # Jual jika harga di atas rata-rata (overbought) dengan threshold
                elif actual_prices[index] > sma * (1 + threshold/100):
                    return 'SELL'
        
        elif strategy == 'Predictive':
            threshold = params.get('threshold', 1.0)
            # Beli jika prediksi menunjukkan harga akan naik dengan threshold
            if predicted_prices[index] > actual_prices[index] * (1 + threshold/100):
                return 'BUY'
            # Jual jika prediksi menunjukkan harga akan turun dengan threshold
            elif predicted_prices[index] < actual_prices[index] * (1 - threshold/100):
                return 'SELL'
        
        return 'HOLD'

# Tambahkan class worker baru untuk menangani simulasi trading pada prediksi
class ForecastTradeWorker(QThread):
    finished = pyqtSignal(object, object, object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, forecast_prices, initial_investment, strategy, params):
        super().__init__()
        self.forecast_prices = forecast_prices
        self.initial_investment = initial_investment
        self.strategy = strategy
        self.params = params
    
    def run(self):
        try:
            # Simulasikan trading pada harga prediksi
            portfolio_values, trades, performance = self.simulate_forecast_trading(
                self.forecast_prices, self.initial_investment, self.strategy, self.params
            )
            
            self.progress.emit(100)
            self.finished.emit(portfolio_values, trades, performance)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def simulate_forecast_trading(self, forecast_prices, initial_investment, strategy, params):
        # Inisialisasi portfolio
        cash = initial_investment
        shares = 0
        portfolio_values = [initial_investment]  # Initial portfolio value
        trades = []
        
        # Buat array harga dengan harga awal (hari ini) sebagai base
        prices = np.array(forecast_prices)
        
        # Jalankan strategi di forecast prices
        for i in range(1, len(prices)):
            prev_portfolio_value = portfolio_values[-1]
            
            # Generate signal berdasarkan strategi
            signal = 'HOLD'
            
            if strategy == 'Trend Following':
                threshold = params.get('threshold', 1.0)
                # Beli jika tren naik
                if prices[i] > prices[i-1] * (1 + threshold/100):
                    signal = 'BUY'
                # Jual jika tren turun
                elif prices[i] < prices[i-1] * (1 - threshold/100):
                    signal = 'SELL'
            
            elif strategy == 'Mean Reversion':
                window = min(i+1, params.get('window', 3))
                if i >= window - 1:
                    # Hitung SMA
                    sma = np.mean(prices[max(0, i-window+1):i+1])
                    threshold = params.get('threshold', 2.0)
                    
                    # Beli jika di bawah SMA (oversold)
                    if prices[i] < sma * (1 - threshold/100):
                        signal = 'BUY'
                    # Jual jika di atas SMA (overbought)
                    elif prices[i] > sma * (1 + threshold/100):
                        signal = 'SELL'
            
            elif strategy == 'Threshold':
                buy_threshold = params.get('buy_threshold', 2.0)
                sell_threshold = params.get('sell_threshold', 2.0)
                
                # Beli jika harga naik melebihi buy threshold
                day_return = (prices[i] / prices[i-1] - 1) * 100
                if day_return >= buy_threshold:
                    signal = 'BUY'
                # Jual jika harga turun melebihi sell threshold
                elif day_return <= -sell_threshold:
                    signal = 'SELL'
            
            # Proses signals
            if signal == 'BUY' and cash > 0:
                # Beli saham sebanyak mungkin dengan uang yang ada
                shares_to_buy = cash / prices[i]
                shares += shares_to_buy
                cash = 0
                trades.append({
                    'day': i,
                    'type': 'BUY',
                    'price': prices[i],
                    'shares': shares_to_buy,
                    'value': shares_to_buy * prices[i]
                })
            elif signal == 'SELL' and shares > 0:
                # Jual semua saham
                cash += shares * prices[i]
                trades.append({
                    'day': i,
                    'type': 'SELL',
                    'price': prices[i],
                    'shares': shares,
                    'value': shares * prices[i]
                })
                shares = 0
            
            # Update portfolio value
            portfolio_value = cash + shares * prices[i]
            portfolio_values.append(portfolio_value)
        
        # Liquidate at end if still holding shares
        if shares > 0:
            final_value = cash + shares * prices[-1]
        else:
            final_value = cash
        
        # Hitung metrics
        total_return = (final_value - initial_investment) / initial_investment * 100
        
        # Hitung drawdown
        peak = portfolio_values[0]
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown = max(drawdown, dd)
        
        # Hitung win rate
        win_trades = 0
        loss_trades = 0
        for i in range(0, len(trades), 2):
            if i+1 < len(trades):
                buy = trades[i]
                sell = trades[i+1]
                profit = sell['value'] - buy['value']
                if profit > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
        
        win_rate = 0
        if win_trades + loss_trades > 0:
            win_rate = win_trades / (win_trades + loss_trades) * 100
        
        performance = {
            'initial_investment': initial_investment,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return portfolio_values, trades, performance

class StockPredictionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Price Prediction")
        self.setGeometry(100, 100, 1200, 800)
        
        # Tab widget untuk memisahkan prediksi, backtesting, dan optimizer
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Tab untuk prediksi
        self.prediction_tab = QWidget()
        self.setup_prediction_tab()
        self.tabs.addTab(self.prediction_tab, "Prediksi")
        
        # Tab untuk backtesting
        self.backtest_tab = QWidget()
        self.setup_backtest_tab()
        self.tabs.addTab(self.backtest_tab, "Backtesting")
        
        # Tab untuk optimizer
        self.optimizer_tab = QWidget()
        self.setup_optimizer_tab()
        self.tabs.addTab(self.optimizer_tab, "Optimizer Strategi")
        
        # Tab untuk simulasi trading pada forecast
        self.forecast_trade_tab = QWidget()
        self.setup_forecast_trade_tab()
        self.tabs.addTab(self.forecast_trade_tab, "Trading Prediksi")
    
    def setup_prediction_tab(self):
        # Widget utama
        layout = QVBoxLayout(self.prediction_tab)
        
        # Input parameters
        input_layout = QHBoxLayout()
        
        # Ticker input
        ticker_layout = QVBoxLayout()
        ticker_label = QLabel("Ticker Symbol:")
        self.ticker_input = QLineEdit("ADRO.JK")
        ticker_layout.addWidget(ticker_label)
        ticker_layout.addWidget(self.ticker_input)
        input_layout.addLayout(ticker_layout)
        
        # Date inputs
        date_layout = QVBoxLayout()
        start_date_label = QLabel("Start Date:")
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate(2020, 1, 1))
        end_date_label = QLabel("End Date:")
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        date_layout.addWidget(start_date_label)
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(end_date_label)
        date_layout.addWidget(self.end_date)
        input_layout.addLayout(date_layout)
        
        # Model selection
        model_layout = QVBoxLayout()
        model_label = QLabel("Model Type:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['cnn_lstm', 'bilstm', 'transformer', 'ensemble'])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        input_layout.addLayout(model_layout)
        
        # Parameters
        param_layout = QVBoxLayout()
        lookback_label = QLabel("Lookback Days:")
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(10, 200)
        self.lookback_spin.setValue(60)
        forecast_label = QLabel("Forecast Days:")
        self.forecast_spin = QSpinBox()
        self.forecast_spin.setRange(1, 100)
        self.forecast_spin.setValue(20)
        param_layout.addWidget(lookback_label)
        param_layout.addWidget(self.lookback_spin)
        param_layout.addWidget(forecast_label)
        param_layout.addWidget(self.forecast_spin)
        input_layout.addLayout(param_layout)
        
        # Hyperparameter tuning
        tune_layout = QVBoxLayout()
        self.tune_checkbox = QCheckBox("Enable Hyperparameter Tuning")
        tune_layout.addWidget(self.tune_checkbox)
        input_layout.addLayout(tune_layout)
        
        layout.addLayout(input_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Run button
        self.run_button = QPushButton("Run Prediction")
        self.run_button.clicked.connect(self.run_prediction)
        layout.addWidget(self.run_button)
        
        # Plot area
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Results area
        results_layout = QHBoxLayout()
        
        # Metrics
        metrics_layout = QVBoxLayout()
        metrics_label = QLabel("Model Metrics:")
        self.metrics_text = QLabel()
        self.metrics_text.setWordWrap(True)
        metrics_layout.addWidget(metrics_label)
        metrics_layout.addWidget(self.metrics_text)
        results_layout.addLayout(metrics_layout)
        
        # Forecast
        forecast_layout = QVBoxLayout()
        forecast_label = QLabel("Forecast Results:")
        self.forecast_text = QLabel()
        self.forecast_text.setWordWrap(True)
        forecast_layout.addWidget(forecast_label)
        forecast_layout.addWidget(self.forecast_text)
        results_layout.addLayout(forecast_layout)
        
        layout.addLayout(results_layout)
        
        # Save button
        save_layout = QHBoxLayout()
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_plot)
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self.save_results)
        save_layout.addWidget(self.save_plot_button)
        save_layout.addWidget(self.save_results_button)
        layout.addLayout(save_layout)
        
        # Disable buttons initially
        self.save_plot_button.setEnabled(False)
        self.save_results_button.setEnabled(False)
    
    def run_prediction(self):
        # Disable run button during prediction
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Get parameters
        ticker = self.ticker_input.text()
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")
        model_type = self.model_combo.currentText()
        lookback = self.lookback_spin.value()
        forecast_days = self.forecast_spin.value()
        tune = self.tune_checkbox.isChecked()
        
        # Create predictor
        self.predictor = StockPredictor(
            ticker, start_date, end_date,
            lookback, forecast_days, model_type, tune
        )
        
        # Create and start worker thread
        self.worker = WorkerThread(self.predictor)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.run_button.setEnabled(True)
    
    def on_prediction_finished(self, y_true, y_pred, forecast, metrics):
        # Update plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(y_true, label='Actual', color='blue')
        ax.plot(y_pred, label='Predicted', color='red', linestyle='--')
        ax.plot(range(len(y_true), len(y_true) + len(forecast)), 
                forecast, label='Forecast', color='green', linestyle='-.')
        ax.set_title(f'{self.predictor.ticker} Stock Price Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        self.canvas.draw()
        
        # Update metrics
        metrics_text = f"MSE: {metrics['mse']:.4f}\n"
        metrics_text += f"RMSE: {metrics['rmse']:.4f}\n"
        metrics_text += f"MAE: {metrics['mae']:.4f}\n"
        metrics_text += f"R2 Score: {metrics['r2']:.4f}"
        self.metrics_text.setText(metrics_text)
        
        # Update forecast
        forecast_text = ""
        for i, price in enumerate(forecast, 1):
            forecast_text += f"Day {i}: {price:.2f}\n"
        self.forecast_text.setText(forecast_text)
        
        # Enable buttons
        self.run_button.setEnabled(True)
        self.save_plot_button.setEnabled(True)
        self.save_results_button.setEnabled(True)
    
    def save_plot(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png);;All Files (*)"
        )
        if file_name:
            self.figure.savefig(file_name)
    
    def save_results(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_name:
            results = {
                'Metrics': self.metrics_text.text().split('\n'),
                'Forecast': self.forecast_text.text().split('\n')
            }
            df = pd.DataFrame(results)
            df.to_csv(file_name, index=False)

    def setup_backtest_tab(self):
        layout = QVBoxLayout(self.backtest_tab)
        
        # Input parameters for backtesting
        input_layout = QHBoxLayout()
        
        # Ticker input (sama dengan tab prediksi)
        ticker_layout = QVBoxLayout()
        ticker_label = QLabel("Ticker Symbol:")
        self.bt_ticker_input = QLineEdit("ADRO.JK")
        ticker_layout.addWidget(ticker_label)
        ticker_layout.addWidget(self.bt_ticker_input)
        input_layout.addLayout(ticker_layout)
        
        # Date inputs (sama dengan tab prediksi)
        date_layout = QVBoxLayout()
        start_date_label = QLabel("Start Date:")
        self.bt_start_date = QDateEdit()
        self.bt_start_date.setDate(QDate(2020, 1, 1))
        end_date_label = QLabel("End Date:")
        self.bt_end_date = QDateEdit()
        self.bt_end_date.setDate(QDate.currentDate())
        date_layout.addWidget(start_date_label)
        date_layout.addWidget(self.bt_start_date)
        date_layout.addWidget(end_date_label)
        date_layout.addWidget(self.bt_end_date)
        input_layout.addLayout(date_layout)
        
        # Model selection (sama dengan tab prediksi)
        model_layout = QVBoxLayout()
        model_label = QLabel("Model Type:")
        self.bt_model_combo = QComboBox()
        self.bt_model_combo.addItems(['cnn_lstm', 'bilstm', 'transformer', 'ensemble'])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.bt_model_combo)
        input_layout.addLayout(model_layout)
        
        # Parameter khusus backtesting
        bt_param_layout = QVBoxLayout()
        
        # Initial investment
        investment_label = QLabel("Initial Investment ($):")
        self.investment_spin = QDoubleSpinBox()
        self.investment_spin.setRange(1000, 1000000)
        self.investment_spin.setValue(10000)
        self.investment_spin.setSingleStep(1000)
        
        # Trading strategy
        strategy_label = QLabel("Trading Strategy:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['Trend Following', 'Mean Reversion', 'Predictive'])
        
        bt_param_layout.addWidget(investment_label)
        bt_param_layout.addWidget(self.investment_spin)
        bt_param_layout.addWidget(strategy_label)
        bt_param_layout.addWidget(self.strategy_combo)
        
        input_layout.addLayout(bt_param_layout)
        
        layout.addLayout(input_layout)
        
        # Progress bar
        self.bt_progress_bar = QProgressBar()
        layout.addWidget(self.bt_progress_bar)
        
        # Run button
        self.bt_run_button = QPushButton("Run Backtest")
        self.bt_run_button.clicked.connect(self.run_backtest)
        layout.addWidget(self.bt_run_button)
        
        # Plot area
        self.bt_figure = Figure(figsize=(8, 4))
        self.bt_canvas = FigureCanvas(self.bt_figure)
        layout.addWidget(self.bt_canvas)
        
        # Tabel untuk trades
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(5)
        self.trades_table.setHorizontalHeaderLabels(['Day', 'Type', 'Price', 'Shares', 'Value'])
        header = self.trades_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.trades_table)
        
        # Performance metrics
        self.bt_metrics_text = QLabel()
        self.bt_metrics_text.setWordWrap(True)
        layout.addWidget(self.bt_metrics_text)
        
        # Save button
        bt_save_layout = QHBoxLayout()
        self.bt_save_plot_button = QPushButton("Save Backtest Plot")
        self.bt_save_plot_button.clicked.connect(self.save_backtest_plot)
        self.bt_save_results_button = QPushButton("Save Backtest Results")
        self.bt_save_results_button.clicked.connect(self.save_backtest_results)
        bt_save_layout.addWidget(self.bt_save_plot_button)
        bt_save_layout.addWidget(self.bt_save_results_button)
        layout.addLayout(bt_save_layout)
        
        # Disable buttons initially
        self.bt_save_plot_button.setEnabled(False)
        self.bt_save_results_button.setEnabled(False)
    
    def run_backtest(self):
        # Disable run button during backtesting
        self.bt_run_button.setEnabled(False)
        self.bt_progress_bar.setValue(0)
        
        # Get parameters
        ticker = self.bt_ticker_input.text()
        start_date = self.bt_start_date.date().toString("yyyy-MM-dd")
        end_date = self.bt_end_date.date().toString("yyyy-MM-dd")
        model_type = self.bt_model_combo.currentText()
        initial_investment = self.investment_spin.value()
        strategy = self.strategy_combo.currentText()
        
        # Create predictor
        self.bt_predictor = StockPredictor(
            ticker, start_date, end_date,
            60, 20, model_type, False  # Menggunakan nilai default untuk lookback dan forecast
        )
        
        # Create and start worker thread
        self.bt_worker = BacktestWorker(self.bt_predictor, initial_investment, strategy)
        self.bt_worker.finished.connect(self.on_backtest_finished)
        self.bt_worker.progress.connect(self.update_backtest_progress)
        self.bt_worker.error.connect(self.show_backtest_error)
        self.bt_worker.start()
    
    def update_backtest_progress(self, value):
        self.bt_progress_bar.setValue(value)
    
    def show_backtest_error(self, message):
        QMessageBox.critical(self, "Backtest Error", message)
        self.bt_run_button.setEnabled(True)
    
    def on_backtest_finished(self, portfolio_values, trades, performance):
        # Update plot
        self.bt_figure.clear()
        ax1 = self.bt_figure.add_subplot(111)
        
        # Plot portfolio value
        ax1.plot(portfolio_values, label='Portfolio Value', color='blue')
        ax1.set_title(f'Backtest Results for {self.bt_predictor.ticker}')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Highlight buy/sell points
        for trade in trades:
            if trade['type'] == 'BUY':
                ax1.plot(trade['day'], portfolio_values[trade['day']-1], 'g^', markersize=10)
            else:  # SELL
                ax1.plot(trade['day'], portfolio_values[trade['day']-1], 'rv', markersize=10)
        
        self.bt_canvas.draw()
        
        # Update trades table
        self.trades_table.setRowCount(len(trades))
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade['day'])))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade['type']))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"${trade['price']:.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(f"{trade['shares']:.2f}"))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"${trade['value']:.2f}"))
        
        # Update metrics
        metrics_text = f"Initial Investment: ${performance['initial_investment']:,.2f}\n"
        metrics_text += f"Final Value: ${performance['final_value']:,.2f}\n"
        metrics_text += f"Total Return: {performance['total_return']:.2f}%\n"
        metrics_text += f"Max Drawdown: {performance['max_drawdown']:.2f}%\n"
        metrics_text += f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}\n"
        metrics_text += f"Win Rate: {performance['win_rate']:.2f}%\n"
        metrics_text += f"Number of Trades: {performance['num_trades']}"
        
        self.bt_metrics_text.setText(metrics_text)
        
        # Enable buttons
        self.bt_run_button.setEnabled(True)
        self.bt_save_plot_button.setEnabled(True)
        self.bt_save_results_button.setEnabled(True)
    
    def save_backtest_plot(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Backtest Plot", "", "PNG Files (*.png);;All Files (*)"
        )
        if file_name:
            self.bt_figure.savefig(file_name)
    
    def save_backtest_results(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Backtest Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_name:
            # Ekstrak data dari trades table
            trades_data = []
            for row in range(self.trades_table.rowCount()):
                trade = {
                    'Day': self.trades_table.item(row, 0).text(),
                    'Type': self.trades_table.item(row, 1).text(),
                    'Price': self.trades_table.item(row, 2).text(),
                    'Shares': self.trades_table.item(row, 3).text(),
                    'Value': self.trades_table.item(row, 4).text()
                }
                trades_data.append(trade)
            
            # Buat dataframe untuk trades
            trades_df = pd.DataFrame(trades_data)
            
            # Tambahkan metrik performa
            performance_data = {
                'Metric': ['Initial Investment', 'Final Value', 'Total Return', 
                          'Max Drawdown', 'Sharpe Ratio', 'Win Rate', 'Number of Trades'],
                'Value': self.bt_metrics_text.text().split('\n')
            }
            performance_df = pd.DataFrame(performance_data)
            
            # Simpan ke Excel dengan multiple sheets
            with pd.ExcelWriter(file_name) as writer:
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
                performance_df.to_excel(writer, sheet_name='Performance', index=False)
    
    def setup_optimizer_tab(self):
        layout = QVBoxLayout(self.optimizer_tab)
        
        # Input parameters
        input_layout = QGridLayout()
        
        # Ticker input
        ticker_label = QLabel("Ticker Symbol:")
        self.opt_ticker_input = QLineEdit("ADRO.JK")
        input_layout.addWidget(ticker_label, 0, 0)
        input_layout.addWidget(self.opt_ticker_input, 0, 1)
        
        # Date inputs
        start_date_label = QLabel("Start Date:")
        self.opt_start_date = QDateEdit()
        self.opt_start_date.setDate(QDate(2020, 1, 1))
        input_layout.addWidget(start_date_label, 1, 0)
        input_layout.addWidget(self.opt_start_date, 1, 1)
        
        end_date_label = QLabel("End Date:")
        self.opt_end_date = QDateEdit()
        self.opt_end_date.setDate(QDate.currentDate())
        input_layout.addWidget(end_date_label, 2, 0)
        input_layout.addWidget(self.opt_end_date, 2, 1)
        
        # Initial investment
        investment_label = QLabel("Initial Investment ($):")
        self.opt_investment_spin = QDoubleSpinBox()
        self.opt_investment_spin.setRange(1000, 1000000)
        self.opt_investment_spin.setValue(10000)
        self.opt_investment_spin.setSingleStep(1000)
        input_layout.addWidget(investment_label, 3, 0)
        input_layout.addWidget(self.opt_investment_spin, 3, 1)
        
        # Strategi untuk testing
        strategy_group = QGroupBox("Strategi untuk Testing")
        strategy_layout = QVBoxLayout()
        
        self.opt_trend_following_cb = QCheckBox("Trend Following")
        self.opt_trend_following_cb.setChecked(True)
        strategy_layout.addWidget(self.opt_trend_following_cb)
        
        self.opt_mean_reversion_cb = QCheckBox("Mean Reversion")
        self.opt_mean_reversion_cb.setChecked(True)
        strategy_layout.addWidget(self.opt_mean_reversion_cb)
        
        self.opt_predictive_cb = QCheckBox("Predictive")
        self.opt_predictive_cb.setChecked(True)
        strategy_layout.addWidget(self.opt_predictive_cb)
        
        strategy_group.setLayout(strategy_layout)
        input_layout.addWidget(strategy_group, 0, 2, 4, 1)
        
        # Model untuk testing
        model_group = QGroupBox("Model untuk Testing")
        model_layout = QVBoxLayout()
        
        self.opt_cnn_lstm_cb = QCheckBox("CNN-LSTM")
        self.opt_cnn_lstm_cb.setChecked(True)
        model_layout.addWidget(self.opt_cnn_lstm_cb)
        
        self.opt_bilstm_cb = QCheckBox("Bidirectional LSTM")
        self.opt_bilstm_cb.setChecked(True)
        model_layout.addWidget(self.opt_bilstm_cb)
        
        self.opt_transformer_cb = QCheckBox("Transformer")
        self.opt_transformer_cb.setChecked(True)
        model_layout.addWidget(self.opt_transformer_cb)
        
        self.opt_ensemble_cb = QCheckBox("Ensemble")
        self.opt_ensemble_cb.setChecked(True)
        model_layout.addWidget(self.opt_ensemble_cb)
        
        model_group.setLayout(model_layout)
        input_layout.addWidget(model_group, 0, 3, 4, 1)
        
        layout.addLayout(input_layout)
        
        # Progress bar
        self.opt_progress_bar = QProgressBar()
        layout.addWidget(self.opt_progress_bar)
        
        # Run button
        self.opt_run_button = QPushButton("Cari Strategi Terbaik")
        self.opt_run_button.clicked.connect(self.run_strategy_optimization)
        layout.addWidget(self.opt_run_button)
        
        # Results table
        self.opt_results_table = QTableWidget()
        self.opt_results_table.setColumnCount(10)
        self.opt_results_table.setHorizontalHeaderLabels([
            'Rank', 'Model', 'Strategi', 'Parameter', 'Return (%)', 
            'Drawdown (%)', 'Sharpe Ratio', 'Win Rate (%)', 
            'Jumlah Trades', 'Final Value ($)'
        ])
        header = self.opt_results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.opt_results_table)
        
        # Save button
        self.opt_save_button = QPushButton("Simpan Hasil Optimasi")
        self.opt_save_button.clicked.connect(self.save_optimization_results)
        self.opt_save_button.setEnabled(False)
        layout.addWidget(self.opt_save_button)
    
    def run_strategy_optimization(self):
        # Disable run button
        self.opt_run_button.setEnabled(False)
        self.opt_progress_bar.setValue(0)
        
        # Get parameters
        ticker = self.opt_ticker_input.text()
        start_date = self.opt_start_date.date().toString("yyyy-MM-dd")
        end_date = self.opt_end_date.date().toString("yyyy-MM-dd")
        initial_investment = self.opt_investment_spin.value()
        
        # Get selected strategies
        strategies = []
        if self.opt_trend_following_cb.isChecked():
            strategies.append('Trend Following')
        if self.opt_mean_reversion_cb.isChecked():
            strategies.append('Mean Reversion')
        if self.opt_predictive_cb.isChecked():
            strategies.append('Predictive')
        
        if not strategies:
            QMessageBox.warning(self, "Warning", "Pilih setidaknya satu strategi untuk diuji.")
            self.opt_run_button.setEnabled(True)
            return
        
        # Get selected models
        models = []
        if self.opt_cnn_lstm_cb.isChecked():
            models.append('cnn_lstm')
        if self.opt_bilstm_cb.isChecked():
            models.append('bilstm')
        if self.opt_transformer_cb.isChecked():
            models.append('transformer')
        if self.opt_ensemble_cb.isChecked():
            models.append('ensemble')
        
        if not models:
            QMessageBox.warning(self, "Warning", "Pilih setidaknya satu model untuk diuji.")
            self.opt_run_button.setEnabled(True)
            return
        
        # Create and start worker thread
        self.opt_worker = StrategyOptimizerWorker(
            ticker, start_date, end_date, models, initial_investment, strategies
        )
        self.opt_worker.finished.connect(self.on_optimization_finished)
        self.opt_worker.progress.connect(self.update_opt_progress)
        self.opt_worker.error.connect(self.show_opt_error)
        self.opt_worker.start()
    
    def update_opt_progress(self, value):
        self.opt_progress_bar.setValue(value)
    
    def show_opt_error(self, message):
        QMessageBox.critical(self, "Optimization Error", message)
        self.opt_run_button.setEnabled(True)
    
    def on_optimization_finished(self, results):
        # Update results table
        self.opt_results = results  # Save for later use
        self.opt_results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            # Rank
            self.opt_results_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            
            # Model
            self.opt_results_table.setItem(i, 1, QTableWidgetItem(result['model']))
            
            # Strategy
            self.opt_results_table.setItem(i, 2, QTableWidgetItem(result['strategy']))
            
            # Parameters
            param_str = ", ".join([f"{k}: {v}" for k, v in result['parameters'].items()])
            self.opt_results_table.setItem(i, 3, QTableWidgetItem(param_str))
            
            # Performance metrics
            perf = result['performance']
            self.opt_results_table.setItem(i, 4, QTableWidgetItem(f"{perf['total_return']:.2f}"))
            self.opt_results_table.setItem(i, 5, QTableWidgetItem(f"{perf['max_drawdown']:.2f}"))
            self.opt_results_table.setItem(i, 6, QTableWidgetItem(f"{perf['sharpe_ratio']:.2f}"))
            self.opt_results_table.setItem(i, 7, QTableWidgetItem(f"{perf['win_rate']:.2f}"))
            self.opt_results_table.setItem(i, 8, QTableWidgetItem(str(perf['num_trades'])))
            self.opt_results_table.setItem(i, 9, QTableWidgetItem(f"${perf['final_value']:,.2f}"))
        
        # Enable buttons
        self.opt_run_button.setEnabled(True)
        self.opt_save_button.setEnabled(True)
        
        # Show best strategy in a message box
        if results:
            best = results[0]
            perf = best['performance']
            message = f"Strategi Terbaik Ditemukan!\n\n"
            message += f"Model: {best['model']}\n"
            message += f"Strategi: {best['strategy']}\n"
            message += f"Parameter: {', '.join([f'{k}: {v}' for k, v in best['parameters'].items()])}\n\n"
            message += f"Return: {perf['total_return']:.2f}%\n"
            message += f"Drawdown Maksimum: {perf['max_drawdown']:.2f}%\n"
            message += f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}\n"
            message += f"Win Rate: {perf['win_rate']:.2f}%\n"
            message += f"Jumlah Transaksi: {perf['num_trades']}\n"
            message += f"Nilai Akhir: ${perf['final_value']:,.2f}"
            
            QMessageBox.information(self, "Hasil Optimasi", message)
    
    def save_optimization_results(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Optimization Results", "", "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        if not file_name:
            return
        
        # Convert results to DataFrame
        data = []
        for i, result in enumerate(self.opt_results):
            perf = result['performance']
            data.append({
                'Rank': i + 1,
                'Model': result['model'],
                'Strategy': result['strategy'],
                'Parameters': str(result['parameters']),
                'Total Return (%)': perf['total_return'],
                'Max Drawdown (%)': perf['max_drawdown'],
                'Sharpe Ratio': perf['sharpe_ratio'],
                'Win Rate (%)': perf['win_rate'],
                'Number of Trades': perf['num_trades'],
                'Final Value ($)': perf['final_value'],
                'Initial Investment ($)': perf['initial_investment']
            })
        
        df = pd.DataFrame(data)
        
        # Save based on file extension
        if file_name.endswith('.xlsx'):
            df.to_excel(file_name, index=False)
        else:
            df.to_csv(file_name, index=False)
        
        QMessageBox.information(self, "Simpan Berhasil", "Hasil optimasi berhasil disimpan!")

    def setup_forecast_trade_tab(self):
        layout = QVBoxLayout(self.forecast_trade_tab)
        
        # Input parameters section
        params_group = QGroupBox("Parameter Simulasi Trading")
        params_layout = QGridLayout()
        
        # Ticker input
        ticker_label = QLabel("Ticker Symbol:")
        self.forecast_ticker_input = QLineEdit("ADRO.JK")
        params_layout.addWidget(ticker_label, 0, 0)
        params_layout.addWidget(self.forecast_ticker_input, 0, 1)
        
        # Model selection
        model_label = QLabel("Model Prediksi:")
        self.forecast_model_combo = QComboBox()
        self.forecast_model_combo.addItems(['cnn_lstm', 'bilstm', 'transformer', 'ensemble'])
        params_layout.addWidget(model_label, 1, 0)
        params_layout.addWidget(self.forecast_model_combo, 1, 1)
        
        # Forecast days
        forecast_days_label = QLabel("Jumlah Hari Prediksi:")
        self.forecast_days_spin = QSpinBox()
        self.forecast_days_spin.setRange(5, 60)
        self.forecast_days_spin.setValue(20)
        params_layout.addWidget(forecast_days_label, 2, 0)
        params_layout.addWidget(self.forecast_days_spin, 2, 1)
        
        # Initial investment
        investment_label = QLabel("Modal Awal (Rp):")
        self.forecast_investment_spin = QDoubleSpinBox()
        self.forecast_investment_spin.setRange(1000, 1000000000)
        self.forecast_investment_spin.setValue(10000000)
        self.forecast_investment_spin.setSingleStep(1000000)
        params_layout.addWidget(investment_label, 3, 0)
        params_layout.addWidget(self.forecast_investment_spin, 3, 1)
        
        # Strategy section
        strategy_label = QLabel("Strategi Trading:")
        params_layout.addWidget(strategy_label, 0, 2)
        
        # Strategy radio buttons
        self.strategy_group = QGroupBox()
        strategy_button_layout = QVBoxLayout()
        
        self.trend_following_radio = QRadioButton("Trend Following")
        self.trend_following_radio.setChecked(True)
        self.trend_following_radio.toggled.connect(self.on_strategy_changed)
        strategy_button_layout.addWidget(self.trend_following_radio)
        
        self.mean_reversion_radio = QRadioButton("Mean Reversion")
        self.mean_reversion_radio.toggled.connect(self.on_strategy_changed)
        strategy_button_layout.addWidget(self.mean_reversion_radio)
        
        self.threshold_radio = QRadioButton("Threshold")
        self.threshold_radio.toggled.connect(self.on_strategy_changed)
        strategy_button_layout.addWidget(self.threshold_radio)
        
        self.strategy_group.setLayout(strategy_button_layout)
        params_layout.addWidget(self.strategy_group, 1, 2, 3, 1)
        
        # Strategy parameters (akan diubah sesuai strategi yang dipilih)
        self.strategy_params_group = QGroupBox("Parameter Strategi")
        self.strategy_params_layout = QGridLayout()
        
        # Default: Trend Following parameters
        threshold_label = QLabel("Threshold (%):")
        self.trend_threshold_spin = QDoubleSpinBox()
        self.trend_threshold_spin.setRange(0.1, 10.0)
        self.trend_threshold_spin.setValue(1.0)
        self.trend_threshold_spin.setSingleStep(0.1)
        self.strategy_params_layout.addWidget(threshold_label, 0, 0)
        self.strategy_params_layout.addWidget(self.trend_threshold_spin, 0, 1)
        
        # Mean Reversion parameters (akan ditampilkan saat Mean Reversion dipilih)
        window_label = QLabel("Window Size:")
        self.mean_window_spin = QSpinBox()
        self.mean_window_spin.setRange(2, 20)
        self.mean_window_spin.setValue(5)
        self.mean_window_spin.hide()
        
        mean_threshold_label = QLabel("Threshold (%):")
        self.mean_threshold_spin = QDoubleSpinBox()
        self.mean_threshold_spin.setRange(0.1, 10.0)
        self.mean_threshold_spin.setValue(2.0)
        self.mean_threshold_spin.setSingleStep(0.1)
        self.mean_threshold_spin.hide()
        
        self.strategy_params_layout.addWidget(window_label, 1, 0)
        self.strategy_params_layout.addWidget(self.mean_window_spin, 1, 1)
        self.strategy_params_layout.addWidget(mean_threshold_label, 2, 0)
        self.strategy_params_layout.addWidget(self.mean_threshold_spin, 2, 1)
        
        # Threshold parameters (akan ditampilkan saat Threshold dipilih)
        buy_threshold_label = QLabel("Buy Threshold (%):")
        self.buy_threshold_spin = QDoubleSpinBox()
        self.buy_threshold_spin.setRange(0.1, 10.0)
        self.buy_threshold_spin.setValue(1.5)
        self.buy_threshold_spin.setSingleStep(0.1)
        self.buy_threshold_spin.hide()
        
        sell_threshold_label = QLabel("Sell Threshold (%):")
        self.sell_threshold_spin = QDoubleSpinBox()
        self.sell_threshold_spin.setRange(0.1, 10.0)
        self.sell_threshold_spin.setValue(1.5)
        self.sell_threshold_spin.setSingleStep(0.1)
        self.sell_threshold_spin.hide()
        
        self.strategy_params_layout.addWidget(buy_threshold_label, 3, 0)
        self.strategy_params_layout.addWidget(self.buy_threshold_spin, 3, 1)
        self.strategy_params_layout.addWidget(sell_threshold_label, 4, 0)
        self.strategy_params_layout.addWidget(self.sell_threshold_spin, 4, 1)
        
        self.strategy_params_group.setLayout(self.strategy_params_layout)
        params_layout.addWidget(self.strategy_params_group, 0, 3, 4, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Progress bar
        self.forecast_progress_bar = QProgressBar()
        layout.addWidget(self.forecast_progress_bar)
        
        # Run button
        self.forecast_run_button = QPushButton("Jalankan Simulasi Trading")
        self.forecast_run_button.clicked.connect(self.run_forecast_trading)
        layout.addWidget(self.forecast_run_button)
        
        # Results section - split into two parts
        results_layout = QHBoxLayout()
        
        # Left side: Chart
        chart_group = QGroupBox("Grafik Simulasi")
        chart_layout = QVBoxLayout()
        
        self.forecast_figure = Figure(figsize=(6, 4))
        self.forecast_canvas = FigureCanvas(self.forecast_figure)
        chart_layout.addWidget(self.forecast_canvas)
        
        chart_group.setLayout(chart_layout)
        results_layout.addWidget(chart_group)
        
        # Right side: Results
        results_group = QGroupBox("Hasil Simulasi")
        results_right_layout = QVBoxLayout()
        
        # Performance metrics
        self.forecast_metrics_label = QLabel("Kinerja Trading:")
        self.forecast_metrics_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        results_right_layout.addWidget(self.forecast_metrics_label)
        
        # Trades table
        trades_label = QLabel("Transaksi:")
        results_right_layout.addWidget(trades_label)
        
        self.forecast_trades_table = QTableWidget()
        self.forecast_trades_table.setColumnCount(4)
        self.forecast_trades_table.setHorizontalHeaderLabels(['Hari', 'Tipe', 'Harga', 'Nilai'])
        self.forecast_trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_right_layout.addWidget(self.forecast_trades_table)
        
        results_group.setLayout(results_right_layout)
        results_layout.addWidget(results_group)
        
        layout.addLayout(results_layout)
        
        # Save button
        self.forecast_save_button = QPushButton("Simpan Hasil Simulasi")
        self.forecast_save_button.clicked.connect(self.save_forecast_results)
        self.forecast_save_button.setEnabled(False)
        layout.addWidget(self.forecast_save_button)
    
    def on_strategy_changed(self):
        # Update strategi parameters yang ditampilkan berdasarkan strategi yang dipilih
        if self.trend_following_radio.isChecked():
            # Tampilkan trend following parameters
            self.trend_threshold_spin.show()
            self.mean_window_spin.hide()
            self.mean_threshold_spin.hide()
            self.buy_threshold_spin.hide()
            self.sell_threshold_spin.hide()
        
        elif self.mean_reversion_radio.isChecked():
            # Tampilkan mean reversion parameters
            self.trend_threshold_spin.hide()
            self.mean_window_spin.show()
            self.mean_threshold_spin.show()
            self.buy_threshold_spin.hide()
            self.sell_threshold_spin.hide()
        
        elif self.threshold_radio.isChecked():
            # Tampilkan threshold parameters
            self.trend_threshold_spin.hide()
            self.mean_window_spin.hide()
            self.mean_threshold_spin.hide()
            self.buy_threshold_spin.show()
            self.sell_threshold_spin.show()
    
    def run_forecast_trading(self):
        # Disable run button
        self.forecast_run_button.setEnabled(False)
        self.forecast_progress_bar.setValue(0)
        
        # Get parameters
        ticker = self.forecast_ticker_input.text()
        model_type = self.forecast_model_combo.currentText()
        forecast_days = self.forecast_days_spin.value()
        initial_investment = self.forecast_investment_spin.value()
        
        # Get selected strategy
        if self.trend_following_radio.isChecked():
            strategy = 'Trend Following'
            params = {'threshold': self.trend_threshold_spin.value()}
        elif self.mean_reversion_radio.isChecked():
            strategy = 'Mean Reversion'
            params = {
                'window': self.mean_window_spin.value(),
                'threshold': self.mean_threshold_spin.value()
            }
        elif self.threshold_radio.isChecked():
            strategy = 'Threshold'
            params = {
                'buy_threshold': self.buy_threshold_spin.value(),
                'sell_threshold': self.sell_threshold_spin.value()
            }
        
        # Buat predictor
        self.forecast_predictor = StockPredictor(
            ticker, 
            (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),  # 1 tahun ke belakang
            datetime.now().strftime('%Y-%m-%d'),  # Hari ini
            60, forecast_days, model_type, False
        )
        
        # Update progress
        self.forecast_progress_bar.setValue(10)
        
        try:
            # Persiapan data dan model
            if not self.forecast_predictor.prepare_data():
                QMessageBox.critical(self, "Error", "Tidak dapat mempersiapkan data")
                self.forecast_run_button.setEnabled(True)
                return
            
            self.forecast_progress_bar.setValue(30)
            
            # Train model
            self.forecast_predictor.train_model()
            self.forecast_progress_bar.setValue(60)
            
            # Get forecast
            _, _, forecast = self.forecast_predictor.predict()
            self.forecast_progress_bar.setValue(70)
            
            # Simpan forecast untuk simulasi trading
            self.forecast_prices = forecast
            
            # Jalankan simulasi trading pada forecast
            self.forecast_worker = ForecastTradeWorker(
                self.forecast_prices, initial_investment, strategy, params
            )
            self.forecast_worker.finished.connect(self.on_forecast_trading_finished)
            self.forecast_worker.progress.connect(self.update_forecast_progress)
            self.forecast_worker.error.connect(self.show_forecast_error)
            self.forecast_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.forecast_run_button.setEnabled(True)
    
    def update_forecast_progress(self, value):
        # Nilai 70-100 untuk simulasi trading
        progress = 70 + (value * 0.3)
        self.forecast_progress_bar.setValue(int(progress))
    
    def show_forecast_error(self, message):
        QMessageBox.critical(self, "Error Simulasi", message)
        self.forecast_run_button.setEnabled(True)
    
    def on_forecast_trading_finished(self, portfolio_values, trades, performance):
        # Update plot
        self.forecast_figure.clear()
        ax = self.forecast_figure.add_subplot(111)
        
        # Persiapkan data untuk date labels
        today = datetime.now()
        dates = [(today + timedelta(days=i)).strftime('%d-%m') for i in range(len(self.forecast_prices))]
        
        # Plot forecast prices
        ax.plot(dates, self.forecast_prices, label='Forecast Price', color='blue', alpha=0.7)
        
        # Plot buy/sell points dan portfolio value
        if portfolio_values:
            # Normalize portfolio values untuk plot di chart yang sama
            portfolio_scale = max(self.forecast_prices) / max(portfolio_values) if max(portfolio_values) > 0 else 1
            scaled_portfolio = [v * portfolio_scale for v in portfolio_values]
            
            ax2 = ax.twinx()
            ax2.plot(dates[:len(scaled_portfolio)], scaled_portfolio, 
                    label='Portfolio Value', color='green', linestyle='--')
            
            # Add buy/sell markers
            for trade in trades:
                day = trade['day']
                if day < len(dates):
                    if trade['type'] == 'BUY':
                        ax.plot(dates[day], self.forecast_prices[day], 'g^', markersize=10)
                    else:  # SELL
                        ax.plot(dates[day], self.forecast_prices[day], 'rv', markersize=10)
            
            # Set up legend and format
            ax.set_title(f'Simulasi Trading {self.forecast_predictor.ticker} - {len(self.forecast_prices)} Hari ke Depan')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga (Rp)')
            ax2.set_ylabel('Nilai Portfolio (Skala)')
            
            # Format x-axis for better readability
            if len(dates) > 20:
                ax.set_xticks(ax.get_xticks()[::3])  # Show every 3rd label
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
        self.forecast_canvas.draw()
        
        # Update metrics
        metrics_text = f"<b>Ringkasan Kinerja:</b><br>"
        metrics_text += f"Modal Awal: Rp {performance['initial_investment']:,.2f}<br>"
        metrics_text += f"Nilai Akhir: Rp {performance['final_value']:,.2f}<br>"
        metrics_text += f"Return Total: {performance['total_return']:.2f}%<br>"
        metrics_text += f"Drawdown Maksimum: {performance['max_drawdown']:.2f}%<br>"
        metrics_text += f"Win Rate: {performance['win_rate']:.2f}%<br>"
        metrics_text += f"Jumlah Transaksi: {performance['num_trades']}"
        
        self.forecast_metrics_label.setText(metrics_text)
        
        # Update trades table
        self.forecast_trades_table.setRowCount(len(trades))
        for i, trade in enumerate(trades):
            day_item = QTableWidgetItem(dates[trade['day']])
            
            type_item = QTableWidgetItem(trade['type'])
            type_item.setForeground(QColor('green' if trade['type'] == 'BUY' else 'red'))
            
            self.forecast_trades_table.setItem(i, 0, day_item)
            self.forecast_trades_table.setItem(i, 1, type_item)
            self.forecast_trades_table.setItem(i, 2, QTableWidgetItem(f"Rp {trade['price']:,.2f}"))
            self.forecast_trades_table.setItem(i, 3, QTableWidgetItem(f"Rp {trade['value']:,.2f}"))
        
        # Enable buttons
        self.forecast_run_button.setEnabled(True)
        self.forecast_save_button.setEnabled(True)
        
        # Save properties for later use
        self.forecast_portfolio_values = portfolio_values
        self.forecast_trades = trades
        self.forecast_performance = performance
        self.forecast_dates = dates
    
    def save_forecast_results(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Simpan Hasil Simulasi", "", "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        if not file_name:
            return
        
        try:
            if file_name.endswith('.xlsx'):
                # Create Excel with multiple sheets
                with pd.ExcelWriter(file_name) as writer:
                    # Forecast prices sheet
                    forecast_df = pd.DataFrame({
                        'Date': self.forecast_dates,
                        'Price': self.forecast_prices
                    })
                    forecast_df.to_excel(writer, sheet_name='Forecast Prices', index=False)
                    
                    # Trades sheet
                    if self.forecast_trades:
                        trades_data = []
                        for trade in self.forecast_trades:
                            trades_data.append({
                                'Date': self.forecast_dates[trade['day']],
                                'Type': trade['type'],
                                'Price': trade['price'],
                                'Shares': trade['shares'],
                                'Value': trade['value']
                            })
                        trades_df = pd.DataFrame(trades_data)
                        trades_df.to_excel(writer, sheet_name='Trades', index=False)
                    
                    # Portfolio values sheet
                    if self.forecast_portfolio_values:
                        portfolio_df = pd.DataFrame({
                            'Date': self.forecast_dates[:len(self.forecast_portfolio_values)],
                            'Portfolio Value': self.forecast_portfolio_values
                        })
                        portfolio_df.to_excel(writer, sheet_name='Portfolio Values', index=False)
                    
                    # Performance sheet
                    performance_data = {
                        'Metric': [
                            'Initial Investment', 
                            'Final Value', 
                            'Total Return (%)', 
                            'Max Drawdown (%)',
                            'Win Rate (%)',
                            'Number of Trades'
                        ],
                        'Value': [
                            self.forecast_performance['initial_investment'],
                            self.forecast_performance['final_value'],
                            self.forecast_performance['total_return'],
                            self.forecast_performance['max_drawdown'],
                            self.forecast_performance['win_rate'],
                            self.forecast_performance['num_trades']
                        ]
                    }
                    performance_df = pd.DataFrame(performance_data)
                    performance_df.to_excel(writer, sheet_name='Performance', index=False)
            else:
                # Create a single CSV with all data
                all_data = {
                    'Date': self.forecast_dates,
                    'Forecast Price': self.forecast_prices
                }
                
                # Add portfolio values if available
                if self.forecast_portfolio_values:
                    # Pad with NaN if needed
                    padded_values = self.forecast_portfolio_values + [np.nan] * (len(self.forecast_dates) - len(self.forecast_portfolio_values))
                    all_data['Portfolio Value'] = padded_values
                
                df = pd.DataFrame(all_data)
                df.to_csv(file_name, index=False)
            
            QMessageBox.information(self, "Sukses", "Hasil simulasi berhasil disimpan!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tidak dapat menyimpan hasil: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictionGUI()
    window.show()
    sys.exit(app.exec_()) 