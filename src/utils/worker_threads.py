"""
Worker Threads Module
====================

Modul ini berisi implementasi thread worker untuk operasi asinkron di UI aplikasi.
"""

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class WorkerThread(QThread):
    """Thread worker untuk menjalankan prediksi saham"""
    finished = pyqtSignal(object, object, object, object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, predictor):
        """
        Inisialisasi worker thread
        
        Parameters:
        -----------
        predictor : StockPredictor
            Objek predictor yang akan digunakan
        """
        super().__init__()
        self.predictor = predictor
        
    def run(self):
        """Jalankan proses prediksi"""
        try:
            # Prepare data
            self.progress.emit(10)
            if not self.predictor.prepare_data():
                self.error.emit("Error preparing data")
                return
                
            # Train model
            self.progress.emit(30)
            history = self.predictor.train_model()
            self.progress.emit(70)
            
            # Make predictions
            y_true, y_pred, forecast = self.predictor.predict()
            self.progress.emit(90)
            
            # Calculate metrics
            metrics = self.predictor.evaluate(y_true, y_pred)
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(y_true, y_pred, forecast, metrics)
        except Exception as e:
            self.error.emit(str(e))

class BacktestWorker(QThread):
    """Thread worker untuk menjalankan backtesting"""
    finished = pyqtSignal(object, object, object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, predictor, initial_investment, strategy):
        """
        Inisialisasi worker thread
        
        Parameters:
        -----------
        predictor : StockPredictor
            Objek predictor yang berisi data
        initial_investment : float
            Jumlah investasi awal
        strategy : str
            Nama strategi trading yang akan digunakan
        """
        super().__init__()
        self.predictor = predictor
        self.initial_investment = initial_investment
        self.strategy = strategy
        
    def run(self):
        """Jalankan proses backtesting"""
        try:
            # Extract required data
            self.progress.emit(10)
            actual_prices = self.predictor.scaler.inverse_transform(
                np.concatenate([
                    self.predictor.y.reshape(-1, 1),
                    np.zeros((len(self.predictor.y), self.predictor.X.shape[2]-1))
                ], axis=1)
            )[:, 0]
            
            y_pred = self.predictor.model.predict(self.predictor.X)
            predicted_prices = self.predictor.scaler.inverse_transform(
                np.concatenate([
                    y_pred,
                    np.zeros((len(y_pred), self.predictor.X.shape[2]-1))
                ], axis=1)
            )[:, 0]
            
            self.progress.emit(30)
            
            # Run backtest
            portfolio_values, trades, performance = self.run_backtest(
                actual_prices, predicted_prices, self.initial_investment, self.strategy
            )
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(portfolio_values, trades, performance)
        except Exception as e:
            self.error.emit(str(e))
    
    def run_backtest(self, actual_prices, predicted_prices, initial_investment, strategy):
        """Implementasi backtest"""
        # Inisialisasi portfolio
        cash = initial_investment
        shares = 0
        portfolio_values = []
        trades = []
        
        # Iterasi melalui harga historis
        for i in range(1, len(actual_prices)):
            # Hitung nilai portfolio saat ini
            portfolio_value = cash + shares * actual_prices[i]
            portfolio_values.append(portfolio_value)
            
            # Generate signal
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
        # Menghitung return total
        total_return = (final_value - initial_investment) / initial_investment * 100
        
        # Menghitung drawdown
        peak = portfolio_values[0] if portfolio_values else initial_investment
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown = max(drawdown, dd)
        
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
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return portfolio_values, trades, performance

    def generate_signal(self, predicted_prices, actual_prices, index, strategy):
        """Generate signal berdasarkan strategi"""
        from src.trading.strategies import TradingStrategy
        
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        return strategy_function(predicted_prices, actual_prices, index)

class StrategyOptimizerWorker(QThread):
    """Thread worker untuk menjalankan optimasi strategi"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, ticker, start_date, end_date, model, initial_investment, strategy=None,
                param_ranges=None):
        """
        Inisialisasi worker thread
        
        Parameters:
        -----------
        ticker : str
            Ticker saham
        start_date : str
            Tanggal mulai
        end_date : str
            Tanggal akhir
        model : str
            Nama model yang akan digunakan
        initial_investment : float
            Jumlah investasi awal
        strategy : str, optional
            Nama strategi trading yang akan dioptimasi
        param_ranges : dict, optional
            Range parameter untuk optimasi
        """
        super().__init__()
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model = model
        self.initial_investment = initial_investment
        self.strategy = strategy
        self.param_ranges = param_ranges
        
    def run(self):
        """Jalankan proses optimasi"""
        try:
            from src.models.predictor import StockPredictor
            
            # Create predictor
            self.progress.emit(5)
            predictor = StockPredictor(
                self.ticker, self.start_date, self.end_date, model_type=self.model
            )
            
            # Prepare data
            self.progress.emit(10)
            if not predictor.prepare_data():
                self.error.emit("Error preparing data")
                return
                
            # Train model
            self.progress.emit(20)
            predictor.train_model()
            self.progress.emit(40)
            
            # Get data for backtest
            actual_prices = predictor.scaler.inverse_transform(
                np.concatenate([
                    predictor.y.reshape(-1, 1),
                    np.zeros((len(predictor.y), predictor.X.shape[2]-1))
                ], axis=1)
            )[:, 0]
            
            y_pred = predictor.model.predict(predictor.X)
            predicted_prices = predictor.scaler.inverse_transform(
                np.concatenate([
                    y_pred,
                    np.zeros((len(y_pred), predictor.X.shape[2]-1))
                ], axis=1)
            )[:, 0]
            
            # Run optimization
            self.progress.emit(50)
            
            from src.trading.optimizer import StrategyOptimizer
            optimizer = StrategyOptimizer(actual_prices, predicted_prices, self.initial_investment)
            best_params, best_performance, _, _ = optimizer.optimize(self.strategy, self.param_ranges)
            
            # Run backtest with different parameters
            self.progress.emit(80)
            
            results = []
            total_combinations = 1
            current_combination = 0
            
            # Dapatkan total kombinasi parameter
            for param_values in self.param_ranges.values():
                total_combinations *= len(param_values)
            
            progress_increment = 20 / total_combinations  # 20% untuk pengujian parameter
            
            # Uji setiap kombinasi parameter
            import itertools
            param_names = list(self.param_ranges.keys())
            param_values = [self.param_ranges[name] for name in param_names]
            
            for param_combo in itertools.product(*param_values):
                params = dict(zip(param_names, param_combo))
                
                # Jalankan backtest dengan parameter ini
                portfolio_values, trades, performance = self.run_backtest_with_params(
                    actual_prices, predicted_prices, self.initial_investment, self.strategy, params
                )
                
                results.append({
                    'strategy': self.strategy,
                    'params': params,
                    'performance': performance
                })
                
                current_combination += 1
                self.progress.emit(80 + int(current_combination * progress_increment))
            
            # Sort results by total_return
            results.sort(key=lambda x: x['performance']['total_return'], reverse=True)
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
            
    def run_backtest_with_params(self, actual_prices, predicted_prices, initial_investment, strategy, params):
        """Implementasi backtest dengan parameter tertentu"""
        # Inisialisasi portfolio
        cash = initial_investment
        shares = 0
        portfolio_values = []
        trades = []
        
        # Iterasi melalui harga historis
        for i in range(1, len(actual_prices)):
            # Hitung nilai portfolio saat ini
            portfolio_value = cash + shares * actual_prices[i]
            portfolio_values.append(portfolio_value)
            
            # Generate signal dengan parameter
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
        # Menghitung return total
        total_return = (final_value - initial_investment) / initial_investment * 100
        
        # Menghitung drawdown
        peak = portfolio_values[0] if portfolio_values else initial_investment
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown = max(drawdown, dd)
        
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
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return portfolio_values, trades, performance

    def generate_signal_with_params(self, predicted_prices, actual_prices, index, strategy, params):
        """Generate signal berdasarkan strategi dengan parameter tertentu"""
        from src.trading.strategies import TradingStrategy
        
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        return strategy_function(predicted_prices, actual_prices, index, params)

class ForecastTradeWorker(QThread):
    """Thread worker untuk menjalankan simulasi trading pada data forecast"""
    finished = pyqtSignal(object, object, object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, forecast_prices, initial_investment, strategy, params):
        """
        Inisialisasi worker thread
        
        Parameters:
        -----------
        forecast_prices : array-like
            Array harga forecast
        initial_investment : float
            Jumlah investasi awal
        strategy : str
            Nama strategi trading yang akan digunakan
        params : dict
            Parameter untuk strategi
        """
        super().__init__()
        self.forecast_prices = forecast_prices
        self.initial_investment = initial_investment
        self.strategy = strategy
        self.params = params
        
    def run(self):
        """Jalankan proses simulasi trading"""
        try:
            # Simulasi trading dengan forecast
            self.progress.emit(20)
            
            portfolio_values, trades, performance = self.simulate_forecast_trading(
                self.forecast_prices, self.initial_investment, self.strategy, self.params
            )
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(portfolio_values, trades, performance)
        except Exception as e:
            self.error.emit(str(e))
    
    def simulate_forecast_trading(self, forecast_prices, initial_investment, strategy, params):
        """Implementasi simulasi trading pada data forecast"""
        # Inisialisasi portfolio
        cash = initial_investment
        shares = 0
        portfolio_values = []
        trades = []
        
        from src.trading.strategies import TradingStrategy
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        
        # Karena kita hanya memiliki data forecast, kita akan menggunakan sinyal prediktif
        # yang langsung dibandingkan dengan harga berikutnya
        for i in range(len(forecast_prices) - 1):
            # Hitung nilai portfolio saat ini
            portfolio_value = cash + shares * forecast_prices[i]
            portfolio_values.append(portfolio_value)
            
            # Generate signal
            current_price = forecast_prices[i]
            next_price = forecast_prices[i + 1]
            
            # Implementasi sederhana: anggap next_price sebagai "prediksi"
            signal = 'HOLD'
            
            if strategy == 'trend_following':
                threshold = params.get('threshold', 0.01)
                if next_price > current_price * (1 + threshold):
                    signal = 'BUY'
                elif next_price < current_price * (1 - threshold):
                    signal = 'SELL'
            
            elif strategy == 'mean_reversion':
                # Untuk mean reversion, kita perlu membangun rata-rata bergerak
                # dari data forecast sejauh ini
                window = params.get('window', 10)
                buy_threshold = params.get('buy_threshold', 0.05)
                sell_threshold = params.get('sell_threshold', 0.05)
                
                if i + 1 >= window:
                    moving_avg = np.mean(forecast_prices[i+1-window:i+1])
                    if current_price < moving_avg * (1 - buy_threshold):
                        signal = 'BUY'  # Beli ketika harga di bawah rata-rata
                    elif current_price > moving_avg * (1 + sell_threshold):
                        signal = 'SELL'  # Jual ketika harga di atas rata-rata
            
            elif strategy == 'predictive':
                buy_threshold = params.get('buy_threshold', 1.01)
                sell_threshold = params.get('sell_threshold', 0.99)
                
                if next_price > current_price * buy_threshold:
                    signal = 'BUY'
                elif next_price < current_price * sell_threshold:
                    signal = 'SELL'
            
            # Proses signals
            if signal == 'BUY' and cash > 0:
                # Beli saham sebanyak mungkin dengan uang yang ada
                shares_to_buy = cash / current_price
                shares += shares_to_buy
                cash = 0
                trades.append({
                    'day': i,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'value': shares_to_buy * current_price
                })
            elif signal == 'SELL' and shares > 0:
                # Jual semua saham
                cash += shares * current_price
                trades.append({
                    'day': i,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'value': shares * current_price
                })
                shares = 0
        
        # Tambahkan nilai akhir portfolio
        if len(forecast_prices) > 0:
            portfolio_values.append(cash + shares * forecast_prices[-1])
        
        # Nilai akhir portfolio
        final_value = cash + (shares * forecast_prices[-1] if len(forecast_prices) > 0 else 0)
        
        # Menghitung metrik performa
        # Menghitung return total
        total_return = (final_value - initial_investment) / initial_investment * 100
        
        # Menghitung drawdown
        peak = portfolio_values[0] if portfolio_values else initial_investment
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown = max(drawdown, dd)
        
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
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return portfolio_values, trades, performance 