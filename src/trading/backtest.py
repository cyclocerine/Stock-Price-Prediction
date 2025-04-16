"""
Backtesting Module
================

Modul ini berisi implementasi kelas Backtester untuk
menguji strategi trading pada data historis.
"""

import numpy as np
from .strategies import TradingStrategy

class Backtester:
    def __init__(self, actual_prices, predicted_prices, initial_investment=10000):
        """
        Inisialisasi Backtester
        
        Parameters:
        -----------
        actual_prices : array-like
            Array harga aktual historis
        predicted_prices : array-like
            Array harga prediksi dari model
        initial_investment : float, optional
            Jumlah investasi awal, default 10000
        """
        self.actual_prices = actual_prices
        self.predicted_prices = predicted_prices
        self.initial_investment = initial_investment
        
    def run(self, strategy, params=None):
        """
        Menjalankan backtesting untuk strategi tertentu
        
        Parameters:
        -----------
        strategy : str
            Nama strategi trading yang akan digunakan
        params : dict, optional
            Parameter tambahan untuk strategi
            
        Returns:
        --------
        tuple
            (portfolio_values, trades, metrics_performance)
            - portfolio_values: nilai portfolio per hari
            - trades: list transaksi yang dilakukan
            - metrics_performance: metrik performa strategi
        """
        # Inisialisasi portfolio
        cash = self.initial_investment
        shares = 0
        portfolio_values = []
        trades = []
        
        # Memastikan data memiliki panjang yang sama
        length = min(len(self.actual_prices), len(self.predicted_prices))
        actual_prices = self.actual_prices[:length]
        predicted_prices = self.predicted_prices[:length]
        
        # Dapatkan fungsi strategi
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        
        # Iterasi melalui harga historis
        for i in range(1, length):
            # Hitung nilai portfolio saat ini
            portfolio_value = cash + shares * actual_prices[i]
            portfolio_values.append(portfolio_value)
            
            signal = strategy_function(predicted_prices, actual_prices, i, params)
            
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
        
        # Tambahkan nilai akhir portfolio jika belum ada
        if len(portfolio_values) < length - 1:
            final_value = cash + shares * actual_prices[-1]
            portfolio_values.append(final_value)
        
        # Nilai akhir portfolio
        final_value = cash + shares * actual_prices[-1]
        
        # Menghitung metrik performa
        metrics = self.calculate_performance_metrics(portfolio_values, trades, final_value)
        
        return portfolio_values, trades, metrics
    
    def calculate_performance_metrics(self, portfolio_values, trades, final_value):
        """
        Menghitung metrik performa untuk hasil backtest
        
        Parameters:
        -----------
        portfolio_values : array-like
            Nilai portfolio per hari
        trades : list
            List transaksi yang dilakukan
        final_value : float
            Nilai akhir portfolio
            
        Returns:
        --------
        dict
            Dictionary berisi metrik performa
        """
        # Menghitung return total
        total_return = (final_value - self.initial_investment) / self.initial_investment * 100
        
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
            'initial_investment': self.initial_investment,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return performance 