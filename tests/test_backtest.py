import pytest
import numpy as np
import pandas as pd
from src.trading.backtest import Backtester

def test_backtester_initialization():
    # Buat data simulasi
    prices = np.array([100, 102, 105, 103, 104, 107, 108, 109])
    predictions = np.array([101, 103, 104, 102, 105, 108, 110, 111])
    
    # Inisialisasi backtester
    backtester = Backtester(
        actual_prices=prices,
        predicted_prices=predictions,
        initial_investment=10000
    )
    
    assert np.array_equal(backtester.actual_prices, prices)
    assert np.array_equal(backtester.predicted_prices, predictions)
    assert backtester.initial_investment == 10000

def test_backtester_run():
    # Buat data simulasi
    prices = np.array([100, 102, 105, 103, 104, 107, 108, 109])
    predictions = np.array([101, 103, 104, 102, 105, 108, 110, 111])
    
    # Inisialisasi backtester
    backtester = Backtester(
        actual_prices=prices,
        predicted_prices=predictions,
        initial_investment=10000
    )
    
    # Jalankan backtest
    portfolio_values, trades, metrics = backtester.run('trend_following', params={'threshold': 0.01})
    
    # Periksa hasil
    assert isinstance(portfolio_values, list)
    assert isinstance(trades, list)
    assert isinstance(metrics, dict)
    assert len(portfolio_values) == len(prices) - 1  # Karena iterasi mulai dari indeks 1
    assert metrics['final_value'] > 0
    
def test_backtester_with_different_strategies():
    # Buat data simulasi
    prices = np.array([100, 102, 105, 103, 104, 107, 108, 109, 110, 112, 115])
    predictions = np.array([101, 103, 104, 102, 105, 108, 110, 111, 112, 114, 116])
    
    # Uji dengan strategi trend following
    trend_backtester = Backtester(
        actual_prices=prices,
        predicted_prices=predictions,
        initial_investment=10000
    )
    _, _, trend_metrics = trend_backtester.run('trend_following')
    
    # Uji dengan strategi mean reversion
    mean_backtester = Backtester(
        actual_prices=prices,
        predicted_prices=predictions,
        initial_investment=10000
    )
    _, _, mean_metrics = mean_backtester.run('mean_reversion', params={'window': 5, 'buy_threshold': 0.02, 'sell_threshold': 0.02})
    
    # Uji dengan strategi predictive
    predictive_backtester = Backtester(
        actual_prices=prices,
        predicted_prices=predictions,
        initial_investment=10000
    )
    _, _, predictive_metrics = predictive_backtester.run('predictive')
    
    # Periksa bahwa hasil berbeda berdasarkan strategi
    assert isinstance(trend_metrics, dict)
    assert isinstance(mean_metrics, dict)
    assert isinstance(predictive_metrics, dict)
    
    # Cek apakah nilai akhir dihitung
    assert 'final_value' in trend_metrics
    assert 'final_value' in mean_metrics
    assert 'final_value' in predictive_metrics
    
    # Cek bahwa strategi berbeda berpotensi menghasilkan hasil yang berbeda
    # Catatan: ini bukan pengujian deterministik, strategi yang berbeda mungkin menghasilkan nilai yang sama
    assert trend_metrics != mean_metrics or trend_metrics != predictive_metrics or mean_metrics != predictive_metrics

def test_calculate_performance_metrics():
    # Buat data simulasi
    prices = np.array([100, 102, 105, 103, 104, 107, 108, 109])
    predictions = np.array([101, 103, 104, 102, 105, 108, 110, 111])
    
    # Inisialisasi backtester
    backtester = Backtester(
        actual_prices=prices,
        predicted_prices=predictions,
        initial_investment=10000
    )
    
    # Buat data portfolio simulasi
    portfolio_values = [10000, 10200, 10400, 10300, 10500]
    trades = [
        {'day': 1, 'type': 'BUY', 'price': 102, 'shares': 98.0392, 'value': 10000},
        {'day': 3, 'type': 'SELL', 'price': 103, 'shares': 98.0392, 'value': 10097.6396}
    ]
    final_value = 10500
    
    # Hitung metrik
    metrics = backtester.calculate_performance_metrics(portfolio_values, trades, final_value)
    
    # Periksa metrik
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics
    assert 'max_drawdown' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'win_rate' in metrics
    assert metrics['initial_investment'] == 10000
    assert metrics['final_value'] == 10500
    assert metrics['total_return'] == 5.0  # (10500 - 10000) / 10000 * 100 