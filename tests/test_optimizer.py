import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.trading.optimizer import StrategyOptimizer
from src.trading.strategies import TradingStrategy

def test_strategy_optimizer_init():
    # Data pengujian
    actual_prices = np.array([100, 101, 102, 103, 104, 105])
    predicted_prices = np.array([101, 102, 103, 104, 105, 106])
    
    # Inisialisasi optimizer
    optimizer = StrategyOptimizer(
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        initial_investment=10000
    )
    
    # Verifikasi atribut
    assert np.array_equal(optimizer.actual_prices, actual_prices)
    assert np.array_equal(optimizer.predicted_prices, predicted_prices)
    assert optimizer.initial_investment == 10000

def test_set_param_grid():
    # Data pengujian
    actual_prices = np.array([100, 101, 102, 103, 104, 105])
    predicted_prices = np.array([101, 102, 103, 104, 105, 106])
    
    # Inisialisasi optimizer
    optimizer = StrategyOptimizer(
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        initial_investment=10000
    )
    
    # Menguji metode _generate_parameter_combinations
    param_ranges = {
        'threshold': [0.01, 0.02, 0.03]
    }
    combinations = optimizer._generate_parameter_combinations(param_ranges)
    
    # Verifikasi kombinasi parameter dihasilkan dengan benar
    assert len(combinations) == 3
    assert all('threshold' in params for params in combinations)

@patch('src.trading.backtest.Backtester')
def test_optimize(mock_backtester):
    # Setup mock backtester
    mock_instance = MagicMock()
    mock_backtester.return_value = mock_instance
    
    # Setup mock result untuk run backtest
    mock_instance.run.return_value = (
        [10000, 10100, 10200], 
        [], 
        {'total_return': 2.0, 'final_value': 10200}
    )
    
    # Data pengujian
    actual_prices = np.array([100, 101, 102, 103, 104, 105])
    predicted_prices = np.array([101, 102, 103, 104, 105, 106])
    
    # Inisialisasi optimizer
    optimizer = StrategyOptimizer(
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        initial_investment=10000
    )
    
    # Jalankan optimisasi
    param_ranges = {'threshold': [0.01, 0.02]}
    best_params, best_performance, _, _ = optimizer.optimize('trend_following', param_ranges)
    
    # Verifikasi
    assert isinstance(best_params, dict)
    assert 'threshold' in best_params
    assert isinstance(best_performance, dict)
    assert 'total_return' in best_performance
    
@patch('src.trading.backtest.Backtester')
def test_run_backtest_with_params(mock_backtester):
    # Setup mock
    mock_instance = MagicMock()
    mock_backtester.return_value = mock_instance
    
    # Setup mock result
    mock_instance.run.return_value = (
        [10000, 10100, 10200], 
        [], 
        {'total_return': 2.0, 'final_value': 10200}
    )
    
    # Data pengujian
    actual_prices = np.array([100, 101, 102, 103, 104, 105])
    predicted_prices = np.array([101, 102, 103, 104, 105, 106])
    
    # Inisialisasi optimizer
    optimizer = StrategyOptimizer(
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        initial_investment=10000
    )
    
    # Mendapatkan fungsi strategi
    strategy_function = TradingStrategy.get_strategy_function('trend_following')
    
    # Jalankan backtest dengan parameter tertentu
    params = {'threshold': 0.02}
    portfolio_values, trades, performance = optimizer.run_backtest(strategy_function, params)
    
    # Verifikasi
    assert isinstance(portfolio_values, list)
    assert isinstance(trades, list)
    assert isinstance(performance, dict) 