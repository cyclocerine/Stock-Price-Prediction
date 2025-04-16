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
        strategy_name="trend_following",
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        initial_investment=10000
    )
    
    # Verifikasi atribut
    assert optimizer.strategy_name == "trend_following"
    assert np.array_equal(optimizer.actual_prices, actual_prices)
    assert np.array_equal(optimizer.predicted_prices, predicted_prices)
    assert optimizer.initial_investment == 10000
    assert optimizer.strategy_function == TradingStrategy.trend_following
    assert isinstance(optimizer.param_grid, dict)

def test_set_param_grid():
    # Data pengujian
    actual_prices = np.array([100, 101, 102, 103, 104, 105])
    predicted_prices = np.array([101, 102, 103, 104, 105, 106])
    
    # Inisialisasi optimizer
    optimizer = StrategyOptimizer(
        strategy_name="trend_following",
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        initial_investment=10000
    )
    
    # Mengatur parameter grid kustom
    custom_param_grid = {
        'threshold': [0.01, 0.02, 0.03]
    }
    optimizer.set_param_grid(custom_param_grid)
    
    # Verifikasi param_grid diperbarui
    assert optimizer.param_grid == custom_param_grid

@patch('src.trading.optimizer.Backtester')
def test_optimize(mock_backtester):
    # Setup mock backtester
    mock_instance = MagicMock()
    mock_backtester.return_value = mock_instance
    
    # Setup mock result untuk run backtest
    mock_instance.run.return_value = (
        np.array([1, 2, 3]), 
        np.array([100, 110, 120]), 
        np.array([1.0, 1.1, 1.2])
    )
    
    # Setup mock untuk performance metrics
    mock_instance.calculate_performance_metrics.return_value = {
        'total_return': 0.2,
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.05
    }
    
    # Data pengujian
    actual_prices = np.array([100, 101, 102, 103, 104, 105])
    predicted_prices = np.array([101, 102, 103, 104, 105, 106])
    
    # Inisialisasi optimizer dengan param_grid sederhana
    optimizer = StrategyOptimizer(
        strategy_name="trend_following",
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        initial_investment=10000
    )
    optimizer.set_param_grid({'threshold': [0.01, 0.02]})
    
    # Jalankan optimisasi
    best_params, results_df = optimizer.optimize(metric='sharpe_ratio')
    
    # Verifikasi
    assert isinstance(best_params, dict)
    assert isinstance(results_df, pd.DataFrame)
    assert 'params' in results_df.columns
    assert 'sharpe_ratio' in results_df.columns
    
    # Verifikasi bahwa backtester dipanggil untuk setiap kombinasi parameter
    assert mock_backtester.call_count >= 2  # Minimal dipanggil untuk setiap nilai threshold
    
@patch('src.trading.optimizer.Backtester')
def test_run_backtest_with_params(mock_backtester):
    # Setup mock
    mock_instance = MagicMock()
    mock_backtester.return_value = mock_instance
    
    # Setup mock result
    mock_instance.run.return_value = (
        np.array([1, 2, 3]), 
        np.array([100, 110, 120]), 
        np.array([1.0, 1.1, 1.2])
    )
    
    # Setup mock untuk performance metrics
    mock_instance.calculate_performance_metrics.return_value = {
        'total_return': 0.2,
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.05
    }
    
    # Data pengujian
    actual_prices = np.array([100, 101, 102, 103, 104, 105])
    predicted_prices = np.array([101, 102, 103, 104, 105, 106])
    
    # Inisialisasi optimizer
    optimizer = StrategyOptimizer(
        strategy_name="trend_following",
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        initial_investment=10000
    )
    
    # Jalankan backtest dengan parameter tertentu
    params = {'threshold': 0.02}
    metrics = optimizer.run_backtest_with_params(params)
    
    # Verifikasi
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    
    # Verifikasi backtester dipanggil dengan benar
    mock_backtester.assert_called_with(
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        strategy_function=TradingStrategy.trend_following,
        strategy_params=params,
        initial_investment=10000
    ) 