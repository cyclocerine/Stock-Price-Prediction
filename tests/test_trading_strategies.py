import pytest
import numpy as np
from src.trading.strategies import TradingStrategy

def test_trend_following_strategy():
    # Membuat data simulasi
    predicted_prices = np.array([100, 105, 110, 108, 106, 112, 115])
    actual_prices = np.array([102, 103, 108, 109, 107, 110, 114])
    
    # Menguji sinyal "BUY" (prediksi > aktual dengan threshold)
    signal = TradingStrategy.trend_following(predicted_prices, actual_prices, 2, {'threshold': 0.01})
    assert signal == 'BUY'
    
    # Menguji sinyal "SELL" (prediksi < aktual dengan threshold)
    signal = TradingStrategy.trend_following(predicted_prices, actual_prices, 3, {'threshold': 0.01})
    assert signal == 'SELL'
    
    # Menguji sinyal "HOLD" (perbedaan dalam threshold)
    signal = TradingStrategy.trend_following(predicted_prices, actual_prices, 4, {'threshold': 0.05})
    assert signal == 'HOLD'

def test_mean_reversion_strategy():
    # Membuat data simulasi
    predicted_prices = np.array([100, 105, 110, 108, 106, 102, 95])
    actual_prices = np.array([102, 104, 107, 110, 115, 112, 108])
    
    # Menguji sinyal "BUY" (harga di bawah moving average)
    signal = TradingStrategy.mean_reversion(
        predicted_prices, 
        actual_prices, 
        5, 
        {'window': 3, 'buy_threshold': 0.98, 'sell_threshold': 1.02}
    )
    assert signal == 'SELL'
    
    # Menguji sinyal "SELL" (harga di atas moving average)
    actual_prices_2 = np.array([102, 104, 107, 110, 105, 100, 98])
    signal = TradingStrategy.mean_reversion(
        predicted_prices, 
        actual_prices_2, 
        5, 
        {'window': 3, 'buy_threshold': 0.98, 'sell_threshold': 1.02}
    )
    assert signal == 'BUY'
    
    # Menguji sinyal "HOLD" (harga dalam threshold)
    actual_prices_3 = np.array([102, 104, 107, 110, 108, 109, 110])
    signal = TradingStrategy.mean_reversion(
        predicted_prices, 
        actual_prices_3, 
        5, 
        {'window': 3, 'buy_threshold': 0.90, 'sell_threshold': 1.10}
    )
    assert signal == 'HOLD'

def test_predictive_strategy():
    # Membuat data simulasi
    predicted_prices = np.array([100, 105, 110, 115, 120, 125, 130])
    actual_prices = np.array([102, 104, 106, 108, 110, 112, 114])
    
    # Menguji sinyal "BUY" (prediksi > aktual * buy_threshold)
    signal = TradingStrategy.predictive(predicted_prices, actual_prices, 3, {'buy_threshold': 1.01, 'sell_threshold': 0.99})
    assert signal == 'BUY'
    
    # Menguji sinyal "SELL" (prediksi < aktual * sell_threshold)
    predicted_prices_2 = np.array([100, 105, 110, 105, 100, 95, 90])
    signal = TradingStrategy.predictive(predicted_prices_2, actual_prices, 5, {'buy_threshold': 1.01, 'sell_threshold': 0.99})
    assert signal == 'SELL'
    
    # Menguji sinyal "HOLD" (prediksi dalam threshold)
    predicted_prices_3 = np.array([100, 104, 106, 109, 111, 113, 115])
    signal = TradingStrategy.predictive(predicted_prices_3, actual_prices, 5, {'buy_threshold': 1.05, 'sell_threshold': 0.95})
    assert signal == 'HOLD'

def test_get_strategy_function():
    # Menguji mendapatkan fungsi strategi yang valid
    strategy_function = TradingStrategy.get_strategy_function('trend_following')
    assert callable(strategy_function)
    assert strategy_function == TradingStrategy.trend_following
    
    strategy_function = TradingStrategy.get_strategy_function('mean_reversion')
    assert callable(strategy_function)
    assert strategy_function == TradingStrategy.mean_reversion
    
    strategy_function = TradingStrategy.get_strategy_function('predictive')
    assert callable(strategy_function)
    assert strategy_function == TradingStrategy.predictive
    
    # Menguji ValueError untuk strategi yang tidak valid
    with pytest.raises(ValueError):
        TradingStrategy.get_strategy_function('strategi_tidak_ada') 