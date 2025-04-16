import numpy as np
from src.data.indicators import TechnicalIndicators

def test_rsi_output_shape():
    data = np.random.rand(100) * 100
    rsi = TechnicalIndicators.calculate_rsi(data)
    assert len(rsi) == len(data)
