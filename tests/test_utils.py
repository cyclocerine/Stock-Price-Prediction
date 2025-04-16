import numpy as np
from src.utils.common import TechnicalIndicators

def test_rsi_output_shape():
    data = np.random.rand(100) * 100
    rsi = TechnicalIndicators.calculate_rsi(data)
    assert len(rsi) == len(data)
