import pytest
import pandas as pd
from main import DataPreprocessor

def test_data_download():
    dp = DataPreprocessor("AAPL", "2022-01-01", "2022-12-31")
    success = dp.download_data()
    assert success
    assert isinstance(dp.data, pd.DataFrame)
    assert not dp.data.empty

def test_calculate_indicators():
    dp = DataPreprocessor("AAPL", "2022-01-01", "2022-12-31")
    assert dp.download_data()
    success = dp.calculate_indicators()
    assert success
    assert dp.features is not None
