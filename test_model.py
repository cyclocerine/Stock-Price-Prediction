from main import download_and_preprocess_data, create_dataset
from sklearn.preprocessing import MinMaxScaler

def test_download_data():
    data = download_and_preprocess_data('^JKSE', '2015-01-01', '2025-03-25')
    assert data is not None
    assert not data.empty
    assert 'SMA' in data.columns
    assert 'RSI' in data.columns

def test_create_dataset():
    data = download_and_preprocess_data('^JKSE', '2015-01-01', '2025-03-25')
    scaler = MinMaxScaler(feature_range=(0, 1))
    X, y, scaled_data = create_dataset(data, 60, scaler)
    assert X.shape[0] == len(y)
    assert X.shape[1] == 60  # sesuai dengan lookback
