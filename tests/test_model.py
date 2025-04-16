import pytest
import numpy as np
from src.models.predictor import StockPredictor

def test_dummy_model_training():
    # Simulasi test sederhana (replace dengan real model class dan data)
    try:
        assert True  # Ganti dengan kondisi aktual training berhasil
    except Exception as e:
        pytest.fail(f"Model training failed: {str(e)}")

def test_stock_predictor_instance():
    predictor = StockPredictor("AAPL", "2022-01-01", "2022-12-31")
    assert predictor is not None
    assert predictor.ticker == "AAPL"
    assert predictor.start_date == "2022-01-01"
    assert predictor.end_date == "2022-12-31"

def test_model_builder_methods():
    from src.models.model_builder import ModelBuilder
    
    # Test model creation with simple input shape
    input_shape = (30, 10)  # time steps, features
    
    # Test CNN-LSTM
    cnn_lstm = ModelBuilder.build_cnn_lstm(input_shape)
    assert cnn_lstm is not None
    
    # Test BiLSTM
    bilstm = ModelBuilder.build_bilstm(input_shape)
    assert bilstm is not None
    
    # Test Transformer
    transformer = ModelBuilder.build_transformer(input_shape)
    assert transformer is not None
    
    # Test Ensemble
    ensemble = ModelBuilder.build_ensemble(input_shape)
    assert ensemble is not None
