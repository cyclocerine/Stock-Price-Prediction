#!/usr/bin/env python
"""
CLI Application
=============

Script untuk menjalankan aplikasi prediksi saham dengan antarmuka command line.
"""

import argparse
from datetime import datetime
import sys
import os

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predictor import StockPredictor

def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--ticker', type=str, default='ADRO.JK', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--lookback', type=int, default=60, help='Number of days to look back')
    parser.add_argument('--forecast_days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--model', type=str, default='ensemble', choices=['cnn_lstm', 'bilstm', 'transformer', 'ensemble'], help='Model type')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    predictor = StockPredictor(
        args.ticker,
        args.start_date,
        args.end_date,
        args.lookback,
        args.forecast_days,
        args.model,
        args.tune
    )
    
    if not predictor.prepare_data():
        print("Error preparing data")
        return
        
    print(f"\nTraining {args.model} model...")
    history = predictor.train_model()
    
    print("\nMaking predictions...")
    y_true, y_pred, forecast = predictor.predict()
    
    print("\nEvaluating model...")
    metrics = predictor.evaluate(y_true, y_pred)
    
    print("\nPlotting results...")
    predictor.plot_results(y_true, y_pred, forecast)
    
    print("\nForecast for next days:")
    for i, price in enumerate(forecast, 1):
        print(f"Day {i}: {price:.2f}")

if __name__ == "__main__":
    main() 