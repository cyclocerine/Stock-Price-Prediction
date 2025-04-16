"""
GUI Application
=============

Modul ini berisi kelas utama aplikasi GUI.
"""

from PyQt5.QtWidgets import QMainWindow, QTabWidget
from PyQt5.QtCore import Qt

from .prediction_tab import PredictionTab
from .backtest_tab import BacktestTab
from .optimizer_tab import OptimizerTab
from .forecast_tab import ForecastTab

class StockPredictionApp(QMainWindow):
    def __init__(self):
        """
        Inisialisasi aplikasi utama
        """
        super().__init__()
        self.setWindowTitle("Stock Price Prediction")
        self.setGeometry(100, 100, 1200, 800)
        
        # Tab widget untuk memisahkan prediksi, backtesting, dan optimizer
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Tab untuk prediksi
        self.prediction_tab = PredictionTab()
        self.tabs.addTab(self.prediction_tab, "Prediksi")
        
        # Tab untuk backtesting
        self.backtest_tab = BacktestTab()
        self.tabs.addTab(self.backtest_tab, "Backtesting")
        
        # Tab untuk optimizer
        self.optimizer_tab = OptimizerTab()
        self.tabs.addTab(self.optimizer_tab, "Optimizer Strategi")
        
        # Tab untuk simulasi trading pada forecast
        self.forecast_tab = ForecastTab()
        self.tabs.addTab(self.forecast_tab, "Trading Prediksi")
        
        # Hubungkan tab prediction dengan backtest dan forecast
        self.prediction_tab.prediction_finished.connect(self._handle_prediction_finished)
        
    def _handle_prediction_finished(self, predictor, y_true, y_pred, forecast):
        """
        Menangani event setelah prediksi selesai
        
        Parameters:
        -----------
        predictor : StockPredictor
            Objek prediktor yang menghasilkan prediksi
        y_true : array-like
            Array harga aktual historis
        y_pred : array-like
            Array harga prediksi dari model
        forecast : array-like
            Array harga forecast ke depan
        """
        # Kirim hasil ke tab backtest
        self.backtest_tab.set_prediction_results(predictor, y_true, y_pred)
        
        # Kirim hasil ke tab forecast
        self.forecast_tab.set_forecast_results(predictor, forecast) 