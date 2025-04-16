"""
Prediction Tab Module
====================

Modul ini berisi implementasi tab prediksi untuk UI aplikasi.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QDateEdit, QComboBox, QSpinBox, QCheckBox, QPushButton, 
                             QProgressBar, QMessageBox)
from PyQt5.QtCore import QDate, pyqtSignal, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from src.models.predictor import StockPredictor
from ..utils.worker_threads import WorkerThread

class PredictionTab(QWidget):
    # Signal untuk memberitahu tab lain bahwa prediksi selesai
    prediction_finished = pyqtSignal(object, object, object, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = None
        self.setup_ui()
        
    def setup_ui(self):
        # Widget utama
        layout = QVBoxLayout(self)
        
        # Input parameters
        input_layout = QHBoxLayout()
        
        # Ticker input
        ticker_layout = QVBoxLayout()
        ticker_label = QLabel("Ticker Symbol:")
        self.ticker_input = QLineEdit("ADRO.JK")
        ticker_layout.addWidget(ticker_label)
        ticker_layout.addWidget(self.ticker_input)
        input_layout.addLayout(ticker_layout)
        
        # Date inputs
        date_layout = QVBoxLayout()
        start_date_label = QLabel("Start Date:")
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate(2020, 1, 1))
        end_date_label = QLabel("End Date:")
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        date_layout.addWidget(start_date_label)
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(end_date_label)
        date_layout.addWidget(self.end_date)
        input_layout.addLayout(date_layout)
        
        # Model selection
        model_layout = QVBoxLayout()
        model_label = QLabel("Model Type:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['cnn_lstm', 'bilstm', 'transformer', 'ensemble'])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        input_layout.addLayout(model_layout)
        
        # Parameters
        param_layout = QVBoxLayout()
        lookback_label = QLabel("Lookback Days:")
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(10, 200)
        self.lookback_spin.setValue(60)
        forecast_label = QLabel("Forecast Days:")
        self.forecast_spin = QSpinBox()
        self.forecast_spin.setRange(1, 100)
        self.forecast_spin.setValue(20)
        param_layout.addWidget(lookback_label)
        param_layout.addWidget(self.lookback_spin)
        param_layout.addWidget(forecast_label)
        param_layout.addWidget(self.forecast_spin)
        input_layout.addLayout(param_layout)
        
        # Hyperparameter tuning
        tune_layout = QVBoxLayout()
        self.tune_checkbox = QCheckBox("Enable Hyperparameter Tuning")
        tune_layout.addWidget(self.tune_checkbox)
        input_layout.addLayout(tune_layout)
        
        layout.addLayout(input_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Run button
        self.run_button = QPushButton("Run Prediction")
        self.run_button.clicked.connect(self.run_prediction)
        layout.addWidget(self.run_button)
        
        # Plot area
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Results area
        results_layout = QHBoxLayout()
        
        # Metrics
        metrics_layout = QVBoxLayout()
        metrics_label = QLabel("Model Metrics:")
        self.metrics_text = QLabel()
        self.metrics_text.setWordWrap(True)
        metrics_layout.addWidget(metrics_label)
        metrics_layout.addWidget(self.metrics_text)
        results_layout.addLayout(metrics_layout)
        
        # Forecast
        forecast_layout = QVBoxLayout()
        forecast_label = QLabel("Forecast Results:")
        self.forecast_text = QLabel()
        self.forecast_text.setWordWrap(True)
        forecast_layout.addWidget(forecast_label)
        forecast_layout.addWidget(self.forecast_text)
        results_layout.addLayout(forecast_layout)
        
        layout.addLayout(results_layout)
        
        # Save button
        save_layout = QHBoxLayout()
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_plot)
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self.save_results)
        save_layout.addWidget(self.save_plot_button)
        save_layout.addWidget(self.save_results_button)
        layout.addLayout(save_layout)
        
        # Disable buttons initially
        self.save_plot_button.setEnabled(False)
        self.save_results_button.setEnabled(False)
    
    def run_prediction(self):
        # Disable run button during prediction
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Get parameters
        ticker = self.ticker_input.text()
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")
        model_type = self.model_combo.currentText()
        lookback = self.lookback_spin.value()
        forecast_days = self.forecast_spin.value()
        tune = self.tune_checkbox.isChecked()
        
        # Create predictor
        self.predictor = StockPredictor(
            ticker, start_date, end_date,
            lookback, forecast_days, model_type, tune
        )
        
        # Create and start worker thread
        self.worker = WorkerThread(self.predictor)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.run_button.setEnabled(True)
    
    def on_prediction_finished(self, y_true, y_pred, forecast, metrics):
        # Update plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(y_true, label='Actual', color='blue')
        ax.plot(y_pred, label='Predicted', color='red', linestyle='--')
        ax.plot(range(len(y_true), len(y_true) + len(forecast)), 
                forecast, label='Forecast', color='green', linestyle='-.')
        ax.set_title(f'{self.predictor.ticker} Stock Price Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        self.canvas.draw()
        
        # Update metrics
        metrics_str = ""
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_str += f"{key}: {value:.4f}\n"
            else:
                metrics_str += f"{key}: {value}\n"
        self.metrics_text.setText(metrics_str)
        
        # Update forecast
        forecast_str = ""
        for i, price in enumerate(forecast, 1):
            forecast_str += f"Day {i}: {price:.2f}\n"
        self.forecast_text.setText(forecast_str)
        
        # Enable save buttons
        self.save_plot_button.setEnabled(True)
        self.save_results_button.setEnabled(True)
        
        # Enable run button
        self.run_button.setEnabled(True)
        
        # Emit signal for other tabs
        self.prediction_finished.emit(self.predictor, y_true, y_pred, forecast)
    
    def save_plot(self):
        # Implement save plot functionality
        pass
        
    def save_results(self):
        # Implement save results functionality
        pass 