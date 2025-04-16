"""
Backtest Tab Module
====================

Modul ini berisi implementasi tab backtesting untuk UI aplikasi.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QProgressBar, QDoubleSpinBox, QMessageBox,
                             QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..utils.worker_threads import BacktestWorker

class BacktestTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = None
        self.actual_prices = None
        self.predicted_prices = None
        self.setup_ui()
        
    def setup_ui(self):
        # Layout utama
        layout = QVBoxLayout(self)
        
        # Input parameters
        input_layout = QHBoxLayout()
        
        # Strategy selection
        strategy_layout = QVBoxLayout()
        strategy_label = QLabel("Trading Strategy:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['trend_following', 'mean_reversion', 'predictive'])
        strategy_layout.addWidget(strategy_label)
        strategy_layout.addWidget(self.strategy_combo)
        input_layout.addLayout(strategy_layout)
        
        # Initial investment
        investment_layout = QVBoxLayout()
        investment_label = QLabel("Initial Investment:")
        self.investment_spin = QDoubleSpinBox()
        self.investment_spin.setRange(1000, 1000000)
        self.investment_spin.setValue(10000)
        self.investment_spin.setSingleStep(1000)
        self.investment_spin.setPrefix("$ ")
        investment_layout.addWidget(investment_label)
        investment_layout.addWidget(self.investment_spin)
        input_layout.addLayout(investment_layout)
        
        layout.addLayout(input_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Run button
        self.run_button = QPushButton("Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)
        self.run_button.setEnabled(False)  # Disabled until prediction is done
        layout.addWidget(self.run_button)
        
        # Plot area
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Trade history table
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(5)
        self.trades_table.setHorizontalHeaderLabels(["Day", "Type", "Price", "Shares", "Value"])
        layout.addWidget(self.trades_table)
        
        # Performance metrics
        self.performance_label = QLabel("Performance Metrics:")
        layout.addWidget(self.performance_label)
        
        # Save buttons
        save_layout = QHBoxLayout()
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_backtest_plot)
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self.save_backtest_results)
        save_layout.addWidget(self.save_plot_button)
        save_layout.addWidget(self.save_results_button)
        layout.addLayout(save_layout)
        
        # Disable save buttons initially
        self.save_plot_button.setEnabled(False)
        self.save_results_button.setEnabled(False)
    
    @pyqtSlot(object, object, object)
    def set_prediction_results(self, predictor, actual_prices, predicted_prices):
        """
        Set the prediction results from the prediction tab
        
        Parameters:
        -----------
        predictor : StockPredictor
            The predictor object
        actual_prices : array-like
            Actual prices from the prediction
        predicted_prices : array-like
            Predicted prices from the model
        """
        self.predictor = predictor
        self.actual_prices = actual_prices
        self.predicted_prices = predicted_prices
        self.run_button.setEnabled(True)
    
    def run_backtest(self):
        # Disable run button during backtesting
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        if self.actual_prices is None or self.predicted_prices is None:
            QMessageBox.warning(self, "Warning", "Please run prediction first")
            self.run_button.setEnabled(True)
            return
        
        # Get parameters
        strategy = self.strategy_combo.currentText()
        initial_investment = self.investment_spin.value()
        
        # Create and start worker thread
        self.worker = BacktestWorker(self.predictor, initial_investment, strategy)
        self.worker.finished.connect(self.on_backtest_finished)
        self.worker.progress.connect(self.update_backtest_progress)
        self.worker.error.connect(self.show_backtest_error)
        self.worker.start()
    
    def update_backtest_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_backtest_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.run_button.setEnabled(True)
    
    def on_backtest_finished(self, portfolio_values, trades, performance):
        # Update plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(portfolio_values, label='Portfolio Value')
        ax.set_title('Backtest Results')
        ax.set_xlabel('Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True)
        self.canvas.draw()
        
        # Update trades table
        self.trades_table.setRowCount(len(trades))
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade['day'])))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade['type']))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"${trade['price']:.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(f"{trade['shares']:.4f}"))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"${trade['value']:.2f}"))
        
        # Update performance metrics
        performance_text = ""
        for key, value in performance.items():
            if key in ['total_return', 'max_drawdown', 'win_rate']:
                performance_text += f"{key}: {value:.2f}%\n"
            elif isinstance(value, float):
                performance_text += f"{key}: ${value:.2f}\n"
            else:
                performance_text += f"{key}: {value}\n"
        self.performance_label.setText(f"Performance Metrics:\n{performance_text}")
        
        # Enable buttons
        self.run_button.setEnabled(True)
        self.save_plot_button.setEnabled(True)
        self.save_results_button.setEnabled(True)
    
    def save_backtest_plot(self):
        # Implement save plot functionality
        pass
        
    def save_backtest_results(self):
        # Implement save results functionality
        pass 