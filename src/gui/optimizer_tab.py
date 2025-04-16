"""
Optimizer Tab Module
====================

Modul ini berisi implementasi tab optimizer strategi untuk UI aplikasi.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QProgressBar, QDoubleSpinBox, QMessageBox,
                             QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout, 
                             QSpinBox, QDateEdit)
from PyQt5.QtCore import Qt, QDate
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..utils.worker_threads import StrategyOptimizerWorker

class OptimizerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        # Layout utama
        layout = QVBoxLayout(self)
        
        # Input parameters
        input_layout = QHBoxLayout()
        
        # Ticker input
        ticker_layout = QVBoxLayout()
        ticker_label = QLabel("Ticker Symbol:")
        self.ticker_input = QComboBox()
        self.ticker_input.setEditable(True)
        self.ticker_input.addItems(['ADRO.JK', 'BBCA.JK', 'BBRI.JK', 'TLKM.JK', 'ASII.JK'])
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
        
        # Strategy parameters groups
        param_section = QHBoxLayout()
        
        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(['cnn_lstm', 'bilstm', 'transformer', 'ensemble'])
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        param_section.addWidget(model_group)
        
        # Strategy selection
        strategy_group = QGroupBox("Strategy")
        strategy_layout = QVBoxLayout()
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['trend_following', 'mean_reversion', 'predictive'])
        self.strategy_combo.currentIndexChanged.connect(self.on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo)
        strategy_group.setLayout(strategy_layout)
        param_section.addWidget(strategy_group)
        
        # Parameter ranges
        self.trend_following_group = QGroupBox("Trend Following Parameters")
        trend_following_layout = QFormLayout()
        self.trend_thresholds = []
        threshold_min = QDoubleSpinBox()
        threshold_min.setRange(0.001, 0.1)
        threshold_min.setSingleStep(0.001)
        threshold_min.setValue(0.005)
        threshold_max = QDoubleSpinBox()
        threshold_max.setRange(0.001, 0.1)
        threshold_max.setSingleStep(0.001)
        threshold_max.setValue(0.02)
        threshold_step = QDoubleSpinBox()
        threshold_step.setRange(0.001, 0.01)
        threshold_step.setSingleStep(0.001)
        threshold_step.setValue(0.005)
        trend_following_layout.addRow("Min Threshold:", threshold_min)
        trend_following_layout.addRow("Max Threshold:", threshold_max)
        trend_following_layout.addRow("Step:", threshold_step)
        self.trend_thresholds = [threshold_min, threshold_max, threshold_step]
        self.trend_following_group.setLayout(trend_following_layout)
        param_section.addWidget(self.trend_following_group)
        
        self.mean_reversion_group = QGroupBox("Mean Reversion Parameters")
        mean_reversion_layout = QFormLayout()
        self.window_sizes = []
        window_min = QSpinBox()
        window_min.setRange(5, 50)
        window_min.setValue(10)
        window_max = QSpinBox()
        window_max.setRange(10, 100)
        window_max.setValue(30)
        window_step = QSpinBox()
        window_step.setRange(1, 10)
        window_step.setValue(5)
        
        self.buy_thresholds = []
        buy_threshold_min = QDoubleSpinBox()
        buy_threshold_min.setRange(0.01, 0.2)
        buy_threshold_min.setSingleStep(0.01)
        buy_threshold_min.setValue(0.05)
        buy_threshold_max = QDoubleSpinBox()
        buy_threshold_max.setRange(0.01, 0.2)
        buy_threshold_max.setSingleStep(0.01)
        buy_threshold_max.setValue(0.1)
        buy_threshold_step = QDoubleSpinBox()
        buy_threshold_step.setRange(0.01, 0.05)
        buy_threshold_step.setSingleStep(0.01)
        buy_threshold_step.setValue(0.01)
        
        self.sell_thresholds = []
        sell_threshold_min = QDoubleSpinBox()
        sell_threshold_min.setRange(0.01, 0.2)
        sell_threshold_min.setSingleStep(0.01)
        sell_threshold_min.setValue(0.05)
        sell_threshold_max = QDoubleSpinBox()
        sell_threshold_max.setRange(0.01, 0.2)
        sell_threshold_max.setSingleStep(0.01)
        sell_threshold_max.setValue(0.1)
        sell_threshold_step = QDoubleSpinBox()
        sell_threshold_step.setRange(0.01, 0.05)
        sell_threshold_step.setSingleStep(0.01)
        sell_threshold_step.setValue(0.01)
        
        mean_reversion_layout.addRow("Min Window Size:", window_min)
        mean_reversion_layout.addRow("Max Window Size:", window_max)
        mean_reversion_layout.addRow("Window Step:", window_step)
        mean_reversion_layout.addRow("Min Buy Threshold:", buy_threshold_min)
        mean_reversion_layout.addRow("Max Buy Threshold:", buy_threshold_max)
        mean_reversion_layout.addRow("Buy Threshold Step:", buy_threshold_step)
        mean_reversion_layout.addRow("Min Sell Threshold:", sell_threshold_min)
        mean_reversion_layout.addRow("Max Sell Threshold:", sell_threshold_max)
        mean_reversion_layout.addRow("Sell Threshold Step:", sell_threshold_step)
        
        self.window_sizes = [window_min, window_max, window_step]
        self.buy_thresholds = [buy_threshold_min, buy_threshold_max, buy_threshold_step]
        self.sell_thresholds = [sell_threshold_min, sell_threshold_max, sell_threshold_step]
        
        self.mean_reversion_group.setLayout(mean_reversion_layout)
        param_section.addWidget(self.mean_reversion_group)
        
        self.predictive_group = QGroupBox("Predictive Parameters")
        predictive_layout = QFormLayout()
        self.buy_multipliers = []
        buy_mult_min = QDoubleSpinBox()
        buy_mult_min.setRange(1.001, 1.05)
        buy_mult_min.setSingleStep(0.001)
        buy_mult_min.setValue(1.005)
        buy_mult_max = QDoubleSpinBox()
        buy_mult_max.setRange(1.001, 1.05)
        buy_mult_max.setSingleStep(0.001)
        buy_mult_max.setValue(1.02)
        buy_mult_step = QDoubleSpinBox()
        buy_mult_step.setRange(0.001, 0.01)
        buy_mult_step.setSingleStep(0.001)
        buy_mult_step.setValue(0.005)
        
        self.sell_multipliers = []
        sell_mult_min = QDoubleSpinBox()
        sell_mult_min.setRange(0.9, 0.999)
        sell_mult_min.setSingleStep(0.001)
        sell_mult_min.setValue(0.98)
        sell_mult_max = QDoubleSpinBox()
        sell_mult_max.setRange(0.9, 0.999)
        sell_mult_max.setSingleStep(0.001)
        sell_mult_max.setValue(0.995)
        sell_mult_step = QDoubleSpinBox()
        sell_mult_step.setRange(0.001, 0.01)
        sell_mult_step.setSingleStep(0.001)
        sell_mult_step.setValue(0.005)
        
        predictive_layout.addRow("Min Buy Multiplier:", buy_mult_min)
        predictive_layout.addRow("Max Buy Multiplier:", buy_mult_max)
        predictive_layout.addRow("Buy Multiplier Step:", buy_mult_step)
        predictive_layout.addRow("Min Sell Multiplier:", sell_mult_min)
        predictive_layout.addRow("Max Sell Multiplier:", sell_mult_max)
        predictive_layout.addRow("Sell Multiplier Step:", sell_mult_step)
        
        self.buy_multipliers = [buy_mult_min, buy_mult_max, buy_mult_step]
        self.sell_multipliers = [sell_mult_min, sell_mult_max, sell_mult_step]
        
        self.predictive_group.setLayout(predictive_layout)
        param_section.addWidget(self.predictive_group)
        
        # Show only relevant parameter group
        self.mean_reversion_group.hide()
        self.predictive_group.hide()
        
        layout.addLayout(param_section)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Run button
        self.run_button = QPushButton("Run Optimizer")
        self.run_button.clicked.connect(self.run_strategy_optimization)
        layout.addWidget(self.run_button)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels(
            ["Strategy", "Parameters", "Initial", "Final", "Return (%)", 
             "Max Drawdown (%)", "Win Rate (%)", "Trades"]
        )
        layout.addWidget(self.results_table)
        
        # Save button
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self.save_optimization_results)
        self.save_results_button.setEnabled(False)
        layout.addWidget(self.save_results_button)
    
    def on_strategy_changed(self):
        # Show relevant parameter group based on selected strategy
        strategy = self.strategy_combo.currentText()
        
        self.trend_following_group.hide()
        self.mean_reversion_group.hide()
        self.predictive_group.hide()
        
        if strategy == 'trend_following':
            self.trend_following_group.show()
        elif strategy == 'mean_reversion':
            self.mean_reversion_group.show()
        elif strategy == 'predictive':
            self.predictive_group.show()
    
    def run_strategy_optimization(self):
        # Disable run button
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Get parameters
        ticker = self.ticker_input.currentText()
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")
        model_type = self.model_combo.currentText()
        strategy = self.strategy_combo.currentText()
        initial_investment = self.investment_spin.value()
        
        # Get parameter ranges based on strategy
        param_ranges = {}
        if strategy == 'trend_following':
            min_threshold = self.trend_thresholds[0].value()
            max_threshold = self.trend_thresholds[1].value()
            step = self.trend_thresholds[2].value()
            thresholds = [round(min_threshold + i * step, 4) 
                         for i in range(int((max_threshold - min_threshold) / step) + 1)]
            param_ranges = {'threshold': thresholds}
            
        elif strategy == 'mean_reversion':
            min_window = self.window_sizes[0].value()
            max_window = self.window_sizes[1].value()
            window_step = self.window_sizes[2].value()
            
            min_buy = self.buy_thresholds[0].value()
            max_buy = self.buy_thresholds[1].value()
            buy_step = self.buy_thresholds[2].value()
            
            min_sell = self.sell_thresholds[0].value()
            max_sell = self.sell_thresholds[1].value()
            sell_step = self.sell_thresholds[2].value()
            
            windows = [min_window + i * window_step 
                       for i in range(int((max_window - min_window) / window_step) + 1)]
            buy_thresholds = [round(min_buy + i * buy_step, 4) 
                             for i in range(int((max_buy - min_buy) / buy_step) + 1)]
            sell_thresholds = [round(min_sell + i * sell_step, 4) 
                              for i in range(int((max_sell - min_sell) / sell_step) + 1)]
            
            param_ranges = {
                'window': windows,
                'buy_threshold': buy_thresholds,
                'sell_threshold': sell_thresholds
            }
            
        elif strategy == 'predictive':
            min_buy = self.buy_multipliers[0].value()
            max_buy = self.buy_multipliers[1].value()
            buy_step = self.buy_multipliers[2].value()
            
            min_sell = self.sell_multipliers[0].value()
            max_sell = self.sell_multipliers[1].value()
            sell_step = self.sell_multipliers[2].value()
            
            buy_multipliers = [round(min_buy + i * buy_step, 4) 
                              for i in range(int((max_buy - min_buy) / buy_step) + 1)]
            sell_multipliers = [round(min_sell + i * sell_step, 4) 
                               for i in range(int((max_sell - min_sell) / sell_step) + 1)]
            
            param_ranges = {
                'buy_threshold': buy_multipliers,
                'sell_threshold': sell_multipliers
            }
        
        # Create and start worker thread
        self.worker = StrategyOptimizerWorker(
            ticker, start_date, end_date,
            model_type, initial_investment,
            strategy, param_ranges
        )
        self.worker.finished.connect(self.on_optimization_finished)
        self.worker.progress.connect(self.update_opt_progress)
        self.worker.error.connect(self.show_opt_error)
        self.worker.start()
    
    def update_opt_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_opt_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.run_button.setEnabled(True)
    
    def on_optimization_finished(self, results):
        # Update results table
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            strategy_item = QTableWidgetItem(result['strategy'])
            
            params_str = ""
            for key, value in result['params'].items():
                params_str += f"{key}: {value}, "
            params_str = params_str[:-2]  # Remove trailing comma and space
            params_item = QTableWidgetItem(params_str)
            
            initial_item = QTableWidgetItem(f"${result['performance']['initial_investment']:.2f}")
            final_item = QTableWidgetItem(f"${result['performance']['final_value']:.2f}")
            return_item = QTableWidgetItem(f"{result['performance']['total_return']:.2f}")
            drawdown_item = QTableWidgetItem(f"{result['performance']['max_drawdown']:.2f}")
            winrate_item = QTableWidgetItem(f"{result['performance']['win_rate']:.2f}")
            trades_item = QTableWidgetItem(f"{result['performance']['num_trades']}")
            
            self.results_table.setItem(i, 0, strategy_item)
            self.results_table.setItem(i, 1, params_item)
            self.results_table.setItem(i, 2, initial_item)
            self.results_table.setItem(i, 3, final_item)
            self.results_table.setItem(i, 4, return_item)
            self.results_table.setItem(i, 5, drawdown_item)
            self.results_table.setItem(i, 6, winrate_item)
            self.results_table.setItem(i, 7, trades_item)
        
        # Enable buttons
        self.run_button.setEnabled(True)
        self.save_results_button.setEnabled(True)
    
    def save_optimization_results(self):
        # Implement save results functionality
        pass 