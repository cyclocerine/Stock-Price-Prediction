#!/usr/bin/env python
"""
GUI Application Runner
====================

Script untuk menjalankan aplikasi prediksi saham dengan antarmuka grafis.
"""

import sys
import os

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5.QtWidgets import QApplication
from src.gui.app import StockPredictionApp

def main():
    """
    Fungsi utama untuk menjalankan aplikasi GUI
    """
    app = QApplication(sys.argv)
    window = StockPredictionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 