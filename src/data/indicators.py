"""
Technical Indicators Module
==========================

Modul ini berisi implementasi indikator-indikator teknikal
yang digunakan dalam analisis saham.
"""

import numpy as np
import pandas as pd

class TechnicalIndicators:
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """Menghitung Average Directional Index (ADX)"""
        try:
            # Konversi ke numpy array 1D
            high = np.array(high).flatten()
            low = np.array(low).flatten()
            close = np.array(close).flatten()
            
            # True Range
            high_low = high - low
            high_close_prev = np.abs(high - np.roll(close, 1))
            low_close_prev = np.abs(low - np.roll(close, 1))
            tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            
            # Directional Movement
            up_move = high - np.roll(high, 1)
            down_move = np.roll(low, 1) - low
            
            # +DM dan -DM
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothing
            tr_smooth = np.convolve(tr, np.ones(period)/period, mode='valid')
            pos_dm_smooth = np.convolve(pos_dm, np.ones(period)/period, mode='valid')
            neg_dm_smooth = np.convolve(neg_dm, np.ones(period)/period, mode='valid')
            
            # Directional Indicators
            pos_di = 100 * (pos_dm_smooth / tr_smooth)
            neg_di = 100 * (neg_dm_smooth / tr_smooth)
            
            # ADX
            dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
            adx = np.convolve(dx, np.ones(period)/period, mode='valid')
            
            # Padding
            pad_size = len(high) - len(adx)
            adx = np.pad(adx, (pad_size, 0), mode='edge')
            
            return adx
        except Exception as e:
            print(f"Error in calculate_adx: {str(e)}")
            return np.zeros_like(high)

    @staticmethod
    def calculate_rsi(close, period=14):
        """Menghitung Relative Strength Index (RSI)"""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return np.pad(rsi, (period, 0), mode='edge')

    @staticmethod
    def calculate_macd(close, fast=12, slow=26, signal=9):
        """Menghitung MACD"""
        exp1 = pd.Series(close).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = pd.Series(macd).ewm(span=signal, adjust=False).mean()
        return macd, signal_line 