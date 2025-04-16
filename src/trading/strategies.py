"""
Trading Strategies Module
=======================

Modul ini berisi implementasi berbagai strategi trading
yang digunakan dalam backtest.
"""

class TradingStrategy:
    @staticmethod
    def trend_following(predicted_prices, actual_prices, index, params=None):
        """
        Strategi Trend Following
        
        Beli jika prediksi menunjukkan tren naik, jual jika prediksi menunjukkan tren turun.
        
        Parameters:
        -----------
        predicted_prices : array-like
            Harga prediksi dari model
        actual_prices : array-like
            Harga aktual historis
        index : int
            Indeks waktu saat ini
        params : dict, optional
            Parameter tambahan untuk strategi ini
            
        Returns:
        --------
        str
            Sinyal trading: 'BUY', 'SELL', atau 'HOLD'
        """
        if params is None:
            params = {'threshold': 0.01}
            
        threshold = params.get('threshold', 0.01)
        
        # Beli jika prediksi menunjukkan tren naik
        if index > 1 and predicted_prices[index] > predicted_prices[index-1] * (1 + threshold):
            return 'BUY'
        # Jual jika prediksi menunjukkan tren turun
        elif index > 1 and predicted_prices[index] < predicted_prices[index-1] * (1 - threshold):
            return 'SELL'
        
        return 'HOLD'
    
    @staticmethod
    def mean_reversion(predicted_prices, actual_prices, index, params=None):
        """
        Strategi Mean Reversion
        
        Beli jika harga di bawah rata-rata (oversold), jual jika harga di atas rata-rata (overbought).
        
        Parameters:
        -----------
        predicted_prices : array-like
            Harga prediksi dari model
        actual_prices : array-like
            Harga aktual historis
        index : int
            Indeks waktu saat ini
        params : dict, optional
            Parameter tambahan untuk strategi ini
            
        Returns:
        --------
        str
            Sinyal trading: 'BUY', 'SELL', atau 'HOLD'
        """
        if params is None:
            params = {'window': 5, 'buy_threshold': 0.98, 'sell_threshold': 1.02}
            
        window = params.get('window', 5)
        buy_threshold = params.get('buy_threshold', 0.98)
        sell_threshold = params.get('sell_threshold', 1.02)
        
        # Hitung rata-rata bergerak
        if index >= window:
            sma = sum(actual_prices[index-window:index]) / window
            # Beli jika harga di bawah rata-rata (oversold)
            if actual_prices[index] < sma * buy_threshold:
                return 'BUY'
            # Jual jika harga di atas rata-rata (overbought)
            elif actual_prices[index] > sma * sell_threshold:
                return 'SELL'
        
        return 'HOLD'
    
    @staticmethod
    def predictive(predicted_prices, actual_prices, index, params=None):
        """
        Strategi Predictive
        
        Beli jika prediksi menunjukkan harga akan naik, jual jika prediksi menunjukkan harga akan turun.
        
        Parameters:
        -----------
        predicted_prices : array-like
            Harga prediksi dari model
        actual_prices : array-like
            Harga aktual historis
        index : int
            Indeks waktu saat ini
        params : dict, optional
            Parameter tambahan untuk strategi ini
            
        Returns:
        --------
        str
            Sinyal trading: 'BUY', 'SELL', atau 'HOLD'
        """
        if params is None:
            params = {'buy_threshold': 1.01, 'sell_threshold': 0.99}
            
        buy_threshold = params.get('buy_threshold', 1.01)
        sell_threshold = params.get('sell_threshold', 0.99)
        
        # Beli jika prediksi menunjukkan harga akan naik
        if predicted_prices[index] > actual_prices[index] * buy_threshold:
            return 'BUY'
        # Jual jika prediksi menunjukkan harga akan turun
        elif predicted_prices[index] < actual_prices[index] * sell_threshold:
            return 'SELL'
        
        return 'HOLD'
    
    @staticmethod
    def get_strategy_function(strategy_name):
        """
        Mendapatkan fungsi strategi berdasarkan nama
        
        Parameters:
        -----------
        strategy_name : str
            Nama strategi
            
        Returns:
        --------
        function
            Fungsi strategi yang sesuai
            
        Raises:
        -------
        ValueError
            Jika strategi tidak ditemukan
        """
        strategies = {
            'Trend Following': TradingStrategy.trend_following,
            'Mean Reversion': TradingStrategy.mean_reversion,
            'Predictive': TradingStrategy.predictive,
            'trend_following': TradingStrategy.trend_following,
            'mean_reversion': TradingStrategy.mean_reversion,
            'predictive': TradingStrategy.predictive
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Strategi '{strategy_name}' tidak ditemukan")
            
        return strategies[strategy_name] 