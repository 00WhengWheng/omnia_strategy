# market_momentum.py
import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta

class MarketMomentum:
    def __init__(self, symbol, lookback_period=252):
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.data = self._fetch_market_data()
        
    def _fetch_market_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period)
        return yf.download(self.symbol, start=start_date, end=end_date)
    
    def calculate_price_momentum(self):
        """Calculate various momentum indicators"""
        price = self.data['Close']
        
        momentum_indicators = pd.DataFrame({
            'ROC_5': price.pct_change(5) * 100,
            'ROC_10': price.pct_change(10) * 100,
            'ROC_20': price.pct_change(20) * 100,
            'MA_Cross': (price.rolling(10).mean() - price.rolling(20).mean()) / price,
            'Price_Distance': (price - price.rolling(50).mean()) / price.rolling(50).std()
        })
        
        return momentum_indicators
    
    def calculate_volume_momentum(self):
        """Analyze volume-based momentum"""
        volume = self.data['Volume']
        price = self.data['Close']
        
        volume_metrics = pd.DataFrame({
            'Volume_ROC': volume.pct_change(5) * 100,
            'Volume_MA_Ratio': volume / volume.rolling(20).mean(),
            'PVO': ((volume.rolling(12).mean() - volume.rolling(26).mean()) / 
                    volume.rolling(26).mean()) * 100,
            'Price_Volume_Trend': (price.pct_change() * volume).cumsum()
        })
        
        return volume_metrics
    
    def identify_momentum_regimes(self):
        """Identify different momentum regimes"""
        returns = self.data['Close'].pct_change()
        
        # Calculate regime metrics
        rolling_mean = returns.rolling(20).mean()
        rolling_std = returns.rolling(20).std()
        
        regimes = pd.DataFrame({
            'Trend_Strength': abs(rolling_mean) / rolling_std,
            'Regime': np.where(rolling_mean > 0, 'Bullish', 'Bearish'),
            'Volatility_Regime': np.where(
                rolling_std > rolling_std.rolling(60).mean(),
                'High Volatility',
                'Low Volatility'
            )
        })
        
        return regimes
