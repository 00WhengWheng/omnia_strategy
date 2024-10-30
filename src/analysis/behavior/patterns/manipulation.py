# manipulation.py
import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta

class MarketManipulation:
    def __init__(self, symbol, lookback_period=252):
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.data = self._fetch_market_data()
        
    def _fetch_market_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period)
        return yf.download(self.symbol, start=start_date, end=end_date)
    
    def detect_price_manipulation(self):
        """
        Detect potential price manipulation patterns:
        - Pump and dump patterns
        - Wash trading
        - Price painting
        - Spoofing patterns
        """
        price = self.data['Close']
        volume = self.data['Volume']
        
        manipulation_indicators = pd.DataFrame({
            'Sudden_Price_Moves': self._detect_sudden_moves(price),
            'Volume_Spikes': self._detect_volume_spikes(volume),
            'Price_Painting': self._detect_price_painting(price),
            'Wash_Trading': self._detect_wash_trading(price, volume)
        })
        
        return manipulation_indicators

    def detect_layered_manipulation(self):
    """Enhanced manipulation detection with multiple layers"""
    price = self.data['Close']
    volume = self.data['Volume']
    
    # Layer 1: Price action analysis
    price_returns = price.pct_change()
    volatility = price_returns.rolling(20).std()
    
    # Layer 2: Volume analysis
    volume_ratio = volume / volume.rolling(20).mean()
    
    # Layer 3: Time-based patterns
    intraday_pattern = self.data['High'] - self.data['Low']
    
    manipulation_score = pd.DataFrame({
        'Price_Manipulation': (
            (abs(price_returns) > 2 * volatility) & 
            (volume_ratio > 2)
        ).astype(int),
        'Volume_Manipulation': (
            (volume_ratio > 3) & 
            (abs(price_returns) < 0.5 * volatility)
        ).astype(int),
        'Pattern_Manipulation': (
            (intraday_pattern < intraday_pattern.rolling(20).mean() * 0.2) &
            (volume_ratio > 2)
        ).astype(int)
    })
    
    # Calculate composite score
    manipulation_score['Composite'] = (
        manipulation_score.sum(axis=1) / 3 * 100
    )
    
    return manipulation_score
    
    def _detect_sudden_moves(self, price):
        """Detect sudden price movements"""
        returns = price.pct_change()
        rolling_std = returns.rolling(20).std()
        
        return (abs(returns) > 3 * rolling_std).astype(int)
    
    def _detect_volume_spikes(self, volume):
        """Detect suspicious volume spikes"""
        vol_ma = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        
        return (volume > vol_ma + 3 * vol_std).astype(int)
    
    def _detect_price_painting(self, price):
        """Detect potential price painting patterns"""
        close_to_high = (self.data['High'] - self.data['Close']) / self.data['High']
        
        return (close_to_high < 0.001).astype(int)
    
    def _detect_wash_trading(self, price, volume):
        """Detect potential wash trading patterns"""
        price_volatility = price.pct_change().rolling(5).std()
        volume_intensity = volume / volume.rolling(20).mean()
        
        return ((price_volatility < 0.001) & (volume_intensity > 2)).astype(int)