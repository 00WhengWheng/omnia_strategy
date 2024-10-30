# fear_greed_indicators.py

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta

class FearGreedIndicators:
    def __init__(self, symbol, lookback_period=252):
        """
        Initialize the Fear & Greed analysis system
        
        Parameters:
        symbol (str): Market symbol to analyze (e.g., "SPY" for S&P 500)
        lookback_period (int): Number of trading days to analyze
        """
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.data = self._fetch_market_data()
        
    def _fetch_market_data(self):
        """Fetch market data using yfinance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period)
        return yf.download(self.symbol, start=start_date, end=end_date)

    def calculate_market_momentum(self):
        """
        Calculate market momentum signal (0-100)
        Uses 1-month and 3-month price changes
        """
        price = self.data['Close']
        
        # Calculate momentum over different timeframes
        mom_1m = price.pct_change(21)  # ~1 month
        mom_3m = price.pct_change(63)  # ~3 months
        
        # Normalize to 0-100 scale
        mom_1m_norm = self._normalize_series(mom_1m)
        mom_3m_norm = self._normalize_series(mom_3m)
        
        # Combine signals (weighted average)
        momentum_signal = (0.6 * mom_1m_norm + 0.4 * mom_3m_norm)
        
        return momentum_signal

    def calculate_volatility_index(self):
        """
        Calculate volatility indicator (0-100)
        Higher values indicate higher fear
        """
        returns = self.data['Close'].pct_change()
        
        # Calculate rolling volatility
        vol_20d = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # Normalize and invert (high volatility = high fear)
        volatility_signal = 100 - self._normalize_series(vol_20d)
        
        return volatility_signal

    def calculate_strength_index(self):
        """
        Calculate market strength index (0-100)
        Based on new highs vs new lows
        """
        price = self.data['Close']
        
        # Calculate 52-week high/low
        rolling_high = price.rolling(window=252).max()
        rolling_low = price.rolling(window=252).min()
        
        # Calculate distance from high/low
        dist_from_high = (rolling_high - price) / rolling_high
        dist_from_low = (price - rolling_low) / rolling_low
        
        # Create strength index
        strength = (dist_from_low / (dist_from_high + dist_from_low)) * 100
        
        return strength

    def calculate_volume_analysis(self):
        """
        Analyze trading volume patterns (0-100)
        High volume on up days = greed
        High volume on down days = fear
        """
        volume = self.data['Volume']
        returns = self.data['Close'].pct_change()
        
        # Calculate volume ratio
        avg_volume = volume.rolling(window=20).mean()
        volume_ratio = volume / avg_volume
        
        # Separate up and down days
        up_vol = volume_ratio.where(returns > 0, 0)
        down_vol = volume_ratio.where(returns < 0, 0)
        
        # Calculate volume signal
        vol_signal = (
            (up_vol.rolling(window=5).mean() - down_vol.rolling(window=5).mean())
            / (up_vol.rolling(window=5).mean() + down_vol.rolling(window=5).mean())
        )
        
        return self._normalize_series(vol_signal)

    def calculate_rsi_signal(self):
        """
        Calculate RSI-based sentiment (0-100)
        Standard RSI calculation with smoothing
        """
        close = self.data['Close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def calculate_price_breadth(self):
        """
        Calculate market breadth indicator (0-100)
        Based on percentage of stocks above moving averages
        """
        price = self.data['Close']
        
        # Calculate various moving averages
        ma_20 = price.rolling(window=20).mean()
        ma_50 = price.rolling(window=50).mean()
        ma_200 = price.rolling(window=200).mean()
        
        # Calculate percentage above each MA
        above_20 = (price > ma_20).astype(int)
        above_50 = (price > ma_50).astype(int)
        above_200 = (price > ma_200).astype(int)
        
        # Combine signals
        breadth = (0.4 * above_20 + 0.3 * above_50 + 0.3 * above_200) * 100
        
        return breadth

    def get_fear_greed_index(self):
        """
        Calculate complete Fear & Greed Index (0-100)
        Combines all indicators into a single score
        0 = Extreme Fear, 100 = Extreme Greed
        """
        # Calculate all components
        momentum = self.calculate_market_momentum()
        volatility = self.calculate_volatility_index()
        strength = self.calculate_strength_index()
        volume = self.calculate_volume_analysis()
        rsi = self.calculate_rsi_signal()
        breadth = self.calculate_price_breadth()
        
        # Combine indicators with weights
        fear_greed_index = pd.DataFrame({
            'Momentum': momentum,
            'Volatility': volatility,
            'Market Strength': strength,
            'Volume Patterns': volume,
            'RSI': rsi,
            'Market Breadth': breadth
        })
        
        # Calculate weighted average
        weights = {
            'Momentum': 0.20,
            'Volatility': 0.20,
            'Market Strength': 0.15,
            'Volume Patterns': 0.15,
            'RSI': 0.15,
            'Market Breadth': 0.15
        }
        
        composite_index = sum(fear_greed_index[col] * weight 
                            for col, weight in weights.items())
        
        return fear_greed_index, composite_index

    def get_market_state(self):
        """
        Get market state interpretation based on Fear & Greed Index
        """
        _, index = self.get_fear_greed_index()
        current_value = index.iloc[-1]
        
        if current_value >= 80:
            return "Extreme Greed"
        elif current_value >= 60:
            return "Greed"
        elif current_value >= 40:
            return "Neutral"
        elif current_value >= 20:
            return "Fear"
        else:
            return "Extreme Fear"

    @staticmethod
    def _normalize_series(series, min_val=0, max_val=100):
        """Normalize a series to a 0-100 scale"""
        return ((series - series.rolling(window=252).min()) /
                (series.rolling(window=252).max() - 
                 series.rolling(window=252).min())) * 100

def main():
    # Example usage
    analyzer = FearGreedIndicators("SPY")
    
    # Get complete analysis
    indicators, composite_index = analyzer.get_fear_greed_index()
    
    # Print current market state
    print(f"\nCurrent Market State: {analyzer.get_market_state()}")
    
    # Print latest indicator values
    print("\nLatest Indicator Values:")
    print(indicators.iloc[-1].round(2))
    
    # Print composite index
    print(f"\nComposite Fear & Greed Index: {composite_index.iloc[-1]:.2f}")

if __name__ == "__main__":
    main()
