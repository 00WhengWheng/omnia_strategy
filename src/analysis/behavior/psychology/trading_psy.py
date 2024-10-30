# trading_psychology.py
class TradingPsychology:
    def __init__(self, symbol, lookback_period=252):
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.data = self._fetch_market_data()
        
    def _fetch_market_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period)
        return yf.download(self.symbol, start=start_date, end=end_date)
    
    def analyze_market_sentiment(self):
        """Analyze market sentiment indicators"""
        price = self.data['Close']
        volume = self.data['Volume']
        
        sentiment = pd.DataFrame({
            'Buying_Pressure': self._calculate_buying_pressure(),
            'Support_Resistance_Ratio': self._calculate_sr_ratio(),
            'Market_Quality': self._calculate_market_quality(),
            'Emotional_Index': self._calculate_emotional_index()
        })
        
        return sentiment
    
    def _calculate_buying_pressure(self):
        """Calculate buying pressure indicator"""
        highs = self.data['High']
        lows = self.data['Low']
        closes = self.data['Close']
        
        return ((closes - lows) / (highs - lows)).rolling(10).mean() * 100
    
    def _calculate_sr_ratio(self):
        """Calculate support/resistance ratio"""
        price = self.data['Close']
        ma_20 = price.rolling(20).mean()
        ma_50 = price.rolling(50).mean()
        
        return (price - ma_50) / (ma_20 - ma_50)
    
    def _calculate_market_quality(self):
        """Calculate market quality indicator"""
        returns = self.data['Close'].pct_change()
        volume = self.data['Volume']
        
        return (returns.abs() * volume).rolling(20).mean()
    
    def _calculate_emotional_index(self):
        """Calculate emotional trading index"""
        returns = self.data['Close'].pct_change()
        volume = self.data['Volume']
        
        volatility = returns.rolling(20).std()
        volume_ratio = volume / volume.rolling(20).mean()
        
        return (volatility * volume_ratio).rolling(5).mean()

class MarketBehaviorAnalyzer:
    """Main class to coordinate all behavioral analysis"""
    
    def __init__(self, symbol, lookback_period=252):
        self.symbol = symbol
        self.lookback_period = lookback_period
        
        # Initialize all analyzers
        self.fear_greed = FearGreedIndicators(symbol, lookback_period)
        self.momentum = MarketMomentum(symbol, lookback_period)
        self.crowd = CrowdBehavior(symbol, lookback_period)
        self.psychology = TradingPsychology(symbol, lookback_period)
    
    def get_complete_analysis(self):
        """Get comprehensive market behavior analysis"""
        
        analysis = {
            'fear_greed_index': self.fear_greed.get_fear_greed_index()[1].iloc[-1],
            'market_state': self.fear_greed.get_market_state(),
            'momentum_indicators': self.momentum.calculate_price_momentum().iloc[-1],
            'volume_momentum': self.momentum.calculate_volume_momentum().iloc[-1],
            'momentum_regime': self.momentum.identify_momentum_regimes().iloc[-1],
            'herd_behavior': self.crowd.analyze_herd_behavior().iloc[-1],
            'market_extremes': self.crowd.detect_market_extremes().iloc[-1],
            'market_sentiment': self.psychology.analyze_market_sentiment().iloc[-1]
        }
        
        return analysis

def main():
    # Example usage
    analyzer = MarketBehaviorAnalyzer("SPY")
    
    # Get complete analysis
    analysis = analyzer.get_complete_analysis()
    
    # Print results
    print("\nMarket Behavior Analysis:")
    print(f"Fear & Greed Index: {analysis['fear_greed_index']:.2f}")
    print(f"Market State: {analysis['market_state']}")
    
    print("\nMomentum Analysis:")
    print(analysis['momentum_indicators'])
    
    print("\nCrowd Behavior:")
    print(analysis['herd_behavior'])
    
    print("\nMarket Sentiment:")
    print(analysis['market_sentiment'])

if __name__ == "__main__":
    main()