# crowd_behavior.py
class CrowdBehavior:
    def __init__(self, symbol, lookback_period=252):
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.data = self._fetch_market_data()
        
    def _fetch_market_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period)
        return yf.download(self.symbol, start=start_date, end=end_date)
    
    def analyze_herd_behavior(self):
        """Analyze potential herding in the market"""
        returns = self.data['Close'].pct_change()
        volume = self.data['Volume']
        
        herd_metrics = pd.DataFrame({
            'Return_Dispersion': returns.rolling(20).std(),
            'Volume_Intensity': (volume - volume.rolling(20).mean()) / volume.rolling(20).std(),
            'Price_Acceleration': returns.diff().rolling(5).mean(),
            'Trend_Following': (
                (self.data['Close'] > self.data['Close'].rolling(10).mean()) &
                (volume > volume.rolling(10).mean())
            ).astype(int)
        })
        
        return herd_metrics
    
    def detect_market_extremes(self):
        """Detect extreme market behavior"""
        returns = self.data['Close'].pct_change()
        volume = self.data['Volume']
        
        extremes = pd.DataFrame({
            'Panic_Selling': (
                (returns < -2 * returns.rolling(20).std()) &
                (volume > 2 * volume.rolling(20).mean())
            ).astype(int),
            'FOMO_Buying': (
                (returns > 2 * returns.rolling(20).std()) &
                (volume > 2 * volume.rolling(20).mean())
            ).astype(int),
            'Capitulation': (
                (returns.rolling(5).sum() < -3 * returns.rolling(20).std()) &
                (volume.rolling(5).mean() > 2 * volume.rolling(20).mean())
            ).astype(int)
        })
        
        return extremes