# smart_money.py
class SmartMoneyAnalysis:
    def __init__(self, symbol, lookback_period=252):
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.data = self._fetch_market_data()
    
    def _fetch_market_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period)
        return yf.download(self.symbol, start=start_date, end=end_date)
    
    def analyze_smart_money_movement(self):
        """
        Analyze smart money movement patterns:
        - Accumulation/Distribution
        - Block trades
        - Dark pool activity estimation
        - Institutional footprints
        """
        analysis = pd.DataFrame({
            'Accumulation': self._detect_accumulation(),
            'Distribution': self._detect_distribution(),
            'Block_Trades': self._detect_block_trades(),
            'Dark_Pool_Activity': self._estimate_dark_pool_activity()
        })
        
        return analysis

    def analyze_smart_money_flows(self):
    """Enhanced smart money flow analysis"""
    
    def calculate_vwap():
        return (self.data['Close'] * self.data['Volume']).cumsum() / self.data['Volume'].cumsum()
    
    def calculate_money_flow():
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        mf = typical_price * self.data['Volume']
        return mf.where(self.data['Close'] > self.data['Close'].shift(1), -mf)
    
    analysis = pd.DataFrame({
        # VWAP-based analysis
        'VWAP_Distance': (self.data['Close'] - calculate_vwap()) / calculate_vwap(),
        
        # Money Flow Analysis
        'Smart_Money_Flow': calculate_money_flow().rolling(20).sum(),
        
        # Block Trade Detection (improved)
        'Block_Trades': self._detect_sophisticated_blocks(),
        
        # Institutional Accumulation
        'Accumulation_Score': self._analyze_accumulation_patterns(),
        
        # Dark Pool Estimation (enhanced)
        'Dark_Pool_Probability': self._estimate_dark_pool_improved()
    })
    
        return analysis

    def _detect_sophisticated_blocks(self):
        """More sophisticated block trade detection"""
        volume = self.data['Volume']
        price = self.data['Close']
        
        # Calculate normal trading ranges
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        
        # Price impact
        price_impact = abs(price.pct_change())
        
        # Detect blocks with minimal price impact
        block_trades = (
            (volume > 3 * vol_mean) &  # Large volume
            (price_impact < price_impact.rolling(20).mean())  # Low price impact
        )
    
        return block_trades.astype(int)
    
    def _detect_accumulation(self):
        """Detect accumulation patterns"""
        price = self.data['Close']
        volume = self.data['Volume']
        
        # Wyckoff Accumulation Detection
        close_location = (price - self.data['Low']) / (self.data['High'] - self.data['Low'])
        volume_force = close_location * volume
        
        return volume_force.rolling(20).mean()
    
    def _detect_distribution(self):
        """Detect distribution patterns"""
        price = self.data['Close']
        volume = self.data['Volume']
        
        # Distribution Detection
        high_location = (self.data['High'] - price) / (self.data['High'] - self.data['Low'])
        selling_pressure = high_location * volume
        
        return selling_pressure.rolling(20).mean()
    
    def _detect_block_trades(self):
        """Detect potential block trades"""
        volume = self.data['Volume']
        avg_volume = volume.rolling(20).mean()
        
        return (volume > 5 * avg_volume).astype(int)
    
    def _estimate_dark_pool_activity(self):
        """Estimate dark pool activity"""
        price = self.data['Close']
        volume = self.data['Volume']
        
        # Estimate based on price impact
        price_impact = abs(price.pct_change()) / (volume / volume.rolling(20).mean())
        
        return (1 - price_impact.rolling(5).mean())