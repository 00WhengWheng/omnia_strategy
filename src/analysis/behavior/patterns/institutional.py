# institutional.py
class EnhancedInstitutionalBehavior:
    def analyze_institutional_patterns(self):
        """Enhanced institutional behavior analysis"""
        
        analysis = pd.DataFrame({
            # Volume Profile Analysis
            'Volume_Profile': self._analyze_volume_profile(),
            
            # Time-Weighted Analysis
            'Time_Weighted_Impact': self._analyze_time_weighted_impact(),
            
            # Order Flow Analysis
            'Order_Flow_Imbalance': self._analyze_order_flow_imbalance(),
            
            # Algorithmic Pattern Detection
            'Algo_Presence': self._detect_algorithmic_patterns(),
            
            # Institutional Timing Analysis
            'Institutional_Timing': self._analyze_institutional_timing()
        })
        
        return analysis
    
    def _analyze_volume_profile(self):
        """Analyze volume distribution patterns"""
        price = self.data['Close']
        volume = self.data['Volume']
        
        # Create price bins
        price_bins = pd.qcut(price, 10)
        
        # Calculate volume profile
        volume_profile = volume.groupby(price_bins).sum()
        
        # Normalize and detect institutional levels
        return (volume_profile / volume_profile.mean()).reindex(price.index)
    
    def _detect_algorithmic_patterns(self):
        """Detect algorithmic trading patterns"""
        price = self.data['Close']
        volume = self.data['Volume']
        
        # Time-based patterns
        time_intervals = pd.Series(self.data.index.time)
        
        # Regular execution patterns
        regular_patterns = (
            volume.groupby(time_intervals)
            .mean()
            .reindex(time_intervals)
        )
        
        # Detect algorithmic footprint
        algo_score = (
            (volume > regular_patterns.mean() + regular_patterns.std()) &
            (abs(price.pct_change()) < price.pct_change().std())
        )
        
        return algo_score.astype(int)