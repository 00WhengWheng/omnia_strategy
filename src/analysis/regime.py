import numpy as np
from typing import Dict, Any
import pandas as pd
from scipy.stats import norm
from datetime import datetime
from .base import BaseAnalyzer, AnalysisResult
from ..core.constants import MarketRegime

class MarketRegimeAnalyzer(BaseAnalyzer):
    def _initialize_analyzer(self) -> None:
        self.lookback = self.config.get('regime.lookback_window', 252)
        self.vol_threshold = self.config.get('regime.crisis_vol_threshold', 30)
        self.corr_threshold = self.config.get('regime.crisis_corr_threshold', 0.7)
        self.trend_threshold = self.config.get('regime.trend_threshold', 0.3)
        
        self.current_regime = MarketRegime.NORMAL
        self.regime_probabilities = {regime: 0.0 for regime in MarketRegime}
        
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        # Analyze the market regime
        if not self.validate_data(data):
            raise ValueError("Invalid data for regime analysis")
            
        # Get the components
        volatility = self._calculate_volatility(data)
        correlation = self._calculate_correlation(data)
        trend = self._calculate_trend(data)
        
        # Determine the regime
        regime = self._determine_regime(volatility, correlation, trend)
        
        # Calculate regime probabilities
        probabilities = self._calculate_regime_probabilities(
            volatility, correlation, trend)
            
        # Calculate confidence in regime determination
        confidence = self._calculate_confidence(probabilities)
        
        result = AnalysisResult(
            timestamp=datetime.now(),
            value=self._regime_to_value(regime),
            confidence=confidence,
            components={
                'volatility': volatility,
                'correlation': correlation,
                'trend': trend
            },
            metadata={
                'regime': regime.value,
                'probabilities': probabilities
            }
        )
        
        self.update_history(result)
        return result
        
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        # Calculate the annualized volatility
        returns = data['close'].pct_change()
        return returns.std() * np.sqrt(252) * 100
        
    def _calculate_correlation(self, data: pd.DataFrame) -> float:
        # Get the correlation between close prices and other assets
        if 'other_assets' in data.columns:
            corr_matrix = data[['close', 'other_assets']].corr()
            return corr_matrix.iloc[0, 1]
        return 0.0
        
    def _calculate_trend(self, data: pd.DataFrame) -> float:
        # Calculate the trend strength
        prices = data['close']
        sma_fast = prices.rolling(50).mean()
        sma_slow = prices.rolling(200).mean()
        
        trend_strength = (sma_fast.iloc[-1] / sma_slow.iloc[-1] - 1)
        return np.clip(trend_strength, -1, 1)
        
    def _determine_regime(self, volatility: float, 
                         correlation: float, trend: float) -> MarketRegime:
        # Determine the market regime based on the components
        if volatility > self.vol_threshold and correlation > self.corr_threshold:
            return MarketRegime.CRISIS
            
        if trend < -self.trend_threshold and correlation > self.corr_threshold:
            return MarketRegime.RISK_OFF
            
        if trend > self.trend_threshold and volatility < self.vol_threshold:
            return MarketRegime.RISK_ON
            
        if trend > self.trend_threshold * 2 and volatility < self.vol_threshold / 2:
            return MarketRegime.EUPHORIA
            
        return MarketRegime.NORMAL
        
    def _calculate_regime_probabilities(self, volatility: float,
                                      correlation: float, 
                                      trend: float) -> Dict[str, float]:
        # Calculate the probabilities of each regime
        probs = {}
        
        for regime in MarketRegime:
            if regime == MarketRegime.CRISIS:
                prob = norm.cdf(volatility, loc=self.vol_threshold, scale=5)
                prob *= norm.cdf(correlation, loc=self.corr_threshold, scale=0.1)
                
            elif regime == MarketRegime.RISK_OFF:
                prob = norm.cdf(-trend, loc=self.trend_threshold, scale=0.2)
                prob *= norm.cdf(correlation, loc=self.corr_threshold, scale=0.1)
                
            elif regime == MarketRegime.RISK_ON:
                prob = norm.cdf(trend, loc=self.trend_threshold, scale=0.2)
                prob *= norm.cdf(-volatility, loc=self.vol_threshold, scale=5)
                
            elif regime == MarketRegime.EUPHORIA:
                prob = norm.cdf(trend, loc=self.trend_threshold*2, scale=0.2)
                prob *= norm.cdf(-volatility, loc=self.vol_threshold/2, scale=5)
                
            else:  # NORMAL
                prob = 1.0 - sum(probs.values())
                
            probs[regime.value] = np.clip(prob, 0, 1)
            
        # Normalize the probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
            
        return probs
        
    def _calculate_confidence(self, probabilities: Dict[str, float]) -> float:
        # Calculate the confidence in the regime determination
        # Get the difference between the highest and second highest probabilities
        max_prob = max(probabilities.values())
        second_max = sorted(probabilities.values())[-2]
        
        # 
        prob_diff = max_prob - second_max
        
        return np.clip(prob_diff * 2, 0, 1)  # Scala 0-1
        
    def _regime_to_value(self, regime: MarketRegime) -> float:
        # Convert the regime to a numerical value
        regime_values = {
            MarketRegime.CRISIS: -1.0,
            MarketRegime.RISK_OFF: -0.5,
            MarketRegime.NORMAL: 0.0,
            MarketRegime.RISK_ON: 0.5,
            MarketRegime.EUPHORIA: 1.0
        }
        return regime_values[regime]
        
    def get_required_columns(self) -> list:
        return ['close', 'high', 'low', 'volume']
