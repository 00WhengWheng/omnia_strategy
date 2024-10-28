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
        """Analizza il regime di mercato corrente"""
        if not self.validate_data(data):
            raise ValueError("Invalid data for regime analysis")
            
        # Calcola metriche di regime
        volatility = self._calculate_volatility(data)
        correlation = self._calculate_correlation(data)
        trend = self._calculate_trend(data)
        
        # Determina il regime
        regime = self._determine_regime(volatility, correlation, trend)
        
        # Calcola probabilità dei regimi
        probabilities = self._calculate_regime_probabilities(
            volatility, correlation, trend)
            
        # Calcola confidenza
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
        """Calcola la volatilità realizzata"""
        returns = data['close'].pct_change()
        return returns.std() * np.sqrt(252) * 100
        
    def _calculate_correlation(self, data: pd.DataFrame) -> float:
        """Calcola la correlazione cross-asset"""
        if 'other_assets' in data.columns:
            corr_matrix = data[['close', 'other_assets']].corr()
            return corr_matrix.iloc[0, 1]
        return 0.0
        
    def _calculate_trend(self, data: pd.DataFrame) -> float:
        """Calcola la forza del trend"""
        prices = data['close']
        sma_fast = prices.rolling(50).mean()
        sma_slow = prices.rolling(200).mean()
        
        trend_strength = (sma_fast.iloc[-1] / sma_slow.iloc[-1] - 1)
        return np.clip(trend_strength, -1, 1)
        
    def _determine_regime(self, volatility: float, 
                         correlation: float, trend: float) -> MarketRegime:
        """Determina il regime di mercato"""
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
        """Calcola probabilità per ogni regime"""
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
            
        # Normalizza probabilità
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
            
        return probs
        
    def _calculate_confidence(self, probabilities: Dict[str, float]) -> float:
        """Calcola la confidenza nella determinazione del regime"""
        # Maggiore confidenza se una probabilità domina le altre
        max_prob = max(probabilities.values())
        second_max = sorted(probabilities.values())[-2]
        
        # Differenza tra la probabilità più alta e la seconda più alta
        prob_diff = max_prob - second_max
        
        return np.clip(prob_diff * 2, 0, 1)  # Scala 0-1
        
    def _regime_to_value(self, regime: MarketRegime) -> float:
        """Converte il regime in un valore numerico"""
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
