import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import norm
from arch import arch_model
from dataclasses import dataclass
from .base import BaseAnalyzer, AnalysisResult

@dataclass
class VolatilityRegime:
    level: str  # 'low', 'medium', 'high', 'extreme'
    value: float
    persistence: int  # numero di periodi nel regime
    threshold: float
    forecast: float

class VolatilityAnalyzer(BaseAnalyzer):
    def _initialize_analyzer(self) -> None:
        # Initialize the volatility analyzer
        # Configurations
        self.lookback_window = self.config.get('volatility.lookback_window', 252)
        self.forecast_horizon = self.config.get('volatility.forecast_horizon', 5)
        self.min_history = self.config.get('volatility.min_history', 30)
        
        # Volatility thresholds
        self.vol_thresholds = {
            'low': self.config.get('volatility.thresholds.low', 10),
            'medium': self.config.get('volatility.thresholds.medium', 20),
            'high': self.config.get('volatility.thresholds.high', 30),
            'extreme': self.config.get('volatility.thresholds.extreme', 40)
        }
        
        # Volatility models
        self.models = {
            'garch': None,  # GARCH model
            'ewma': None,   # EWMA model
            'range': None   # Range-based model
        }
        
        # Volatility cache
        self.cache = {}
        self.last_update = None

    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        # Execute the volatility analysis
        if not self.validate_data(data):
            raise ValueError("Invalid data for volatility analysis")

        # Get the latest data
        realized_vol = self._calculate_realized_volatility(data)
        implied_vol = self._calculate_implied_volatility(data)
        range_vol = self._calculate_range_volatility(data)
        
        # Update volatility models
        self._update_volatility_models(data)
        
        # Generate volatility forecasts
        forecasts = self._generate_volatility_forecasts(data)
        
        # Identify current volatility regime
        current_regime = self._identify_volatility_regime(
            realized_vol, implied_vol, range_vol)
        
        # Analyze volatility patterns
        patterns = self._analyze_volatility_patterns(data)
        
        # Calculate final signal
        signal = self._calculate_volatility_signal(
            realized_vol,
            implied_vol,
            range_vol,
            forecasts,
            current_regime,
            patterns
        )
        
        # Calculate confidence
        confidence = self._calculate_volatility_confidence(
            realized_vol,
            implied_vol,
            forecasts
        )

        result = AnalysisResult(
            timestamp=datetime.now(),
            value=signal,
            confidence=confidence,
            components={
                'realized_vol': realized_vol['current'],
                'implied_vol': implied_vol['current'],
                'range_vol': range_vol['current'],
                'regime': current_regime.level,
                'forecast': forecasts['combined']
            },
            metadata={
                'realized_volatility': realized_vol,
                'implied_volatility': implied_vol,
                'range_volatility': range_vol,
                'forecasts': forecasts,
                'current_regime': current_regime.__dict__,
                'patterns': patterns
            }
        )

        self.update_history(result)
        return result

    def _calculate_realized_volatility(self, data: pd.DataFrame) -> Dict:
        # Calculate and analyze realized volatility
        returns = data['close'].pct_change().dropna()
        
        # Standard volatility
        std_vol = returns.std() * np.sqrt(252)
        
        # Rolling volatility
        rolling_vol = returns.rolling(
            window=self.lookback_window).std() * np.sqrt(252)
        
        # Parkinson volatility (High-Low based)
        hl_vol = self._calculate_parkinson_volatility(data)
        
        # Yang-Zhang volatility (OHLC based)
        yz_vol = self._calculate_yang_zhang_volatility(data)
        
        return {
            'current': std_vol,
            'rolling': rolling_vol,
            'parkinson': hl_vol,
            'yang_zhang': yz_vol,
            'zscore': self._calculate_volatility_zscore(std_vol, rolling_vol)
        }

    def _calculate_implied_volatility(self, data: pd.DataFrame) -> Dict:
        # Calculate and analyze implied volatility
        # If VIX is available, use it for analysis
        if 'vix' in data.columns:
            vix = data['vix']
            vix_term_structure = self._analyze_vix_term_structure(data)
            skew = self._calculate_volatility_skew(data)
        else:
            # Otherwise, use a proxy for VIX
            vix = self._calculate_volatility_proxy(data)
            vix_term_structure = None
            skew = None
        
        return {
            'current': vix.iloc[-1] if isinstance(vix, pd.Series) else vix,
            'term_structure': vix_term_structure,
            'skew': skew,
            'premium': self._calculate_volatility_premium(vix, data)
        }

    def _calculate_range_volatility(self, data: pd.DataFrame) -> Dict:
        # Calculate and analyze range-based volatility
        # True Range
        tr = self._calculate_true_range(data)
        
        # Average True Range
        atr = tr.rolling(window=14).mean()
        
        # Normalized ATR
        natr = (atr / data['close']) * 100
        
        return {
            'current': atr.iloc[-1],
            'normalized': natr.iloc[-1],
            'true_range': tr.iloc[-1],
            'trend': self._analyze_range_trend(atr)
        }

    def _update_volatility_models(self, data: pd.DataFrame) -> None:
        # Update volatility models
        returns = data['close'].pct_change().dropna()
        
        # Update GARCH model
        self.models['garch'] = arch_model(
            returns,
            vol='Garch',
            p=1,
            q=1
        ).fit(disp='off')
        
        # Update EWMA model
        self.models['ewma'] = self._calculate_ewma_volatility(returns)
        
        # Update range-based model
        self.models['range'] = self._update_range_model(data)

    def _generate_volatility_forecasts(self, data: pd.DataFrame) -> Dict:
        # Generate volatility forecasts
        forecasts = {}
        
        # GARCH forecast
        if self.models['garch'] is not None:
            garch_forecast = self.models['garch'].forecast(
                horizon=self.forecast_horizon)
            forecasts['garch'] = garch_forecast.variance.values[-1]
        
        # EWMA forecast
        if self.models['ewma'] is not None:
            ewma_forecast = self._forecast_ewma_volatility(
                self.models['ewma'], self.forecast_horizon)
            forecasts['ewma'] = ewma_forecast
        
        # Range-based forecast
        if self.models['range'] is not None:
            range_forecast = self._forecast_range_volatility(
                self.models['range'], self.forecast_horizon)
            forecasts['range'] = range_forecast
        
        # Combine forecasts
        forecasts['combined'] = self._combine_volatility_forecasts(forecasts)
        
        return forecasts

    def _identify_volatility_regime(self,
                                  realized_vol: Dict,
                                  implied_vol: Dict,
                                  range_vol: Dict) -> VolatilityRegime:
        # Identify the current volatility regime
        # Generate composite volatility
        composite_vol = (
            realized_vol['current'] * 0.4 +
            implied_vol['current'] * 0.4 +
            range_vol['current'] * 0.2
        )
        
        # Identify regime level
        for level, threshold in sorted(
            self.vol_thresholds.items(),
            key=lambda x: x[1]
        ):
            if composite_vol <= threshold:
                break
        
        # Calculate regime persistence
        persistence = self._calculate_regime_persistence(composite_vol)
        
        # Generate regime forecast
        regime_forecast = self._forecast_regime_transition(
            level, persistence, composite_vol)
        
        return VolatilityRegime(
            level=level,
            value=composite_vol,
            persistence=persistence,
            threshold=threshold,
            forecast=regime_forecast
        )

    def _analyze_volatility_patterns(self, data: pd.DataFrame) -> Dict:
        # Analyze volatility patterns
        vol = self._calculate_realized_volatility(data)['rolling']
        
        # Identify volatility cycles
        cycles = self._identify_volatility_cycles(vol)
        
        # Identify volatility clusters
        clusters = self._identify_volatility_clusters(vol)
        
        # Analyze volatility asymmetry
        asymmetry = self._analyze_volatility_asymmetry(vol, data)
        
        return {
            'cycles': cycles,
            'clusters': clusters,
            'asymmetry': asymmetry,
            'seasonality': self._analyze_volatility_seasonality(vol)
        }

    def _calculate_volatility_signal(self,
                                   realized_vol: Dict,
                                   implied_vol: Dict,
                                   range_vol: Dict,
                                   forecasts: Dict,
                                   regime: VolatilityRegime,
                                   patterns: Dict) -> float:
        # Calculate the final volatility signal
        # Current state assessment
        current_state = self._assess_current_volatility_state(
            realized_vol, implied_vol, range_vol)
        
        # Forward looking component
        forward_looking = self._assess_forward_volatility(
            implied_vol, forecasts)
        
        # Pattern based adjustment
        pattern_adjustment = self._calculate_pattern_adjustment(patterns)
        
        # Regime based scaling
        regime_scalar = self._calculate_regime_scalar(regime)
        
        # Combine components
        raw_signal = (
            current_state * 0.4 +
            forward_looking * 0.4 +
            pattern_adjustment * 0.2
        ) * regime_scalar
        
        return np.clip(raw_signal, -1, 1)

    def _calculate_volatility_confidence(self,
                                      realized_vol: Dict,
                                      implied_vol: Dict,
                                      forecasts: Dict) -> float:
        # Calculate the confidence in the volatility signal
        # Model agreement
        model_agreement = self._calculate_model_agreement(
            realized_vol, implied_vol, forecasts)
        
        # Forecast confidence
        forecast_confidence = self._calculate_forecast_confidence(forecasts)
        
        # Data quality
        data_quality = self._assess_data_quality()
        
        # Combine components
        confidence = (
            model_agreement * 0.4 +
            forecast_confidence * 0.4 +
            data_quality * 0.2
        )
        
        return np.clip(confidence, 0, 1)

    def get_required_columns(self) -> list:
        # Specify the required columns for volatility analysis
        return ['open', 'high', 'low', 'close', 'volume']

    def get_volatility_summary(self) -> Dict:
        # Get the latest current volatility summary
        if self.results_history.empty:
            return {}
            
        latest = self.results_history.iloc[-1]
        
        return {
            'timestamp': latest['timestamp'],
            'current_level': latest['components']['realized_vol'],
            'regime': latest['components']['regime'],
            'forecast': latest['components']['forecast'],
            'confidence': latest['confidence'],
            'patterns': latest['metadata']['patterns']
        }
