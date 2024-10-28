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
        """Inizializza l'analizzatore della volatilità"""
        # Parametri base
        self.lookback_window = self.config.get('volatility.lookback_window', 252)
        self.forecast_horizon = self.config.get('volatility.forecast_horizon', 5)
        self.min_history = self.config.get('volatility.min_history', 30)
        
        # Thresholds per regimi di volatilità
        self.vol_thresholds = {
            'low': self.config.get('volatility.thresholds.low', 10),
            'medium': self.config.get('volatility.thresholds.medium', 20),
            'high': self.config.get('volatility.thresholds.high', 30),
            'extreme': self.config.get('volatility.thresholds.extreme', 40)
        }
        
        # Modelli di volatilità
        self.models = {
            'garch': None,  # GARCH model
            'ewma': None,   # EWMA model
            'range': None   # Range-based model
        }
        
        # Cache per risultati
        self.cache = {}
        self.last_update = None

    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        """Esegue l'analisi completa della volatilità"""
        if not self.validate_data(data):
            raise ValueError("Invalid data for volatility analysis")

        # Calcola diverse misure di volatilità
        realized_vol = self._calculate_realized_volatility(data)
        implied_vol = self._calculate_implied_volatility(data)
        range_vol = self._calculate_range_volatility(data)
        
        # Aggiorna modelli di volatilità
        self._update_volatility_models(data)
        
        # Genera previsioni
        forecasts = self._generate_volatility_forecasts(data)
        
        # Identifica regime corrente
        current_regime = self._identify_volatility_regime(
            realized_vol, implied_vol, range_vol)
        
        # Analizza pattern di volatilità
        patterns = self._analyze_volatility_patterns(data)
        
        # Calcola segnale composito
        signal = self._calculate_volatility_signal(
            realized_vol,
            implied_vol,
            range_vol,
            forecasts,
            current_regime,
            patterns
        )
        
        # Calcola confidenza
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
        """Calcola la volatilità realizzata usando diversi metodi"""
        returns = data['close'].pct_change().dropna()
        
        # Volatilità standard
        std_vol = returns.std() * np.sqrt(252)
        
        # Volatilità rolling
        rolling_vol = returns.rolling(
            window=self.lookback_window).std() * np.sqrt(252)
        
        # Volatilità parkinson (high-low based)
        hl_vol = self._calculate_parkinson_volatility(data)
        
        # Volatilità yang-zhang (OHLC based)
        yz_vol = self._calculate_yang_zhang_volatility(data)
        
        return {
            'current': std_vol,
            'rolling': rolling_vol,
            'parkinson': hl_vol,
            'yang_zhang': yz_vol,
            'zscore': self._calculate_volatility_zscore(std_vol, rolling_vol)
        }

    def _calculate_implied_volatility(self, data: pd.DataFrame) -> Dict:
        """Calcola e analizza la volatilità implicita"""
        # Se disponibili dati di opzioni
        if 'vix' in data.columns:
            vix = data['vix']
            vix_term_structure = self._analyze_vix_term_structure(data)
            skew = self._calculate_volatility_skew(data)
        else:
            # Usa proxy della volatilità
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
        """Calcola volatilità basata su range di prezzo"""
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
        """Aggiorna i modelli di volatilità"""
        returns = data['close'].pct_change().dropna()
        
        # Aggiorna GARCH model
        self.models['garch'] = arch_model(
            returns,
            vol='Garch',
            p=1,
            q=1
        ).fit(disp='off')
        
        # Aggiorna EWMA
        self.models['ewma'] = self._calculate_ewma_volatility(returns)
        
        # Aggiorna Range-based model
        self.models['range'] = self._update_range_model(data)

    def _generate_volatility_forecasts(self, data: pd.DataFrame) -> Dict:
        """Genera previsioni di volatilità"""
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
        
        # Combina le previsioni
        forecasts['combined'] = self._combine_volatility_forecasts(forecasts)
        
        return forecasts

    def _identify_volatility_regime(self,
                                  realized_vol: Dict,
                                  implied_vol: Dict,
                                  range_vol: Dict) -> VolatilityRegime:
        """Identifica il regime corrente di volatilità"""
        # Calcola volatilità composita
        composite_vol = (
            realized_vol['current'] * 0.4 +
            implied_vol['current'] * 0.4 +
            range_vol['current'] * 0.2
        )
        
        # Determina il regime
        for level, threshold in sorted(
            self.vol_thresholds.items(),
            key=lambda x: x[1]
        ):
            if composite_vol <= threshold:
                break
        
        # Calcola persistenza
        persistence = self._calculate_regime_persistence(composite_vol)
        
        # Genera previsione per il regime
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
        """Analizza pattern nella volatilità"""
        vol = self._calculate_realized_volatility(data)['rolling']
        
        # Identifica cicli di volatilità
        cycles = self._identify_volatility_cycles(vol)
        
        # Trova cluster di volatilità
        clusters = self._identify_volatility_clusters(vol)
        
        # Analizza asimmetrie
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
        """Calcola il segnale finale di volatilità"""
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
        """Calcola la confidenza nella analisi della volatilità"""
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
        """Ritorna le colonne richieste per l'analisi"""
        return ['open', 'high', 'low', 'close', 'volume']

    def get_volatility_summary(self) -> Dict:
        """Fornisce un sommario della volatilità corrente"""
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
