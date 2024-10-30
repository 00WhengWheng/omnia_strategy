from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import talib as ta
from scipy import stats
import logging
from ..base import BaseAnalyzer

@dataclass
class TechnicalState:
    timestamp: datetime
    trend: Dict[str, any]        # Trend information
    momentum: Dict[str, float]   # Momentum indicators
    volatility: Dict[str, float] # Volatility measures
    support_resistance: Dict     # Support/Resistance levels
    patterns: List[Dict]         # Detected patterns
    signals: Dict[str, float]    # Generated signals
    volume_analysis: Dict        # Volume analysis
    confidence: float            # Analysis confidence
    metadata: Dict              # Additional information

class TechnicalAnalyzer(BaseAnalyzer):
    """Technical Analysis Component"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Trend Configuration
        self.trend_periods = self.config.get('trend_periods', [20, 50, 200])
        self.trend_methods = self.config.get('trend_methods', ['sma', 'ema'])
        
        # Momentum Configuration
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_params = self.config.get('macd', {
            'fast': 12,
            'slow': 26,
            'signal': 9
        })
        self.stoch_params = self.config.get('stochastic', {
            'k': 14,
            'd': 3,
            'slow_d': 3
        })
        
        # Volatility Configuration
        self.atr_period = self.config.get('atr_period', 14)
        self.bb_params = self.config.get('bollinger', {
            'period': 20,
            'std_dev': 2
        })
        
        # Pattern Configuration
        self.pattern_recognition = self.config.get('pattern_recognition', True)
        self.min_pattern_quality = self.config.get('min_pattern_quality', 70)

    def analyze(self, data: pd.DataFrame) -> TechnicalState:
        """
        Perform comprehensive technical analysis
        
        Parameters:
        - data: DataFrame with OHLCV data
        
        Returns:
        - TechnicalState object containing analysis results
        """
        try:
            # Validate input data
            self._validate_data(data)
            
            # Analyze trends
            trend_analysis = self._analyze_trends(data)
            
            # Analyze momentum
            momentum_analysis = self._analyze_momentum(data)
            
            # Analyze volatility
            volatility_analysis = self._analyze_volatility(data)
            
            # Find support/resistance levels
            support_resistance = self._find_support_resistance(data)
            
            # Identify patterns
            patterns = self._identify_patterns(data) if self.pattern_recognition else []
            
            # Analyze volume
            volume_analysis = self._analyze_volume(data)
            
            # Generate signals
            signals = self._generate_signals(
                trend_analysis,
                momentum_analysis,
                volatility_analysis,
                support_resistance,
                patterns
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                trend_analysis,
                momentum_analysis,
                volatility_analysis,
                patterns,
                volume_analysis
            )
            
            # Generate metadata
            metadata = self._generate_metadata(
                data, trend_analysis, momentum_analysis,
                volatility_analysis, patterns
            )
            
            return TechnicalState(
                timestamp=datetime.now(),
                trend=trend_analysis,
                momentum=momentum_analysis,
                volatility=volatility_analysis,
                support_resistance=support_resistance,
                patterns=patterns,
                signals=signals,
                volume_analysis=volume_analysis,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            raise

    def _analyze_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze market trends using multiple methods"""
        trends = {}
        
        # Calculate moving averages
        for period in self.trend_periods:
            if 'sma' in self.trend_methods:
                trends[f'sma_{period}'] = ta.SMA(data['close'], timeperiod=period)
            if 'ema' in self.trend_methods:
                trends[f'ema_{period}'] = ta.EMA(data['close'], timeperiod=period)
        
        # Determine trend direction and strength
        current_price = data['close'].iloc[-1]
        trend_scores = []
        
        for period in self.trend_periods:
            sma = trends.get(f'sma_{period}')
            ema = trends.get(f'ema_{period}')
            
            if sma is not None and not pd.isna(sma.iloc[-1]):
                score = (current_price - sma.iloc[-1]) / sma.iloc[-1]
                trend_scores.append(score)
                
            if ema is not None and not pd.isna(ema.iloc[-1]):
                score = (current_price - ema.iloc[-1]) / ema.iloc[-1]
                trend_scores.append(score)
        
        # ADX for trend strength
        adx = ta.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        
        return {
            'direction': 'up' if np.mean(trend_scores) > 0 else 'down',
            'strength': abs(np.mean(trend_scores)),
            'adx': adx.iloc[-1],
            'moving_averages': trends,
            'trend_scores': trend_scores,
            'price_position': self._analyze_price_position(data, trends)
        }

    def _analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """Analyze market momentum"""
        # RSI
        rsi = ta.RSI(data['close'], timeperiod=self.rsi_period)
        
        # MACD
        macd, macd_signal, macd_hist = ta.MACD(
            data['close'],
            fastperiod=self.macd_params['fast'],
            slowperiod=self.macd_params['slow'],
            signalperiod=self.macd_params['signal']
        )
        
        # Stochastic
        slowk, slowd = ta.STOCH(
            data['high'],
            data['low'],
            data['close'],
            fastk_period=self.stoch_params['k'],
            slowk_period=self.stoch_params['d'],
            slowk_matype=0,
            slowd_period=self.stoch_params['slow_d']
        )
        
        return {
            'rsi': {
                'value': rsi.iloc[-1],
                'trend': self._analyze_indicator_trend(rsi),
                'divergence': self._check_divergence(data['close'], rsi)
            },
            'macd': {
                'macd': macd.iloc[-1],
                'signal': macd_signal.iloc[-1],
                'histogram': macd_hist.iloc[-1],
                'trend': self._analyze_indicator_trend(macd_hist)
            },
            'stochastic': {
                'k': slowk.iloc[-1],
                'd': slowd.iloc[-1],
                'trend': self._analyze_indicator_trend(slowk)
            }
        }

    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """Analyze market volatility"""
        # ATR
        atr = ta.ATR(data['high'], data['low'], data['close'], 
                     timeperiod=self.atr_period)
        
        # Bollinger Bands
        upper, middle, lower = ta.BBANDS(
            data['close'],
            timeperiod=self.bb_params['period'],
            nbdevup=self.bb_params['std_dev'],
            nbdevdn=self.bb_params['std_dev']
        )
        
        # Calculate historical volatility
        returns = np.log(data['close'] / data['close'].shift(1))
        hist_vol = returns.rolling(window=21).std() * np.sqrt(252)
        
        return {
            'atr': {
                'value': atr.iloc[-1],
                'trend': self._analyze_indicator_trend(atr)
            },
            'bollinger': {
                'upper': upper.iloc[-1],
                'middle': middle.iloc[-1],
                'lower': lower.iloc[-1],
                'width': (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
            },
            'historical': {
                'value': hist_vol.iloc[-1],
                'percentile': stats.percentileofscore(hist_vol.dropna(), hist_vol.iloc[-1])
            }
        }

    def _find_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        levels = {
            'support': [],
            'resistance': [],
            'dynamic': {}
        }
        
        # Find local minima/maxima
        high_peaks = self._find_peaks(data['high'])
        low_peaks = self._find_peaks(-data['low'])
        
        # Cluster levels
        resistance_levels = self._cluster_levels(high_peaks)
        support_levels = self._cluster_levels(low_peaks)
        
        # Calculate dynamic levels (moving averages, etc.)
        for period in self.trend_periods:
            ma = ta.SMA(data['close'], timeperiod=period)
            levels['dynamic'][f'ma_{period}'] = ma.iloc[-1]
        
        # Sort and filter levels
        levels['resistance'] = sorted(resistance_levels)
        levels['support'] = sorted(support_levels)
        
        return levels

    def _identify_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Identify chart patterns"""
        patterns = []
        
        # Candlestick patterns
        for pattern_func in self._get_pattern_functions():
            result = pattern_func(
                data['open'],
                data['high'],
                data['low'],
                data['close']
            )
            if result[-1] != 0:
                patterns.append({
                    'type': pattern_func.__name__,
                    'direction': 'bullish' if result[-1] > 0 else 'bearish',
                    'strength': abs(result[-1]),
                    'location': len(data) - 1
                })
        
        # Chart patterns (head and shoulders, triangles, etc.)
        chart_patterns = self._identify_chart_patterns(data)
        patterns.extend(chart_patterns)
        
        return patterns

    def _generate_signals(self, trend: Dict, momentum: Dict,
                         volatility: Dict, levels: Dict,
                         patterns: List[Dict]) -> Dict:
        """Generate trading signals"""
        signals = {}
        
        # Trend signals
        signals['trend'] = self._generate_trend_signals(trend)
        
        # Momentum signals
        signals['momentum'] = self._generate_momentum_signals(momentum)
        
        # Volatility signals
        signals['volatility'] = self._generate_volatility_signals(volatility)
        
        # Pattern signals
        signals['patterns'] = self._generate_pattern_signals(patterns)
        
        # Combine signals
        signals['composite'] = self._combine_signals(signals)
        
        return signals

    def _calculate_confidence(self, trend: Dict, momentum: Dict,
                            volatility: Dict, patterns: List,
                            volume: Dict) -> float:
        """Calculate confidence in technical analysis"""
        # Weight different components
        weights = {
            'trend': 0.3,
            'momentum': 0.25,
            'volatility': 0.15,
            'patterns': 0.15,
            'volume': 0.15
        }
        
        # Calculate individual confidences
        confidences = {
            'trend': self._calculate_trend_confidence(trend),
            'momentum': self._calculate_momentum_confidence(momentum),
            'volatility': self._calculate_volatility_confidence(volatility),
            'patterns': self._calculate_pattern_confidence(patterns),
            'volume': self._calculate_volume_confidence(volume)
        }
        
        # Calculate weighted average
        total_confidence = sum(
            confidences[k] * weights[k] for k in weights.keys()
        )
        
        return np.clip(total_confidence, 0, 1)

    @property
    def required_columns(self) -> List[str]:
        """Required data columns"""
        return ['open', 'high', 'low', 'close', 'volume']

    def get_analysis_summary(self) -> Dict:
        """Get summary of current technical state"""
        if not self.analysis_history:
            return {}
            
        latest = self.analysis_history[-1]
        return {
            'timestamp': latest.timestamp,
            'trend': latest.trend['direction'],
            'momentum': latest.momentum['rsi']['value'],
            'volatility': latest.volatility['atr']['value'],
            'signals': latest.signals['composite'],
            'confidence': latest.confidence
        }
