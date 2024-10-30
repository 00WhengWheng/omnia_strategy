from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import talib as ta
from abc import ABC, abstractmethod
import logging

@dataclass
class TechnicalState:
    trend: str  # 'uptrend', 'downtrend', 'sideways'
    strength: float  # 0 to 1
    support_levels: List[float]
    resistance_levels: List[float]
    key_levels: Dict[str, float]
    patterns: List[Dict]
    indicators: Dict[str, float]
    divergences: List[Dict]
    confidence: float

class TechnicalAnalyzer:
    """Technical Analysis Component"""
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.trend_periods = config.get('trend_periods', [20, 50, 200])
        self.volume_ma_period = config.get('volume_ma_period', 20)
        self.rsi_period = config.get('rsi_period', 14)
        self.macd_params = config.get('macd_params', {'fast': 12, 'slow': 26, 'signal': 9})
        
        # Analysis cache
        self.analysis_history = []
        
    def analyze(self, data: pd.DataFrame) -> Dict:
        """Perform technical analysis"""
        try:
            # Trend Analysis
            trend_analysis = self._analyze_trend(data)
            
            # Support/Resistance Levels
            levels = self._identify_key_levels(data)
            
            # Pattern Recognition
            patterns = self._identify_patterns(data)
            
            # Technical Indicators
            indicators = self._calculate_indicators(data)
            
            # Divergence Analysis
            divergences = self._identify_divergences(data, indicators)
            
            # Volume Analysis
            volume_analysis = self._analyze_volume(data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                trend_analysis, patterns, indicators, volume_analysis)
            
            return TechnicalState(
                trend=trend_analysis['trend'],
                strength=trend_analysis['strength'],
                support_levels=levels['support'],
                resistance_levels=levels['resistance'],
                key_levels=levels['key_levels'],
                patterns=patterns,
                indicators=indicators,
                divergences=divergences,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            raise

    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze market trend"""
        trends = {}
        strengths = []
        
        for period in self.trend_periods:
            # Calculate moving averages
            ma = ta.SMA(data['close'], timeperiod=period)
            ema = ta.EMA(data['close'], timeperiod=period)
            
            # Calculate trend direction
            current_price = data['close'].iloc[-1]
            ma_current = ma.iloc[-1]
            ma_prev = ma.iloc[-2]
            
            # Determine trend
            if current_price > ma_current and ma_current > ma_prev:
                trends[period] = 'uptrend'
                strengths.append(abs(current_price - ma_current) / ma_current)
            elif current_price < ma_current and ma_current < ma_prev:
                trends[period] = 'downtrend'
                strengths.append(-abs(current_price - ma_current) / ma_current)
            else:
                trends[period] = 'sideways'
                strengths.append(0)
        
        # Calculate overall trend
        trend_strength = np.mean(strengths)
        if abs(trend_strength) < 0.02:
            overall_trend = 'sideways'
        else:
            overall_trend = 'uptrend' if trend_strength > 0 else 'downtrend'
        
        return {
            'trend': overall_trend,
            'strength': abs(trend_strength),
            'trends_by_period': trends,
            'strength_by_period': dict(zip(self.trend_periods, strengths))
        }

    def _identify_key_levels(self, data: pd.DataFrame) -> Dict:
        """Identify key price levels"""
        # Calculate pivot points
        pivots = self._calculate_pivot_points(data)
        
        # Find support levels
        support_levels = self._find_support_levels(data)
        
        # Find resistance levels
        resistance_levels = self._find_resistance_levels(data)
        
        # Identify key psychological levels
        psych_levels = self._identify_psychological_levels(data)
        
        # Volume profile levels
        volume_levels = self._analyze_volume_profile(data)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'pivots': pivots,
            'psychological': psych_levels,
            'volume_levels': volume_levels,
            'key_levels': {
                'major_support': self._get_major_level(support_levels),
                'major_resistance': self._get_major_level(resistance_levels)
            }
        }

    def _identify_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Identify chart patterns"""
        patterns = []
        
        # Candlestick patterns
        candle_patterns = self._identify_candlestick_patterns(data)
        patterns.extend(candle_patterns)
        
        # Chart patterns
        chart_patterns = self._identify_chart_patterns(data)
        patterns.extend(chart_patterns)
        
        # Harmonic patterns
        harmonic_patterns = self._identify_harmonic_patterns(data)
        patterns.extend(harmonic_patterns)
        
        return [{
            'type': pattern['type'],
            'confidence': pattern['confidence'],
            'start_idx': pattern['start_idx'],
            'end_idx': pattern['end_idx'],
            'target': pattern.get('target'),
            'stop_loss': pattern.get('stop_loss')
        } for pattern in patterns]

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        indicators = {}
        
        # Momentum indicators
        indicators['rsi'] = ta.RSI(data['close'], timeperiod=self.rsi_period)
        
        # Trend indicators
        macd, macdsignal, macdhist = ta.MACD(
            data['close'],
            fastperiod=self.macd_params['fast'],
            slowperiod=self.macd_params['slow'],
            signalperiod=self.macd_params['signal']
        )
        indicators['macd'] = {
            'macd': macd,
            'signal': macdsignal,
            'histogram': macdhist
        }
        
        # Volatility indicators
        indicators['bbands'] = {
            'upper': ta.BBANDS(data['close'])[0],
            'middle': ta.BBANDS(data['close'])[1],
            'lower': ta.BBANDS(data['close'])[2]
        }
        
        # Volume indicators
        indicators['obv'] = ta.OBV(data['close'], data['volume'])
        indicators['adl'] = ta.AD(data['high'], data['low'], 
                                data['close'], data['volume'])
        
        return indicators

    def _identify_divergences(self, data: pd.DataFrame, 
                            indicators: Dict) -> List[Dict]:
        """Identify technical divergences"""
        divergences = []
        
        # RSI Divergences
        rsi_div = self._check_rsi_divergence(data, indicators['rsi'])
        if rsi_div:
            divergences.extend(rsi_div)
        
        # MACD Divergences
        macd_div = self._check_macd_divergence(
            data, indicators['macd']['macd'])
        if macd_div:
            divergences.extend(macd_div)
        
        # Volume Divergences
        vol_div = self._check_volume_divergence(data)
        if vol_div:
            divergences.extend(vol_div)
        
        return divergences

    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        volume = data['volume']
        close = data['close']
        
        # Calculate volume moving average
        volume_ma = volume.rolling(window=self.volume_ma_period).mean()
        
        # Calculate volume trend
        volume_trend = 'increasing' if volume.iloc[-1] > volume_ma.iloc[-1] else 'decreasing'
        
        # Calculate price/volume correlation
        price_vol_corr = close.corr(volume)
        
        # Identify volume spikes
        volume_std = volume.std()
        spikes = volume[volume > volume.mean() + 2 * volume_std]
        
        return {
            'trend': volume_trend,
            'price_correlation': price_vol_corr,
            'spikes': spikes,
            'current_ratio': volume.iloc[-1] / volume_ma.iloc[-1]
        }

    def _calculate_confidence(self, trend_analysis: Dict,
                            patterns: List,
                            indicators: Dict,
                            volume_analysis: Dict) -> float:
        """Calculate confidence in technical analysis"""
        # Trend confidence
        trend_conf = abs(trend_analysis['strength'])
        
        # Pattern confidence
        pattern_conf = np.mean([p['confidence'] for p in patterns]) if patterns else 0.5
        
        # Indicator confidence
        indicator_conf = self._calculate_indicator_confidence(indicators)
        
        # Volume confidence
        volume_conf = self._calculate_volume_confidence(volume_analysis)
        
        # Weight components
        confidence = (
            trend_conf * 0.35 +
            pattern_conf * 0.25 +
            indicator_conf * 0.25 +
            volume_conf * 0.15
        )
        
        return np.clip(confidence, 0, 1)

    def _calculate_indicator_confidence(self, indicators: Dict) -> float:
        """Calculate confidence based on indicators"""
        confidences = []
        
        # RSI
        rsi = indicators['rsi'].iloc[-1]
        if rsi > 70 or rsi < 30:
            confidences.append(0.8)
        else:
            confidences.append(0.5)
        
        # MACD
        macd = indicators['macd']['macd'].iloc[-1]
        macd_signal = indicators['macd']['signal'].iloc[-1]
        if abs(macd - macd_signal) > 0.5:
            confidences.append(0.7)
        else:
            confidences.append(0.5)
        
        return np.mean(confidences)

    def _calculate_volume_confidence(self, volume_analysis: Dict) -> float:
        """Calculate confidence based on volume"""
        if volume_analysis['price_correlation'] > 0.7:
            base_conf = 0.8
        elif volume_analysis['price_correlation'] > 0.5:
            base_conf = 0.6
        else:
            base_conf = 0.4
            
        # Adjust for volume trend
        if volume_analysis['trend'] == 'increasing':
            base_conf *= 1.2
            
        return np.clip(base_conf, 0, 1)
