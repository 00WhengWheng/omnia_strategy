from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import pandas_ta as ta
from .base import BaseStrategy, StrategySignal
from ..core.constants import TimeFrame
import logging

class MomentumStrategy(BaseStrategy):
    def _initialize_strategy(self) -> None:
        """Initialize strategy parameters."""
        # RSI Parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_threshold = self.config.get('rsi_threshold', 60)
        
        # MACD Parameters
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        
        # ADX Parameters
        self.adx_period = self.config.get('adx_period', 14)
        self.adx_threshold = self.config.get('adx_threshold', 25)
        
        # Rate of Change
        self.roc_period = self.config.get('roc_period', 10)
        self.roc_threshold = self.config.get('roc_threshold', 5)
        
        # Multi-timeframe settings
        self.timeframes = {
            TimeFrame.D1: 0.4,  # Daily weight
            TimeFrame.H4: 0.3,  # 4H weight
            TimeFrame.H1: 0.3   # 1H weight
        }
        
        # Money Flow
        self.mfi_period = self.config.get('mfi_period', 14)
        self.mfi_threshold = self.config.get('mfi_threshold', 60)
        
        # Filters
        self.min_momentum = self.config.get('min_momentum', 0.3)
        self.volume_filter = self.config.get('volume_filter', True)
        self.volatility_filter = self.config.get('volatility_filter', True)
        self.correlation_filter = self.config.get('correlation_filter', True)
        
        # Risk Management
        self.atr_multiplier = self.config.get('atr_multiplier', 2.0)
        self.trailing_activation = self.config.get('trailing_activation', 1.5)
        self.profit_targets = self.config.get('profit_targets', [2.0, 3.0, 4.0])
        
        # Performance tracking
        self.momentum_history = pd.DataFrame()

        self._initialize_logging()

    def _initialize_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized MomentumStrategy")

    def generate_signals(self, data: pd.DataFrame) -> StrategySignal:
        """Generate momentum signals."""
        try:
            self.logger.info("Generating signals")
            # Calculate indicators
            indicators = self._calculate_momentum_indicators(data)
            
            # Multi-timeframe analysis
            mtf_analysis = self._analyze_multiple_timeframes(data)
            
            # Identify momentum signals
            momentum_signals = self._identify_momentum_signals(data, indicators, mtf_analysis)
            
            # Check filters
            if not self._check_momentum_filters(data, momentum_signals, indicators):
                return self._generate_neutral_signal(data)
            
            # Calculate momentum strength
            momentum_strength = self._calculate_momentum_strength(indicators, momentum_signals)
            
            # Generate signal if valid
            if momentum_signals['valid']:
                signal = self._generate_momentum_signal(data, momentum_signals, indicators, momentum_strength)
                
                # Update statistics
                self._update_momentum_statistics(momentum_signals, signal)
            else:
                signal = self._generate_neutral_signal(data)
            
            self.logger.debug(f"Generated signal: {signal}")
            return signal
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return self._generate_neutral_signal(data)

    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate momentum indicators."""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # RSI
        rsi = ta.RSI(close, timeperiod=self.rsi_period)
        
        # MACD
        macd, macd_signal, macd_hist = ta.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        
        # ADX and DMI
        adx = ta.ADX(high, low, close, timeperiod=self.adx_period)
        plus_di = ta.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        minus_di = ta.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        
        # Rate of Change
        roc = ta.ROC(close, timeperiod=self.roc_period)
        
        # Money Flow Index
        mfi = ta.MFI(high, low, close, volume, timeperiod=self.mfi_period)
        
        # Volatility
        atr = ta.ATR(high, low, close)
        
        return {
            'rsi': {
                'value': rsi,
                'slope': self._calculate_slope(rsi),
                'divergence': self._check_divergence(close, rsi)
            },
            'macd': {
                'macd': macd,
                'signal': macd_signal,
                'hist': macd_hist,
                'histogram_slope': self._calculate_slope(macd_hist)
            },
            'adx': {
                'value': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'trend_strength': self._calculate_trend_strength(adx, plus_di, minus_di)
            },
            'roc': {
                'value': roc,
                'acceleration': self._calculate_acceleration(roc)
            },
            'mfi': {
                'value': mfi,
                'divergence': self._check_divergence(close, mfi)
            },
            'atr': atr
        }

    def _analyze_multiple_timeframes(self, data: pd.DataFrame) -> Dict:
        """Analyze momentum on multiple timeframes."""
        mtf_analysis = {}
        
        for timeframe in self.timeframes:
            # Resample data
            tf_data = self._resample_data(data, timeframe)
            
            # Calculate indicators for this timeframe
            indicators = self._calculate_momentum_indicators(tf_data)
            
            # Analyze momentum for this timeframe
            mtf_analysis[timeframe] = {
                'momentum_score': self._calculate_timeframe_momentum(indicators),
                'trend_alignment': self._check_trend_alignment(indicators),
                'strength': self._calculate_signal_strength(indicators)
            }
            
        return mtf_analysis
        
    def _calculate_timeframe_momentum(self, indicators: Dict) -> float:
        """Calculate momentum score for a single timeframe."""
        # RSI momentum
        rsi_score = (indicators['rsi']['value'].iloc[-1] - 50) / 50
        
        # MACD momentum
        macd_score = indicators['macd']['hist'].iloc[-1] / abs(indicators['macd']['hist']).mean()
        
        # ROC momentum
        roc_score = indicators['roc']['value'].iloc[-1] / self.roc_threshold
        
        # Calculate composite score
        momentum_score = (
            rsi_score * 0.3 +
            macd_score * 0.4 +
            roc_score * 0.3
        )
        
        return np.clip(momentum_score, -1, 1)

    def _check_trend_alignment(self, indicators: Dict) -> float:
        """Check trend alignment."""
        adx = indicators['adx']['value'].iloc[-1]
        plus_di = indicators['adx']['plus_di'].iloc[-1]
        minus_di = indicators['adx']['minus_di'].iloc[-1]
        
        if adx > self.adx_threshold:
            if plus_di > minus_di:
                return 1.0
            else:
                return -1.0
        return 0.0

    def _calculate_signal_strength(self, indicators: Dict) -> float:
        """Calculate signal strength."""
        # ADX strength
        adx_strength = min(indicators['adx']['value'].iloc[-1] / 50, 1.0)
        
        # Volume strength
        volume_strength = min(indicators['mfi']['value'].iloc[-1] / 80, 1.0)
        
        # Momentum consistency
        momentum_cons = self._calculate_momentum_consistency(indicators)
        
        return (adx_strength * 0.4 + 
                volume_strength * 0.3 + 
                momentum_cons * 0.3)

    def _calculate_momentum_consistency(self, indicators: Dict) -> float:
        """Calculate momentum consistency."""
        # Check RSI consistency
        rsi_trend = all(indicators['rsi']['value'].tail(5) > 50)
        
        # Check MACD consistency
        macd_trend = all(indicators['macd']['hist'].tail(5) > 0)
        
        # Check ROC consistency
        roc_trend = all(indicators['roc']['value'].tail(5) > 0)
        
        # Calculate consistency score
        consistency_score = (
            rsi_trend * 0.3 +
            macd_trend * 0.4 +
            roc_trend * 0.3
        )
        
        return consistency_score

    def _check_momentum_filters(self, data: pd.DataFrame,
                              momentum_signals: Dict,
                              indicators: Dict) -> bool:
        """Apply momentum filters."""
        # Check minimum strength
        if abs(momentum_signals['strength']) < self.min_momentum:
            return False
        
        # Check volume
        if self.volume_filter and not self._check_volume_conditions(data):
            return False
        
        # Check volatility
        if self.volatility_filter and not self._check_volatility_conditions(indicators):
            return False
        
        # Check correlation
        if self.correlation_filter and not self._check_correlation_conditions(data):
            return False
        
        return True

    def _calculate_momentum_confidence(self, signals: Dict,
                                    indicators: Dict,
                                    strength: float) -> float:
        """Calculate confidence in the momentum signal."""
        # Trend strength
        trend_confidence = indicators['adx']['trend_strength']
        
        # Signal alignment
        signal_alignment = signals['mtf_consensus']
        
        # Momentum strength
        momentum_conf = strength
        
        # Volume confirmation
        volume_conf = indicators['mfi']['value'].iloc[-1] / 100
        
        # Combine confidence metrics
        confidence = (
            trend_confidence * 0.3 +
            abs(signal_alignment) * 0.3 +
            momentum_conf * 0.2 +
            volume_conf * 0.2
        )
        
        return np.clip(confidence, 0, 1)

    def get_required_columns(self) -> List[str]:
        """Return the columns required by the strategy."""
        return ['open', 'high', 'low', 'close', 'volume']

    def get_min_required_history(self) -> int:
        """Return the minimum required history."""
        return max(self.macd_slow + self.macd_signal,
                  self.rsi_period,
                  self.adx_period) + 50

    def _apply_strategy_filters(self, signal: StrategySignal,
                              data: pd.DataFrame) -> bool:
        """Apply strategy-specific filters."""
        # Check minimum momentum
        if signal.strength < self.min_momentum:
            return False
        
        # Check timeframe consensus
        if abs(signal.metadata['mtf_consensus']) < 0.5:
            return False
        
        return True

    def _update_momentum_statistics(self, signals: Dict,
                                  generated_signal: StrategySignal) -> None:
        """Update momentum statistics."""
        new_stats = {
            'timestamp': generated_signal.timestamp,
            'direction': generated_signal.direction,
            'strength': signals['strength'],
            'confidence': generated_signal.confidence,
            'mtf_consensus': signals['mtf_consensus']
        }
        
        self.momentum_history = pd.concat([
            self.momentum_history,
            pd.DataFrame([new_stats])
        ])