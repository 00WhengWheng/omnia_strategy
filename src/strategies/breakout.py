from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import talib as ta
from .base import BaseStrategy, StrategySignal
from ..core.constants import TimeFrame
import logging

class BreakoutStrategy(BaseStrategy):
    def _initialize_strategy(self) -> None:
        """Initialize the breakout strategy."""
        required_keys = [
            'breakout_period', 'confirmation_period', 'volatility_lookback',
            'pivot_points_lookback', 'volume_threshold', 'volatility_threshold',
            'min_consolidation_bars', 'atr_multiplier', 'partial_profit_levels'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Parameters for breakout detection
        self.breakout_period = self.config.get('breakout_period', 20)
        self.confirmation_period = self.config.get('confirmation_period', 3)
        self.volatility_lookback = self.config.get('volatility_lookback', 20)
        
        # Key levels for breakout detection
        self.support_resistance_periods = [20, 50, 200]  # Multiple timeframes
        self.pivot_points_lookback = self.config.get('pivot_points_lookback', 10)
        self.fibonacci_levels = self.config.get('fibonacci_levels', True)
        
        # Volume and volatility thresholds
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        self.volatility_threshold = self.config.get('volatility_threshold', 1.2)
        self.min_consolidation_bars = self.config.get('min_consolidation_bars', 5)
        self.false_breakout_filter = self.config.get('false_breakout_filter', True)
        
        # Risk Management
        self.atr_multiplier = self.config.get('atr_multiplier', 1.5)
        self.trailing_stop = self.config.get('trailing_stop', True)
        self.partial_profit_levels = self.config.get('partial_profit_levels', [2.0, 3.0])
        
        # Performance Tracking
        self.breakout_history = []
        self.false_breakout_count = 0
        self.successful_breakout_count = 0

        self._initialize_logging()

    def _initialize_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized BreakoutStrategy")

    def generate_signals(self, data: pd.DataFrame) -> StrategySignal:
        """Generate trading signals based on the data provided."""
        try:
            self.logger.info("Generating signals")
            # Identify key levels
            levels = self._identify_key_levels(data)
            
            # Calculate indicators
            indicators = self._calculate_indicators(data)
            
            # Identify breakout
            breakout = self._identify_breakout(data, levels, indicators)
            
            # Check filters
            if not self._check_filters(data, breakout, indicators):
                return self._generate_neutral_signal(data)
            
            # Generate signal based on breakout
            if breakout['valid']:
                signal = self._generate_breakout_signal(data, breakout, levels, indicators)
                # Update breakout statistics
                self._update_breakout_statistics(breakout, signal)
            else:
                signal = self._generate_neutral_signal(data)
            
            self.logger.debug(f"Generated signal: {signal}")
            return signal
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return self._generate_neutral_signal(data)

    def _identify_key_levels(self, data: pd.DataFrame) -> Dict:
        """Identify key levels for breakout detection."""
        # Support & Resistance levels
        sr_levels = self._calculate_support_resistance(data)
        
        # Pivot Points
        pivots = self._calculate_pivot_points(data)
        
        # Range of consolidation
        consolidation = self._identify_consolidation_range(data)
        
        # Levels based on Fibonacci retracement
        fib_levels = self._calculate_fibonacci_levels(data) if self.fibonacci_levels else None
        
        # Volume Profile levels
        volume_levels = self._calculate_volume_profile_levels(data)
        
        return {
            'support_resistance': sr_levels,
            'pivots': pivots,
            'consolidation': consolidation,
            'fibonacci': fib_levels,
            'volume_levels': volume_levels
        }

    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels."""
        levels = {}
        
        for period in self.support_resistance_periods:
            # Calculate rolling high and low levels
            high_levels = data['high'].rolling(period).max()
            low_levels = data['low'].rolling(period).min()
            
            # Find clusters of price levels
            high_clusters = self._cluster_price_levels(high_levels)
            low_clusters = self._cluster_price_levels(low_levels)
            
            levels[period] = {
                'resistance': high_clusters,
                'support': low_clusters
            }
            
        return levels

    def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict:
        """Calculate pivot points."""
        pivots = {}
        
        high = data['high'].iloc[-self.pivot_points_lookback:]
        low = data['low'].iloc[-self.pivot_points_lookback:]
        close = data['close'].iloc[-self.pivot_points_lookback:]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        
        pivots['classic'] = {
            'p': pivot.iloc[-1],
            'r1': r1.iloc[-1],
            'r2': r2.iloc[-1],
            's1': s1.iloc[-1],
            's2': s2.iloc[-1]
        }
        
        return pivots

    def _identify_consolidation_range(self, data: pd.DataFrame) -> Dict:
        """Identify consolidation range."""
        # Calculate volatility
        volatility = data['close'].rolling(self.volatility_lookback).std()
        avg_volatility = volatility.mean()
        
        # Check for low volatility periods
        low_vol_periods = volatility < (avg_volatility * 0.8)
        
        # Check for recent consolidation
        if low_vol_periods.iloc[-self.min_consolidation_bars:].all():
            recent_data = data.iloc[-self.min_consolidation_bars:]
            
            return {
                'valid': True,
                'high': recent_data['high'].max(),
                'low': recent_data['low'].min(),
                'duration': self.min_consolidation_bars,
                'avg_volume': recent_data['volume'].mean()
            }
            
        return {'valid': False}

    def _calculate_volume_profile_levels(self, data: pd.DataFrame) -> Dict:
        """Calculate volume profile levels."""
        # Calculate price range
        price_range = np.linspace(
            data['low'].min(),
            data['high'].max(),
            50  # number of bins
        )
        
        volume_profile = {}
        for i in range(len(price_range)-1):
            mask = (data['close'] >= price_range[i]) & (data['close'] < price_range[i+1])
            volume_profile[price_range[i]] = data.loc[mask, 'volume'].sum()
            
        # Find top volume levels
        sorted_levels = sorted(
            volume_profile.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'high_volume_levels': [level[0] for level in sorted_levels[:5]],
            'volume_profile': volume_profile
        }

    def _identify_breakout(self, data: pd.DataFrame,
                          levels: Dict,
                          indicators: Dict) -> Dict:
        """Identify breakout based on key levels."""
        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Check resistance and support breakouts
        resistance_break = self._check_resistance_breakout(
            current_price, levels, indicators)
        support_break = self._check_support_breakout(
            current_price, levels, indicators)
        
        # Check for valid breakout
        if resistance_break['valid'] or support_break['valid']:
            breakout_strength = self._calculate_breakout_strength(
                data, current_price, current_volume, indicators)
            
            # Determine breakout direction
            if resistance_break['valid']:
                direction = 'long'
                level_broken = resistance_break['level']
                distance = abs(current_price - level_broken)
            else:
                direction = 'short'
                level_broken = support_break['level']
                distance = abs(current_price - level_broken)
            
            return {
                'valid': True,
                'direction': direction,
                'strength': breakout_strength,
                'level_broken': level_broken,
                'distance': distance,
                'volume_surge': current_volume / indicators['volume_sma'].iloc[-1]
            }
            
        return {'valid': False}

    def _calculate_breakout_strength(self, data: pd.DataFrame,
                                   current_price: float,
                                   current_volume: float,
                                   indicators: Dict) -> float:
        """Calculate breakout strength."""
        # Calculate price momentum
        price_momentum = self._calculate_momentum_score(data)
        volume_strength = current_volume / indicators['volume_sma'].iloc[-1]
        volatility_ratio = indicators['atr'].iloc[-1] / indicators['atr'].rolling(20).mean().iloc[-1]
        
        # Combine components
        strength = (
            price_momentum * 0.4 +
            min(volume_strength, 3.0) / 3.0 * 0.4 +
            min(volatility_ratio, 2.0) / 2.0 * 0.2
        )
        
        return np.clip(strength, 0, 1)

    def _generate_breakout_signal(self, data: pd.DataFrame,
                                breakout: Dict,
                                levels: Dict,
                                indicators: Dict) -> StrategySignal:
        """Generate breakout signal."""
        current_price = data['close'].iloc[-1]
        atr = indicators['atr'].iloc[-1]
        
        # Generate stop loss and target levels
        if breakout['direction'] == 'long':
            next_level = self._find_next_resistance(current_price, levels)
            stop_loss = current_price - (atr * self.atr_multiplier)
        else:
            next_level = self._find_next_support(current_price, levels)
            stop_loss = current_price + (atr * self.atr_multiplier)
            
        # Generate target levels
        targets = self._calculate_breakout_targets(
            current_price,
            next_level,
            breakout['direction']
        )
        
        # Calculate confidence
        confidence = self._calculate_breakout_confidence(
            breakout, indicators)
        
        return StrategySignal(
            timestamp=datetime.now(),
            direction=breakout['direction'],
            strength=breakout['strength'],
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            targets=targets,
            timeframe=self.timeframe,
            metadata={
                'breakout_level': breakout['level_broken'],
                'volume_surge': breakout['volume_surge'],
                'next_level': next_level,
                'atr': atr
            }
        )

    def _check_filters(self, data: pd.DataFrame,
                      breakout: Dict,
                      indicators: Dict) -> bool:
        """Check breakout filters."""
        if not breakout['valid']:
            return False
            
        # Volume filter
        if breakout['volume_surge'] < self.volume_threshold:
            return False
            
        # Volatility filter
        if indicators['atr'].iloc[-1] < \
           (indicators['atr'].rolling(20).mean().iloc[-1] * self.volatility_threshold):
            return False
            
        # False breakout filter
        if self.false_breakout_filter and \
           self._detect_false_breakout_pattern(data, breakout):
            return False
            
        return True

    def _detect_false_breakout_pattern(self, data: pd.DataFrame,
                                     breakout: Dict) -> bool:
        """Detect false breakout patterns."""
        # Check for recent consolidation
        recent_close = data['close'].iloc[-1]
        recent_high = data['high'].iloc[-1]
        recent_low = data['low'].iloc[-1]
        
        if breakout['direction'] == 'long':
            # Check for bearish engulfing pattern
            return recent_close < breakout['level_broken']
        else:
            # Check for bullish engulfing pattern
            return recent_close > breakout['level_broken']

    def _update_breakout_statistics(self, breakout: Dict,
                                  signal: StrategySignal) -> None:
        """Update breakout statistics."""
        self.breakout_history.append({
            'timestamp': signal.timestamp,
            'direction': breakout['direction'],
            'strength': breakout['strength'],
            'level': breakout['level_broken'],
            'volume_surge': breakout['volume_surge'],
            'confidence': signal.confidence
        })

    def get_required_columns(self) -> List[str]:
        """Get the required columns for the strategy."""
        return ['open', 'high', 'low', 'close', 'volume']

    def get_min_required_history(self) -> int:
        """Get the minimum required history length."""
        return max(self.support_resistance_periods)

    def _apply_strategy_filters(self, signal: StrategySignal,
                              data: pd.DataFrame) -> bool:
        """Apply strategy-specific filters."""
        return signal.confidence >= 0.7  # High confidence