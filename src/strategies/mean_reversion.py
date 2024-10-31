from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import pandas_ta as ta
from scipy import stats
from .base import BaseStrategy, StrategySignal
from ..core.constants import TimeFrame

class MeanReversionStrategy(BaseStrategy):
    def _initialize_strategy(self) -> None:
        # Timeframe
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        
        # Parameters for RSI
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        
        # Z-Score settings
        self.zscore_period = self.config.get('zscore_period', 20)
        self.zscore_threshold = self.config.get('zscore_threshold', 2.0)
        
        # Mean Reversion Parameters
        self.lookback_period = self.config.get('lookback_period', 50)
        self.mean_period = self.config.get('mean_period', 20)
        self.std_dev_multiplier = self.config.get('std_dev_multiplier', 2.0)
        self.mean_reversion_threshold = self.config.get('mean_reversion_threshold', 0.8)
        
        # Filtri
        self.min_volatility = self.config.get('min_volatility', 0.1)
        self.max_volatility = self.config.get('max_volatility', 0.4)
        self.volume_filter = self.config.get('volume_filter', True)
        self.trend_filter = self.config.get('trend_filter', True)
        
        # Risk Management
        self.profit_target_stdev = self.config.get('profit_target_stdev', 1.0)
        self.stop_loss_stdev = self.config.get('stop_loss_stdev', 3.0)
        self.max_holding_period = self.config.get('max_holding_period', 10)
        
        # Track statistics
        self.stats_history = pd.DataFrame()

    def generate_signals(self, data: pd.DataFrame) -> StrategySignal:
        # Get required columns
        stats = self._calculate_statistics(data)
        
        # Get technical indicators
        indicators = self._calculate_indicators(data)
        
        # Check mean reversion conditions
        reversion_conditions = self._check_reversion_conditions(stats, indicators)
        
        # Check filters
        if not self._check_filters(data, stats, indicators):
            return self._generate_neutral_signal(data)
        
        # Generate signal
        if reversion_conditions['valid']:
            signal = self._generate_reversion_signal(
                data, stats, indicators, reversion_conditions)
        else:
            signal = self._generate_neutral_signal(data)
            
        # Update statistics
        self._update_statistics(stats, signal)
        
        return signal

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        # Get basic statistics
        close = data['close']
        returns = close.pct_change()
        
        # Get Z-Score
        mean = close.rolling(self.zscore_period).mean()
        std = close.rolling(self.zscore_period).std()
        zscore = (close - mean) / std
        
        # Get Half-Life
        half_life = self._calculate_half_life(close)
        
        # Get Hurst Exponent
        hurst = self._calculate_hurst_exponent(close)
        
        # Beta and R-squared media
        beta, r_squared = self._calculate_mean_regression_stats(close)
        
        # Mean reversion speed
        reversion_speed = self._calculate_reversion_speed(close, mean)
        
        return {
            'zscore': zscore,
            'mean': mean,
            'std': std,
            'half_life': half_life,
            'hurst': hurst,
            'beta': beta,
            'r_squared': r_squared,
            'reversion_speed': reversion_speed,
            'returns': returns
        }

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        # Get required columns for indicators
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            close, 
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )
        
        # RSI
        rsi = ta.RSI(close, timeperiod=self.rsi_period)
        
        # Stochastic
        slowk, slowd = ta.STOCH(high, low, close)
        
        # ATR for volatility
        atr = ta.ATR(high, low, close)
        
        # CCI for trend
        cci = ta.CCI(high, low, close)
        
        return {
            'bb': {
                'upper': bb_upper,
                'middle': bb_middle,
                'lower': bb_lower,
                'width': (bb_upper - bb_lower) / bb_middle
            },
            'rsi': rsi,
            'stoch': {'k': slowk, 'd': slowd},
            'atr': atr,
            'cci': cci
        }

    def _check_reversion_conditions(self, stats: Dict, indicators: Dict) -> Dict:
        # Check conditions for mean reversion
        current_zscore = stats['zscore'].iloc[-1]
        current_rsi = indicators['rsi'].iloc[-1]
        current_cci = indicators['cci'].iloc[-1]
        
        # Conditions for oversold
        oversold_conditions = (
            current_zscore < -self.zscore_threshold and
            current_rsi < self.rsi_oversold and
            current_cci < -100
        )
        
        # Conditions for overbought
        overbought_conditions = (
            current_zscore > self.zscore_threshold and
            current_rsi > self.rsi_overbought and
            current_cci > 100
        )
        
        # Check if conditions are met
        if oversold_conditions or overbought_conditions:
            # Get additional statistics
            signal_quality = self._calculate_signal_quality(stats, indicators)
            
            return {
                'valid': True,
                'direction': 'long' if oversold_conditions else 'short',
                'zscore': current_zscore,
                'rsi': current_rsi,
                'cci': current_cci,
                'signal_quality': signal_quality
            }
            
        return {'valid': False}

    def _check_filters(self, data: pd.DataFrame, stats: Dict, 
                      indicators: Dict) -> bool:
        # Apply filters to the mean reversion signal
        # Check volatility
        if not self._check_volatility_conditions(indicators):
            return False
            
        # Check volume
        if self.volume_filter and not self._check_volume_filter(data):
            return False
            
        # Check trend filter
        if self.trend_filter and not self._check_trend_filter(data, stats):
            return False
            
        # Check mean reversion likelihood
        if not self._check_mean_reversion_likelihood(stats):
            return False
            
        return True

    def _calculate_signal_quality(self, stats: Dict, indicators: Dict) -> float:
        # Calculate signal quality based on statistics and indicators
        # Z-Score strength
        zscore_strength = min(abs(stats['zscore'].iloc[-1]) / self.zscore_threshold, 1.0)
        
        # Mean reversion speed
        reversion_strength = min(stats['reversion_speed'] / self.mean_reversion_threshold, 1.0)
        
        # Quality of correlation
        correlation_quality = stats['r_squared']
        
        # Quality of technical indicators
        technical_quality = self._calculate_technical_quality(indicators)
        
        # Combine all factors
        signal_quality = (
            zscore_strength * 0.3 +
            reversion_strength * 0.3 +
            correlation_quality * 0.2 +
            technical_quality * 0.2
        )
        
        return np.clip(signal_quality, 0, 1)

    def _generate_reversion_signal(self, data: pd.DataFrame,
                                 stats: Dict,
                                 indicators: Dict,
                                 conditions: Dict) -> StrategySignal:
        # Generate mean reversion signal
        current_price = data['close'].iloc[-1]
        atr = indicators['atr'].iloc[-1]
        
        # Generate price target
        price_target = stats['mean'].iloc[-1]
        
        # Generate stop loss
        stop_loss = self._calculate_stop_loss(
            current_price, 
            atr,
            stats['std'].iloc[-1],
            conditions['direction']
        )
        
        # Generate signal confidence
        confidence = self._calculate_signal_confidence(
            stats, indicators, conditions)
        
        return StrategySignal(
            timestamp=datetime.now(),
            direction=conditions['direction'],
            strength=abs(conditions['zscore']),
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            targets=[price_target],
            timeframe=self.timeframe,
            metadata={
                'zscore': conditions['zscore'],
                'rsi': conditions['rsi'],
                'signal_quality': conditions['signal_quality'],
                'half_life': stats['half_life'],
                'hurst': stats['hurst']
            }
        )

    def _calculate_half_life(self, prices: pd.Series) -> float:
        # Calculate half-life of mean reversion
        returns = prices.pct_change()
        lag_returns = returns.shift(1)
        lag_returns = lag_returns[1:]
        returns = returns[1:]
        
        # Regression
        slope, intercept = np.polyfit(lag_returns, returns, 1)
        
        # Calculate half-life
        half_life = -np.log(2) / np.log(1 + slope)
        return half_life

    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        # Get Hurst Exponent
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag])))
               for lag in lags]
        
        # Regression
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        return poly[0]

    def _calculate_mean_regression_stats(self, 
                                       prices: pd.Series) -> Tuple[float, float]:
        # Calculate statistics for mean reversion
        mean = prices.rolling(self.mean_period).mean()
        spread = prices - mean
        
        # Generate beta and R-squared
        spread_lag = spread.shift(1)
        slope, intercept = np.polyfit(spread_lag.dropna(), spread.dropna(), 1)
        
        # Generate R-squared
        correlation_matrix = np.corrcoef(spread_lag.dropna(), spread.dropna())
        r_squared = correlation_matrix[0,1]**2
        
        return slope, r_squared

    def _calculate_reversion_speed(self, prices: pd.Series, 
                                 mean: pd.Series) -> float:
        # Calculate speed of mean reversion
        spread = prices - mean
        spread_changes = spread.diff()
        
        # Calculate reversion speed
        reversion_speed = -np.mean(spread * spread_changes)
        
        return reversion_speed

    def _check_mean_reversion_likelihood(self, stats: Dict) -> bool:
        # Check likelihood of mean reversion
        # Check Hurst exponent
        if stats['hurst'] > 0.5:  # Trend following behavior
            return False
            
        # Check half-life
        if stats['half_life'] > self.max_holding_period:
            return False
            
        # Check mean reversion speed
        if stats['reversion_speed'] < self.mean_reversion_threshold:
            return False
            
        return True

    def get_required_columns(self) -> List[str]:
        # Return the required columns for the strategy
        return ['open', 'high', 'low', 'close', 'volume']

    def get_min_required_history(self) -> int:
        # Return the minimum required history for the strategy
        return max(self.lookback_period, 100)  # Per Hurst calculation

    def _apply_strategy_filters(self, signal: StrategySignal,
                              data: pd.DataFrame) -> bool:
        # Apply strategy-specific filters
        return signal.confidence >= 0.6  # Minimum confidence
