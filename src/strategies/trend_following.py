from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import talib as ta
from .base import BaseStrategy, StrategySignal
from ..core.constants import TimeFrame

class TrendFollowingStrategy(BaseStrategy):
    def _initialize_strategy(self) -> None:
        """Inizializza i parametri specifici della strategia"""
        # Parametri Moving Average
        self.fast_period = self.config.get('fast_ma', 20)
        self.medium_period = self.config.get('medium_ma', 50)
        self.slow_period = self.config.get('slow_ma', 200)
        
        # Parametri ADX
        self.adx_period = self.config.get('adx_period', 14)
        self.adx_threshold = self.config.get('adx_threshold', 25)
        
        # Parametri Multi-timeframe
        self.mtf_weights = {
            TimeFrame.D1: 0.5,    # Daily
            TimeFrame.H4: 0.3,    # 4 ore
            TimeFrame.H1: 0.2     # 1 ora
        }
        
        # Filtri
        self.min_trend_strength = self.config.get('min_trend_strength', 0.3)
        self.volatility_filter = self.config.get('volatility_filter', True)
        self.volume_filter = self.config.get('volume_filter', True)
        
        # Risk Management
        self.trailing_stop = self.config.get('trailing_stop', True)
        self.atr_multiplier = self.config.get('atr_multiplier', 2.0)
        self.profit_targets = self.config.get('profit_targets', [2.0, 3.0, 4.0])  # In ATR units
        
        # Inizializza cache per i calcoli
        self.cache = {}

    def generate_signals(self, data: pd.DataFrame) -> StrategySignal:
        """Genera segnali di trading basati sul trend"""
        # Calcola indicatori principali
        indicators = self._calculate_indicators(data)
        
        # Analisi multi-timeframe
        mtf_analysis = self._analyze_multiple_timeframes(data)
        
        # Identifica la direzione del trend
        trend_direction = self._determine_trend_direction(indicators, mtf_analysis)
        
        # Calcola la forza del trend
        trend_strength = self._calculate_trend_strength(indicators)
        
        # Identifica punti di entrata
        entry_conditions = self._identify_entry_conditions(
            data, indicators, trend_direction)
        
        # Genera segnale
        if entry_conditions['valid']:
            signal = self._generate_entry_signal(
                data, indicators, trend_direction, trend_strength)
        else:
            signal = self._generate_exit_signal(data, indicators)
            
        return signal

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calcola gli indicatori tecnici principali"""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Moving Averages
        sma_fast = ta.SMA(close, timeperiod=self.fast_period)
        sma_medium = ta.SMA(close, timeperiod=self.medium_period)
        sma_slow = ta.SMA(close, timeperiod=self.slow_period)
        
        # Exponential Moving Averages
        ema_fast = ta.EMA(close, timeperiod=self.fast_period)
        ema_medium = ta.EMA(close, timeperiod=self.medium_period)
        ema_slow = ta.EMA(close, timeperiod=self.slow_period)
        
        # ADX per forza del trend
        adx = ta.ADX(high, low, close, timeperiod=self.adx_period)
        plus_di = ta.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        minus_di = ta.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        
        # Volume indicators
        obv = ta.OBV(close, volume)
        ad = ta.AD(high, low, close, volume)
        
        # ATR per volatilità e sizing
        atr = ta.ATR(high, low, close, timeperiod=14)
        
        return {
            'sma': {'fast': sma_fast, 'medium': sma_medium, 'slow': sma_slow},
            'ema': {'fast': ema_fast, 'medium': ema_medium, 'slow': ema_slow},
            'adx': {'value': adx, 'plus_di': plus_di, 'minus_di': minus_di},
            'volume': {'obv': obv, 'ad': ad},
            'atr': atr,
            'current_close': close.iloc[-1],
            'current_high': high.iloc[-1],
            'current_low': low.iloc[-1]
        }

    def _analyze_multiple_timeframes(self, data: pd.DataFrame) -> Dict:
        """Analizza il trend su multipli timeframe"""
        mtf_analysis = {}
        
        for timeframe in self.mtf_weights.keys():
            # Resample data to timeframe
            tf_data = self._resample_data(data, timeframe)
            
            # Calcola indicatori per questo timeframe
            indicators = self._calculate_indicators(tf_data)
            
            # Analizza trend per questo timeframe
            mtf_analysis[timeframe] = {
                'trend_direction': self._determine_trend_direction(indicators, {}),
                'trend_strength': self._calculate_trend_strength(indicators),
                'indicators': indicators
            }
            
        return mtf_analysis

    def _determine_trend_direction(self, indicators: Dict,
                                 mtf_analysis: Dict) -> float:
        """Determina la direzione del trend (-1 a 1)"""
        # Analisi Moving Average
        ma_score = self._analyze_moving_averages(indicators)
        
        # Analisi ADX
        adx_score = self._analyze_adx(indicators)
        
        # Multi-timeframe consensus
        mtf_score = self._calculate_mtf_consensus(mtf_analysis)
        
        # Combina i segnali
        trend_direction = (
            ma_score * 0.4 +
            adx_score * 0.3 +
            mtf_score * 0.3
        )
        
        return np.clip(trend_direction, -1, 1)

    def _analyze_moving_averages(self, indicators: Dict) -> float:
        """Analizza la configurazione delle medie mobili"""
        current_close = indicators['current_close']
        
        # Controlla allineamento MA
        fast = indicators['ema']['fast'].iloc[-1]
        medium = indicators['ema']['medium'].iloc[-1]
        slow = indicators['ema']['slow'].iloc[-1]
        
        # Calcola score basato sull'allineamento
        if fast > medium > slow:
            base_score = 1.0  # Trend rialzista
        elif fast < medium < slow:
            base_score = -1.0  # Trend ribassista
        else:
            # Trend non chiaro, usa la distanza relativa
            distances = [
                (current_close - slow) / slow,
                (current_close - medium) / medium,
                (current_close - fast) / fast
            ]
            base_score = np.mean(distances)
        
        # Aggiusta per la forza del trend
        score_adjustment = self._calculate_ma_strength(indicators)
        
        return base_score * score_adjustment

    def _analyze_adx(self, indicators: Dict) -> float:
        """Analizza ADX e Directional Indicators"""
        adx = indicators['adx']['value'].iloc[-1]
        plus_di = indicators['adx']['plus_di'].iloc[-1]
        minus_di = indicators['adx']['minus_di'].iloc[-1]
        
        # Calcola forza del trend
        trend_strength = min(adx / 100.0, 1.0)
        
        # Determina direzione
        if plus_di > minus_di:
            direction = 1.0
        elif minus_di > plus_di:
            direction = -1.0
        else:
            direction = 0.0
        
        return direction * trend_strength

    def _calculate_trend_strength(self, indicators: Dict) -> float:
        """Calcola la forza complessiva del trend"""
        # ADX strength
        adx_strength = indicators['adx']['value'].iloc[-1] / 100.0
        
        # Moving Average alignment strength
        ma_strength = self._calculate_ma_strength(indicators)
        
        # Volume trend strength
        volume_strength = self._calculate_volume_trend_strength(indicators)
        
        # Combine strengths
        trend_strength = (
            adx_strength * 0.4 +
            ma_strength * 0.4 +
            volume_strength * 0.2
        )
        
        return np.clip(trend_strength, 0, 1)

    def _identify_entry_conditions(self, data: pd.DataFrame,
                                 indicators: Dict,
                                 trend_direction: float) -> Dict:
        """Identifica condizioni di entrata valide"""
        # Verifica trend minimo
        if abs(trend_direction) < self.min_trend_strength:
            return {'valid': False, 'reason': 'weak_trend'}
            
        # Verifica filtro volatilità
        if self.volatility_filter and not self._check_volatility_filter(indicators):
            return {'valid': False, 'reason': 'volatility_filter'}
            
        # Verifica filtro volume
        if self.volume_filter and not self._check_volume_filter(indicators):
            return {'valid': False, 'reason': 'volume_filter'}
            
        # Verifica pullback
        pullback = self._identify_pullback(data, indicators, trend_direction)
        if not pullback['valid']:
            return {'valid': False, 'reason': 'no_pullback'}
            
        return {
            'valid': True,
            'pullback': pullback,
            'trend_direction': trend_direction
        }

    def _generate_entry_signal(self, data: pd.DataFrame,
                             indicators: Dict,
                             trend_direction: float,
                             trend_strength: float) -> StrategySignal:
        """Genera segnale di entrata"""
        current_price = indicators['current_close']
        
        # Calcola stop loss
        atr = indicators['atr'].iloc[-1]
        stop_loss = self._calculate_stop_loss(
            current_price, atr, trend_direction)
        
        # Calcola target prices
        targets = self._calculate_profit_targets(
            current_price, atr, trend_direction)
        
        # Calcola confidenza
        confidence = self._calculate_signal_confidence(
            trend_direction, trend_strength, indicators)
        
        return StrategySignal(
            timestamp=datetime.now(),
            direction='long' if trend_direction > 0 else 'short',
            strength=abs(trend_direction),
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            targets=targets,
            timeframe=self.timeframe,
            metadata={
                'trend_strength': trend_strength,
                'atr': atr,
                'adx': indicators['adx']['value'].iloc[-1]
            }
        )

    def _calculate_stop_loss(self, current_price: float,
                           atr: float,
                           trend_direction: float) -> float:
        """Calcola livello di stop loss"""
        stop_distance = atr * self.atr_multiplier
        
        if trend_direction > 0:  # Long
            return current_price - stop_distance
        else:  # Short
            return current_price + stop_distance

    def _calculate_profit_targets(self, current_price: float,
                                atr: float,
                                trend_direction: float) -> List[float]:
        """Calcola livelli di take profit"""
        targets = []
        
        for target_multiplier in self.profit_targets:
            target_distance = atr * target_multiplier
            
            if trend_direction > 0:  # Long
                target = current_price + target_distance
            else:  # Short
                target = current_price - target_distance
                
            targets.append(target)
            
        return targets

    def get_required_columns(self) -> List[str]:
        """Ritorna le colonne richieste dalla strategia"""
        return ['open', 'high', 'low', 'close', 'volume']

    def get_min_required_history(self) -> int:
        """Ritorna il minimo storico richiesto"""
        return max(self.slow_period, self.adx_period) + 50

    def _apply_strategy_filters(self, signal: StrategySignal,
                              data: pd.DataFrame) -> bool:
        """Applica filtri specifici della strategia"""
        # Verifica trend minimo
        if abs(signal.strength) < self.min_trend_strength:
            return False
            
        # Altri filtri specifici della strategia
        return True
