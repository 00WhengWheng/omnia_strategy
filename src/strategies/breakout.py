from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import talib as ta
from .base import BaseStrategy, StrategySignal
from ..core.constants import TimeFrame

class BreakoutStrategy(BaseStrategy):
    def _initialize_strategy(self) -> None:
        """Inizializza i parametri specifici della strategia"""
        # Parametri Breakout
        self.breakout_period = self.config.get('breakout_period', 20)
        self.confirmation_period = self.config.get('confirmation_period', 3)
        self.volatility_lookback = self.config.get('volatility_lookback', 20)
        
        # Livelli di Breakout
        self.support_resistance_periods = [20, 50, 200]  # Multiple timeframes
        self.pivot_points_lookback = self.config.get('pivot_points_lookback', 10)
        self.fibonacci_levels = self.config.get('fibonacci_levels', True)
        
        # Filtri
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

    def generate_signals(self, data: pd.DataFrame) -> StrategySignal:
        """Genera segnali di breakout"""
        # Identifica livelli chiave
        levels = self._identify_key_levels(data)
        
        # Calcola indicatori
        indicators = self._calculate_indicators(data)
        
        # Identifica breakout
        breakout = self._identify_breakout(data, levels, indicators)
        
        # Verifica filtri
        if not self._check_filters(data, breakout, indicators):
            return self._generate_neutral_signal(data)
        
        # Genera segnale se breakout è valido
        if breakout['valid']:
            signal = self._generate_breakout_signal(
                data, breakout, levels, indicators)
            
            # Aggiorna statistiche
            self._update_breakout_statistics(breakout, signal)
        else:
            signal = self._generate_neutral_signal(data)
        
        return signal

    def _identify_key_levels(self, data: pd.DataFrame) -> Dict:
        """Identifica livelli chiave per breakout"""
        # Support & Resistance levels
        sr_levels = self._calculate_support_resistance(data)
        
        # Pivot Points
        pivots = self._calculate_pivot_points(data)
        
        # Range di consolidamento
        consolidation = self._identify_consolidation_range(data)
        
        # Livelli Fibonacci se abilitati
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
        """Calcola livelli di supporto e resistenza"""
        levels = {}
        
        for period in self.support_resistance_periods:
            # Calcola massimi e minimi del periodo
            high_levels = data['high'].rolling(period).max()
            low_levels = data['low'].rolling(period).min()
            
            # Trova cluster di livelli
            high_clusters = self._cluster_price_levels(high_levels)
            low_clusters = self._cluster_price_levels(low_levels)
            
            levels[period] = {
                'resistance': high_clusters,
                'support': low_clusters
            }
            
        return levels

    def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict:
        """Calcola pivot points"""
        pivots = {}
        
        # Calcola pivot classici
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
        """Identifica range di consolidamento"""
        # Calcola volatilità rolling
        volatility = data['close'].rolling(self.volatility_lookback).std()
        avg_volatility = volatility.mean()
        
        # Identifica periodi di bassa volatilità
        low_vol_periods = volatility < (avg_volatility * 0.8)
        
        # Trova l'ultimo range di consolidamento
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
        """Calcola livelli basati sul volume profile"""
        # Crea price bins
        price_range = np.linspace(
            data['low'].min(),
            data['high'].max(),
            50  # numero di bins
        )
        
        volume_profile = {}
        for i in range(len(price_range)-1):
            mask = (data['close'] >= price_range[i]) & (data['close'] < price_range[i+1])
            volume_profile[price_range[i]] = data.loc[mask, 'volume'].sum()
            
        # Trova high volume nodes
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
        """Identifica pattern di breakout"""
        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Verifica breakout dai livelli chiave
        resistance_break = self._check_resistance_breakout(
            current_price, levels, indicators)
        support_break = self._check_support_breakout(
            current_price, levels, indicators)
        
        # Verifica la forza del breakout
        if resistance_break['valid'] or support_break['valid']:
            breakout_strength = self._calculate_breakout_strength(
                data, current_price, current_volume, indicators)
            
            # Determina direzione e dettagli
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
        """Calcola la forza del breakout"""
        # Componenti della forza del breakout
        price_momentum = self._calculate_momentum_score(data)
        volume_strength = current_volume / indicators['volume_sma'].iloc[-1]
        volatility_ratio = indicators['atr'].iloc[-1] / indicators['atr'].rolling(20).mean().iloc[-1]
        
        # Calcola score composito
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
        """Genera segnale di breakout"""
        current_price = data['close'].iloc[-1]
        atr = indicators['atr'].iloc[-1]
        
        # Calcola target basato sui livelli successivi
        if breakout['direction'] == 'long':
            next_level = self._find_next_resistance(current_price, levels)
            stop_loss = current_price - (atr * self.atr_multiplier)
        else:
            next_level = self._find_next_support(current_price, levels)
            stop_loss = current_price + (atr * self.atr_multiplier)
            
        # Calcola target multipli
        targets = self._calculate_breakout_targets(
            current_price,
            next_level,
            breakout['direction']
        )
        
        # Calcola confidenza
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
        """Applica filtri per breakout"""
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
        """Rileva pattern di falso breakout"""
        # Verifica price action recente
        recent_close = data['close'].iloc[-1]
        recent_high = data['high'].iloc[-1]
        recent_low = data['low'].iloc[-1]
        
        if breakout['direction'] == 'long':
            # Verifica se il prezzo è tornato sotto il livello dopo il breakout
            return recent_close < breakout['level_broken']
        else:
            # Verifica se il prezzo è tornato sopra il livello dopo il breakout
            return recent_close > breakout['level_broken']

    def _update_breakout_statistics(self, breakout: Dict,
                                  signal: StrategySignal) -> None:
        """Aggiorna statistiche dei breakout"""
        self.breakout_history.append({
            'timestamp': signal.timestamp,
            'direction': breakout['direction'],
            'strength': breakout['strength'],
            'level': breakout['level_broken'],
            'volume_surge': breakout['volume_surge'],
            'confidence': signal.confidence
        })

    def get_required_columns(self) -> List[str]:
        """Ritorna le colonne richieste dalla strategia"""
        return ['open', 'high', 'low', 'close', 'volume']

    def get_min_required_history(self) -> int:
        """Ritorna il minimo storico richiesto"""
        return max(self.support_resistance_periods)

    def _apply_strategy_filters(self, signal: StrategySignal,
                              data: pd.DataFrame) -> bool:
        """Applica filtri specifici della strategia"""
        return signal.confidence >= 0.7  # Alta confidenza richiesta per breakout
