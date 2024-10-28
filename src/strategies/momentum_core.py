from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import talib as ta
from .base import BaseStrategy, StrategySignal
from ..core.constants import TimeFrame

class MomentumStrategy(BaseStrategy):
    def _initialize_strategy(self) -> None:
        """Inizializza i parametri specifici della strategia"""
        # Parametri RSI
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_threshold = self.config.get('rsi_threshold', 60)
        
        # Parametri MACD
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        
        # Parametri ADX
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
        
        # Filtri
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

    def generate_signals(self, data: pd.DataFrame) -> StrategySignal:
        """Genera segnali di momentum"""
        # Calcola indicatori
        indicators = self._calculate_momentum_indicators(data)
        
        # Analisi multi-timeframe
        mtf_analysis = self._analyze_multiple_timeframes(data)
        
        # Identifica momentum signals
        momentum_signals = self._identify_momentum_signals(
            data, indicators, mtf_analysis)
        
        # Verifica filtri
        if not self._check_momentum_filters(data, momentum_signals, indicators):
            return self._generate_neutral_signal(data)
        
        # Calcola forza del momentum
        momentum_strength = self._calculate_momentum_strength(
            indicators, momentum_signals)
        
        # Genera segnale se valido
        if momentum_signals['valid']:
            signal = self._generate_momentum_signal(
                data, momentum_signals, indicators, momentum_strength)
            
            # Aggiorna statistiche
            self._update_momentum_statistics(momentum_signals, signal)
        else:
            signal = self._generate_neutral_signal(data)
        
        return signal

    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict:
        """Calcola indicatori di momentum"""
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
        
        # ADX e DMI
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
