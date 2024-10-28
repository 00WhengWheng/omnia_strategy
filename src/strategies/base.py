from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import numpy as np
from ..core.constants import TimeFrame, SignalStrength

@dataclass
class StrategyState:
    is_active: bool
    current_position: str  # 'long', 'short', or 'flat'
    entry_price: Optional[float]
    entry_time: Optional[datetime]
    current_stop: Optional[float]
    current_target: Optional[float]
    risk_per_trade: float
    position_size: float
    metadata: Dict

@dataclass
class StrategySignal:
    timestamp: datetime
    direction: str  # 'long', 'short', or 'flat'
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    entry_price: Optional[float]
    stop_loss: Optional[float]
    targets: List[float]
    timeframe: TimeFrame
    metadata: Dict

class BaseStrategy(ABC):
    """Classe base per tutte le strategie di trading"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.timeframe = TimeFrame(config.get('timeframe', 'D1'))
        
        # Stato della strategia
        self.state = StrategyState(
            is_active=True,
            current_position='flat',
            entry_price=None,
            entry_time=None,
            current_stop=None,
            current_target=None,
            risk_per_trade=config.get('risk_per_trade', 0.01),
            position_size=0.0,
            metadata={}
        )
        
        # Performance tracking
        self.trades_history = []
        self.signals_history = []
        
        # Risk management
        self.risk_manager = self._initialize_risk_manager()
        
        # Analizzatori
        self.analyzers = self._initialize_analyzers()
        
        # Setup addizionale specifico della strategia
        self._initialize_strategy()
    
    @abstractmethod
    def _initialize_strategy(self) -> None:
        """Inizializzazione specifica della strategia"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> StrategySignal:
        """Genera segnali di trading"""
        pass

    def update(self, data: pd.DataFrame) -> Optional[StrategySignal]:
        """Aggiorna la strategia con nuovi dati"""
        if not self.state.is_active:
            return None
            
        # Valida i dati
        if not self._validate_data(data):
            return None
            
        # Aggiorna analizzatori
        self._update_analyzers(data)
        
        # Genera segnali
        signal = self.generate_signals(data)
        
        # Applica filtri
        if not self._apply_filters(signal, data):
            return None
            
        # Applica risk management
        signal = self.risk_manager.adjust_signal(signal, self.state)
        
        # Aggiorna stato
        self._update_state(signal)
        
        # Traccia il segnale
        self._track_signal(signal)
        
        return signal
    
    def _initialize_risk_manager(self):
        """Inizializza il gestore del rischio"""
        from ..risk.manager import RiskManager
        return RiskManager(self.config.get('risk', {}))
    
    def _initialize_analyzers(self) -> Dict:
        """Inizializza gli analizzatori richiesti"""
        return {
            'technical': self._create_technical_analyzer(),
            'volatility': self._create_volatility_analyzer(),
            'sentiment': self._create_sentiment_analyzer()
        }
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Valida i dati in input"""
        required_columns = self.get_required_columns()
        if not all(col in data.columns for col in required_columns):
            return False
            
        if len(data) < self.get_min_required_history():
            return False
            
        return True
    
    def _update_analyzers(self, data: pd.DataFrame) -> None:
        """Aggiorna tutti gli analizzatori"""
        for analyzer in self.analyzers.values():
            analyzer.analyze(data)
    
    def _apply_filters(self, signal: StrategySignal, 
                      data: pd.DataFrame) -> bool:
        """Applica filtri al segnale"""
        # Time filters
        if not self._check_time_filters(signal.timestamp):
            return False
            
        # Market condition filters
        if not self._check_market_conditions(data):
            return False
            
        # Position filters
        if not self._check_position_filters(signal):
            return False
            
        # Strategy specific filters
        if not self._apply_strategy_filters(signal, data):
            return False
            
        return True
    
    def _update_state(self, signal: StrategySignal) -> None:
        """Aggiorna lo stato della strategia"""
        # Update position state
        if signal.direction != self.state.current_position:
            self._handle_position_change(signal)
        
        # Update stops and targets
        self.state.current_stop = signal.stop_loss
        self.state.current_target = signal.targets[0] if signal.targets else None
        
        # Update metadata
        self.state.metadata.update({
            'last_signal': signal,
            'last_update': signal.timestamp
        })
    
    def _handle_position_change(self, signal: StrategySignal) -> None:
        """Gestisce il cambio di posizione"""
        # Close current position if exists
        if self.state.current_position != 'flat':
            self._close_position(signal.timestamp, signal.entry_price)
        
        # Open new position if signal is not flat
        if signal.direction != 'flat':
            self._open_position(signal)
    
    def _open_position(self, signal: StrategySignal) -> None:
        """Apre una nuova posizione"""
        self.state.current_position = signal.direction
        self.state.entry_price = signal.entry_price
        self.state.entry_time = signal.timestamp
        self.state.position_size = self._calculate_position_size(signal)
        
        # Track trade
        self._track_trade('open', signal)
    
    def _close_position(self, timestamp: datetime, price: float) -> None:
        """Chiude la posizione corrente"""
        # Calculate P&L
        pnl = self._calculate_pnl(price)
        
        # Update state
        self.state.current_position = 'flat'
        self.state.entry_price = None
        self.state.entry_time = None
        self.state.position_size = 0.0
        
        # Track trade
        self._track_trade('close', price, pnl)
    
    def _calculate_position_size(self, signal: StrategySignal) -> float:
        """Calcola la dimensione della posizione"""
        return self.risk_manager.calculate_position_size(
            signal, self.state.risk_per_trade)
    
    def _calculate_pnl(self, current_price: float) -> float:
        """Calcola il P&L della posizione"""
        if not self.state.entry_price:
            return 0.0
            
        price_diff = current_price - self.state.entry_price
        if self.state.current_position == 'short':
            price_diff = -price_diff
            
        return price_diff * self.state.position_size
    
    def _track_signal(self, signal: StrategySignal) -> None:
        """Traccia il segnale generato"""
        self.signals_history.append(signal)
    
    def _track_trade(self, action: str, data: any, pnl: float = None) -> None:
        """Traccia l'attività di trading"""
        trade = {
            'timestamp': datetime.now(),
            'action': action,
            'position': self.state.current_position,
            'price': self.state.entry_price if action == 'open' else data,
            'size': self.state.position_size,
            'pnl': pnl
        }
        self.trades_history.append(trade)
    
    @abstractmethod
    def _apply_strategy_filters(self, signal: StrategySignal,
                              data: pd.DataFrame) -> bool:
        """Applica filtri specifici della strategia"""
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Ritorna le colonne richieste dalla strategia"""
        pass
    
    @abstractmethod
    def get_min_required_history(self) -> int:
        """Ritorna il minimo storico richiesto"""
        pass
    
    def get_strategy_state(self) -> Dict:
        """Ritorna lo stato corrente della strategia"""
        return {
            'name': self.name,
            'active': self.state.is_active,
            'position': self.state.current_position,
            'entry_price': self.state.entry_price,
            'current_stop': self.state.current_stop,
            'current_target': self.state.current_target,
            'position_size': self.state.position_size,
            'metadata': self.state.metadata
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calcola e ritorna le metriche di performance"""
        if not self.trades_history:
            return {}
            
        trades_df = pd.DataFrame(self.trades_history)
        
        return {
            'total_trades': len(trades_df[trades_df['action'] == 'close']),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
            'total_pnl': trades_df['pnl'].sum(),
            'avg_pnl': trades_df['pnl'].mean(),
            'max_drawdown': self._calculate_max_drawdown(trades_df),
            'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df),
            'avg_trade_duration': self._calculate_avg_trade_duration()
        }

    def _check_time_filters(self, timestamp: datetime) -> bool:
        """Verifica filtri temporali"""
        # Trading hours filter
        if not self._is_trading_hour(timestamp):
            return False
            
        # Day of week filter
        if not self._is_trading_day(timestamp):
            return False
            
        return True
        
    def _check_market_conditions(self, data: pd.DataFrame) -> bool:
        """Verifica condizioni di mercato"""
        # Volatility filter
        if not self._check_volatility(data):
            return False
            
        # Liquidity filter
        if not self._check_liquidity(data):
            return False
            
        return True
        
    def _check_position_filters(self, signal: StrategySignal) -> bool:
        """Verifica filtri relativi alla posizione"""
        # Max open positions
        if not self._check_max_positions():
            return False
            
        # Position holding time
        if not self._check_holding_time():
            return False
            
        return True
