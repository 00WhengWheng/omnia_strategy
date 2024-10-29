from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import numpy as np
import logging
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
    # Base class for trading strategies
    def __init__(self, config: Dict):
        required_keys = ['name', 'timeframe', 'risk_per_trade']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.timeframe = TimeFrame(config.get('timeframe', 'D1'))
        
        # Strategy state
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
        
        # Analyzers
        self.analyzers = self._initialize_analyzers()
        
        # Setup the strategy
        self._initialize_strategy()

        # Initialize logging
        self._initialize_logging()
    
    def _initialize_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.name)
        self.logger.info(f"Initialized strategy: {self.name}")

    @abstractmethod
    def _initialize_strategy(self) -> None:
        """Initialize the strategy."""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> StrategySignal:
        """Generate trading signals based on the data provided."""
        pass

    def update(self, data: pd.DataFrame) -> Optional[StrategySignal]:
        """Update the strategy with new data."""
        self.logger.info("Updating strategy with new data.")
        if not self.state.is_active:
            return None
            
        # Validate the input data
        if not self._validate_data(data):
            return None
            
        # Update analyzers
        self._update_analyzers(data)
        
        # Generate signals
        signal = self.generate_signals(data)
        
        # Apply filters and risk management
        if not self._apply_filters(signal, data):
            return None
            
        # Adjust signal based on risk management
        signal = self.risk_manager.adjust_signal(signal, self.state)
        
        # Apply the signal
        self._update_state(signal)
        
        # Track the signal
        self._track_signal(signal)
        
        self.logger.debug(f"Generated signal: {signal}")
        return signal
    
    def _initialize_risk_manager(self):
        """Initialize the risk manager."""
        from ..risk.manager import RiskManager
        return RiskManager(self.config.get('risk', {}))
    
    def _initialize_analyzers(self) -> Dict:
        """Initialize the analyzers."""
        return {
            'technical': self._create_technical_analyzer(),
            'volatility': self._create_volatility_analyzer(),
            'sentiment': self._create_sentiment_analyzer()
        }
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the input data."""
        required_columns = self.get_required_columns()
        if not all(col in data.columns for col in required_columns):
            return False
            
        if len(data) < self.get_min_required_history():
            return False
            
        return True
    
    def _update_analyzers(self, data: pd.DataFrame) -> None:
        """Update the analyzers with new data."""
        for analyzer in self.analyzers.values():
            analyzer.analyze(data)
    
    def _apply_filters(self, signal: StrategySignal, data: pd.DataFrame) -> bool:
        """Apply filters to the signal."""
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
        """Update the strategy state based on the signal."""
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
        """Handle a change in position."""
        # Close current position if exists
        if self.state.current_position != 'flat':
            self._close_position(signal.timestamp, signal.entry_price)
        
        # Open new position if signal is not flat
        if signal.direction != 'flat':
            self._open_position(signal)
    
    def _open_position(self, signal: StrategySignal) -> None:
        """Opens a new position based on the signal."""
        self.state.current_position = signal.direction
        self.state.entry_price = signal.entry_price
        self.state.entry_time = signal.timestamp
        self.state.position_size = self._calculate_position_size(signal)
        
        # Track trade
        self._track_trade('open', signal)
    
    def _close_position(self, timestamp: datetime, price: float) -> None:
        """Close the current position."""
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
        """Calculate the position size based on the signal."""
        return self.risk_manager.calculate_position_size(
            signal, self.state.risk_per_trade)
    
    def _calculate_pnl(self, current_price: float) -> float:
        """Calculate the profit or loss of the current position."""
        if not self.state.entry_price:
            return 0.0
            
        price_diff = current_price - self.state.entry_price
        if self.state.current_position == 'short':
            price_diff = -price_diff
            
        return price_diff * self.state.position_size
    
    def _track_signal(self, signal: StrategySignal) -> None:
        """Track the trading signal."""
        self.signals_history.append(signal)
    
    def _track_trade(self, action: str, data: any, pnl: float = None) -> None:
        """Track the trading history."""
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
    def _apply_strategy_filters(self, signal: StrategySignal, data: pd.DataFrame) -> bool:
        """Apply strategy specific filters."""
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Get the required columns for the analysis."""
        pass
    
    @abstractmethod
    def get_min_required_history(self) -> int:
        """Get the minimum required history length."""
        pass
    
    def get_strategy_state(self) -> Dict:
        """Get the current state of the strategy."""
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
        """Get the performance metrics of the strategy."""
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
        """Check time filters."""
        # Trading hours filter
        if not self._is_trading_hour(timestamp):
            return False
            
        # Day of week filter
        if not self._is_trading_day(timestamp):
            return False
            
        return True
        
    def _check_market_conditions(self, data: pd.DataFrame) -> bool:
        """Check market conditions."""
        # Volatility filter
        if not self._check_volatility(data):
            return False
            
        # Liquidity filter
        if not self._check_liquidity(data):
            return False
            
        return True
        
    def _check_position_filters(self, signal: StrategySignal) -> bool:
        """Check position filters."""
        # Max open positions
        if not self._check_max_positions():
            return False
            
        # Position holding time
        if not self._check_holding_time():
            return False
            
        return True