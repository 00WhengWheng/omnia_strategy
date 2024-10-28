from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class BacktestConfig:
    initial_capital: float
    commission: float
    slippage: float
    position_size: float
    max_positions: int
    use_fractional: bool
    use_stop_loss: bool
    use_trailing_stop: bool
    use_take_profit: bool
    risk_per_trade: float
    max_drawdown: float

@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float
    commission: float
    slippage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy_name: str
    exit_reason: Optional[str]

@dataclass
class BacktestResult:
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    metrics: Dict
    parameters: Dict
    start_date: datetime
    end_date: datetime
    strategy_name: str

class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        """Inizializza il Backtest Engine"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.equity = config.initial_capital
        self.positions = {}
        self.trades = []
        
        # Performance tracking
        self.equity_curve = []
        self.drawdown_curve = []
        self.peak_equity = config.initial_capital
        
        # Market data
        self.current_data = {}
        self.historical_data = {}
        
    def run_backtest(self, 
                    strategy,
                    data: Dict[str, pd.DataFrame],
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> BacktestResult:
        """Esegue il backtest della strategia"""
        try:
            # Validate data
            self._validate_data(data)
            
            # Initialize backtest
            self._initialize_backtest(data, start_date, end_date)
            
            # Run simulation
            for timestamp in self.timestamps:
                # Update market data
                self._update_market_data(timestamp, data)
                
                # Check stops and targets
                self._check_exit_conditions()
                
                # Generate signals
                signals = strategy.generate_signals(self.current_data)
                
                # Process signals
                self._process_signals(signals)
                
                # Update portfolio state
                self._update_portfolio_state()
                
                # Track performance
                self._track_performance()
                
            # Generate results
            results = self._generate_results(strategy)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            raise
            
    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Valida i dati di input"""
        for symbol, df in data.items():
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns for {symbol}")
                
            # Check for missing values
            if df[required_columns].isnull().any().any():
                raise ValueError(f"Found missing values in {symbol}")
                
            # Check data order
            if not df.index.is_monotonic_increasing:
                raise ValueError(f"Data for {symbol} is not properly ordered")
                
        return True
        
    def _initialize_backtest(self, 
                           data: Dict[str, pd.DataFrame],
                           start_date: Optional[datetime],
                           end_date: Optional[datetime]):
        """Inizializza lo stato del backtest"""
        # Get common timestamps
        all_timestamps = set.intersection(*[set(df.index) for df in data.values()])
        self.timestamps = sorted(all_timestamps)
        
        # Apply date range if specified
        if start_date:
            self.timestamps = [t for t in self.timestamps if t >= start_date]
        if end_date:
            self.timestamps = [t for t in self.timestamps if t <= end_date]
            
        # Reset state
        self.equity = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.peak_equity = self.config.initial_capital
        
        # Store historical data
        self.historical_data = data
        
    def _update_market_data(self, 
                           timestamp: datetime,
                           data: Dict[str, pd.DataFrame]):
        """Aggiorna i dati di mercato correnti"""
        self.current_data = {
            symbol: df.loc[:timestamp]
            for symbol, df in data.items()
        }
        
    def _check_exit_conditions(self):
        """Verifica condizioni di uscita per le posizioni aperte"""
        for symbol, position in list(self.positions.items()):
            current_price = self._get_current_price(symbol)
            
            # Check stop loss
            if (self.config.use_stop_loss and 
                position.stop_loss is not None and
                self._is_stop_loss_triggered(position, current_price)):
                self._close_position(symbol, current_price, "stop_loss")
                continue
                
            # Check take profit
            if (self.config.use_take_profit and
                position.take_profit is not None and
                self._is_take_profit_triggered(position, current_price)):
                self._close_position(symbol, current_price, "take_profit")
                continue
                
            # Check trailing stop
            if (self.config.use_trailing_stop and
                self._is_trailing_stop_triggered(position, current_price)):
                self._close_position(symbol, current_price, "trailing_stop")
                
    def _process_signals(self, signals: List[Dict]):
        """Processa i segnali di trading"""
        for signal in signals:
            # Validate signal
            if not self._validate_signal(signal):
                continue
                
            symbol = signal['symbol']
            direction = signal['direction']
            
            # Check if we can open new position
            if len(self.positions) >= self.config.max_positions:
                continue
                
            # Check if we already have position
            if symbol in self.positions:
                # Update stops if needed
                self._update_position_stops(symbol, signal)
                continue
                
            # Calculate position size
            quantity = self._calculate_position_size(signal)
            
            # Open new position
            self._open_position(symbol, direction, quantity, signal)
            
    def _update_portfolio_state(self):
        """Aggiorna lo stato del portfolio"""
        # Update position values
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            position.current_price = current_price
            position.unrealized_pnl = self._calculate_position_pnl(position)
            
        # Update equity
        self.equity = self.config.initial_capital + \
                     sum(pos.unrealized_pnl for pos in self.positions.values()) + \
                     sum(trade.pnl for trade in self.trades)
                     
        # Update peak equity
        self.peak_equity = max(self.peak_equity, self.equity)
        
    def _track_performance(self):
        """Traccia le performance"""
        # Track equity
        self.equity_curve.append(self.equity)
        
        # Track drawdown
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        self.drawdown_curve.append(current_drawdown)
        
        # Check drawdown limit
        if current_drawdown > self.config.max_drawdown:
            self._close_all_positions("max_drawdown")
            
    def _open_position(self, 
                      symbol: str,
                      direction: str,
                      quantity: float,
                      signal: Dict):
        """Apre una nuova posizione"""
        current_price = self._get_current_price(symbol)
        
        # Calculate transaction costs
        commission = self._calculate_commission(current_price, quantity)
        slippage = self._calculate_slippage(current_price, quantity)
        
        # Adjust entry price for costs
        entry_price = current_price * (1 + slippage if direction == 'long' else 1 - slippage)
        
        # Create trade record
        trade = BacktestTrade(
            entry_time=self.current_data[symbol].index[-1],
            exit_time=None,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            pnl=0.0,
            commission=commission,
            slippage=slippage,
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit'),
            strategy_name=signal.get('strategy_name', 'unknown'),
            exit_reason=None
        )
        
        # Add to positions
        self.positions[symbol] = trade
        
    def _close_position(self,
                       symbol: str,
                       current_price: float,
                       reason: str):
        """Chiude una posizione esistente"""
        position = self.positions[symbol]
        
        # Calculate transaction costs
        commission = self._calculate_commission(current_price, position.quantity)
        slippage = self._calculate_slippage(current_price, position.quantity)
        
        # Adjust exit price for costs
        exit_price = current_price * (1 - slippage if position.direction == 'long' else 1 + slippage)
        
        # Calculate PnL
        if position.direction == 'long':
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
            
        pnl -= (position.commission + commission)
        
        # Update trade record
        position.exit_time = self.current_data[symbol].index[-1]
        position.exit_price = exit_price
        position.pnl = pnl
        position.exit_reason = reason
        
        # Move to completed trades
        self.trades.append(position)
        
        # Remove from active positions
        del self.positions[symbol]
        
    def _generate_results(self, strategy) -> BacktestResult:
        """Genera i risultati del backtest"""
        return BacktestResult(
            trades=self.trades,
            equity_curve=pd.Series(self.equity_curve, index=self.timestamps),
            drawdown_curve=pd.Series(self.drawdown_curve, index=self.timestamps),
            metrics=self._calculate_metrics(),
            parameters=strategy.get_parameters(),
            start_date=self.timestamps[0],
            end_date=self.timestamps[-1],
            strategy_name=strategy.__class__.__name__
        )
        
    def _calculate_metrics(self) -> Dict:
        """Calcola le metriche di performance"""
        if not self.trades:
            return {}
            
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t.pnl > 0]),
            'losing_trades': len([t for t in self.trades if t.pnl <= 0]),
            'win_rate': len([t for t in self.trades if t.pnl > 0]) / len(self.trades),
            'profit_factor': abs(sum(t.pnl for t in self.trades if t.pnl > 0) / 
                               sum(t.pnl for t in self.trades if t.pnl < 0))
                               if sum(t.pnl for t in self.trades if t.pnl < 0) != 0 else np.inf,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() != 0 else 0,
            'max_drawdown': max(self.drawdown_curve),
            'total_return': (self.equity - self.config.initial_capital) / 
                          self.config.initial_capital,
            'avg_trade': np.mean([t.pnl for t in self.trades]),
            'std_trade': np.std([t.pnl for t in self.trades])
        }
