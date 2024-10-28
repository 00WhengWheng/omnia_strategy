from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import logging
from enum import Enum
import time

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    order_id: Optional[str] = None
    timestamp: datetime = datetime.now()
    metadata: Dict = None

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    metadata: Dict

class LiveTrader:
    def __init__(self, config: Dict):
        """Inizializza il trader live"""
        self.config = config
        self.broker = self._initialize_broker()
        self.data_manager = self._initialize_data_manager()
        self.risk_manager = self._initialize_risk_manager()
        self.strategy_manager = self._initialize_strategy_manager()
        
        # Order and position tracking
        self.open_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Order] = []
        
        # Queues for async processing
        self.order_queue = queue.Queue()
        self.data_queue = queue.Queue()
        
        # Threading controls
        self.is_running = False
        self.trading_thread = None
        self.data_thread = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Performance tracking
        self.performance_metrics = pd.DataFrame()
        
        # Market state
        self.market_hours = self._get_market_hours()
        self.is_market_open = False
        
    def start(self):
        """Avvia il sistema di trading"""
        self.logger.info("Starting live trading system...")
        self.is_running = True
        
        # Start threads
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.data_thread = threading.Thread(target=self._data_loop)
        
        self.trading_thread.start()
        self.data_thread.start()
        
        self.logger.info("Live trading system started")
        
    def stop(self):
        """Ferma il sistema di trading"""
        self.logger.info("Stopping live trading system...")
        self.is_running = False
        
        # Cancel all pending orders
        self._cancel_all_pending_orders()
        
        # Wait for threads to finish
        if self.trading_thread:
            self.trading_thread.join()
        if self.data_thread:
            self.data_thread.join()
            
        self.logger.info("Live trading system stopped")
        
    def submit_order(self, order: Order) -> bool:
        """Sottomette un nuovo ordine"""
        # Validate order
        if not self._validate_order(order):
            return False
            
        # Check risk limits
        if not self.risk_manager.validate_order(order, self.positions):
            return False
            
        # Submit to broker
        try:
            order_id = self.broker.submit_order(order)
            order.order_id = order_id
            order.status = OrderStatus.SUBMITTED
            self.open_orders[order_id] = order
            self.logger.info(f"Order submitted: {order}")
            return True
        except Exception as e:
            self.logger.error(f"Order submission failed: {str(e)}")
            return False
            
    def cancel_order(self, order_id: str) -> bool:
        """Cancella un ordine pendente"""
        if order_id not in self.open_orders:
            return False
            
        try:
            self.broker.cancel_order(order_id)
            order = self.open_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.open_orders[order_id]
            self.order_history.append(order)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Order cancellation failed: {str(e)}")
            return False
            
    def modify_order(self, order_id: str, modifications: Dict) -> bool:
        """Modifica un ordine esistente"""
        if order_id not in self.open_orders:
            return False
            
        try:
            self.broker.modify_order(order_id, modifications)
            order = self.open_orders[order_id]
            for key, value in modifications.items():
                setattr(order, key, value)
            self.logger.info(f"Order modified: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Order modification failed: {str(e)}")
            return False
            
    def _trading_loop(self):
        """Loop principale di trading"""
        while self.is_running:
            try:
                # Check market hours
                self.is_market_open = self._check_market_hours()
                
                if not self.is_market_open:
                    time.sleep(60)  # Check every minute
                    continue
                    
                # Process order queue
                while not self.order_queue.empty():
                    order = self.order_queue.get()
                    self.submit_order(order)
                    
                # Update positions
                self._update_positions()
                
                # Check stops and targets
                self._check_exit_conditions()
                
                # Generate new signals
                self._process_signals()
                
                time.sleep(1)  # Tempo tra iterazioni
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(5)
                
    def _data_loop(self):
        """Loop di aggiornamento dati"""
        while self.is_running:
            try:
                # Get market data updates
                market_data = self.data_manager.get_live_data()
                
                # Update data queue
                self.data_queue.put(market_data)
                
                # Update market state
                self._update_market_state(market_data)
                
                time.sleep(1)  # Intervallo aggiornamento
                
            except Exception as e:
                self.logger.error(f"Error in data loop: {str(e)}")
                time.sleep(5)
                
    def _update_positions(self):
        """Aggiorna le posizioni aperte"""
        try:
            # Get current positions from broker
            broker_positions = self.broker.get_positions()
            
            # Update internal position tracking
            for symbol, pos in broker_positions.items():
                if symbol in self.positions:
                    self.positions[symbol].current_price = pos['current_price']
                    self.positions[symbol].unrealized_pnl = pos['unrealized_pnl']
                else:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=pos['quantity'],
                        entry_price=pos['entry_price'],
                        current_price=pos['current_price'],
                        unrealized_pnl=pos['unrealized_pnl'],
                        realized_pnl=0.0,
                        metadata={}
                    )
                    
            # Remove closed positions
            for symbol in list(self.positions.keys()):
                if symbol not in broker_positions:
                    del self.positions[symbol]
                    
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            
    def _check_exit_conditions(self):
        """Verifica condizioni di uscita per le posizioni aperte"""
        for symbol, position in self.positions.items():
            try:
                # Check stop loss
                if self._check_stop_loss(position):
                    self._close_position(symbol, "stop_loss")
                    continue
                    
                # Check take profit
                if self._check_take_profit(position):
                    self._close_position(symbol, "take_profit")
                    continue
                    
                # Check trailing stop
                if self._check_trailing_stop(position):
                    self._close_position(symbol, "trailing_stop")
                    continue
                    
                # Check time stop
                if self._check_time_stop(position):
                    self._close_position(symbol, "time_stop")
                    
            except Exception as e:
                self.logger.error(f"Error checking exit conditions for {symbol}: {str(e)}")
                
    def _process_signals(self):
        """Processa i segnali delle strategie"""
        try:
            # Get latest market data
            market_data = self.data_queue.get_nowait()
            
            # Generate signals from strategies
            signals = self.strategy_manager.generate_signals(market_data)
            
            # Process each signal
            for signal in signals:
                # Validate signal
                if not self._validate_signal(signal):
                    continue
                    
                # Convert signal to order
                order = self._create_order_from_signal(signal)
                
                # Add to order queue
                self.order_queue.put(order)
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing signals: {str(e)}")
            
    def get_account_status(self) -> Dict:
        """Ritorna lo stato corrente dell'account"""
        return {
            'equity': self.broker.get_equity(),
            'available_funds': self.broker.get_available_funds(),
            'positions': len(self.positions),
            'open_orders': len(self.open_orders),
            'daily_pnl': self._calculate_daily_pnl(),
            'margin_used': self._calculate_margin_used()
        }
        
    def get_position_summary(self) -> pd.DataFrame:
        """Ritorna un sommario delle posizioni aperte"""
        if not self.positions:
            return pd.DataFrame()
            
        data = []
        for symbol, pos in self.positions.items():
            data.append({
                'symbol': symbol,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'pnl_percentage': (pos.current_price / pos.entry_price - 1) * 100
            })
            
        return pd.DataFrame(data)
        
    def _setup_logging(self):
        """Configura il logging"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        fh = logging.FileHandler('logs/live_trading.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Stream handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        
        self.logger.setLevel(logging.INFO)
