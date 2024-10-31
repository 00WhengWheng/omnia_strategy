import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class Position(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position: Position
    size: float
    pnl: float
    
class AdvancedBacktester:
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        transaction_costs: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 0.1
    ):
        """
        Inizializza il sistema di backtesting
        
        Args:
            data: DataFrame con OHLCV data
            initial_capital: Capitale iniziale
            transaction_costs: Costi di transazione (%)
            slippage: Slippage stimato (%)
            position_size: Dimensione posizione come % del capitale
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_costs = transaction_costs
        self.slippage = slippage
        self.position_size = position_size
        
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
        # Performance metrics
        self.metrics = {}
        
    def generate_signals(self, strategy_func) -> pd.Series:
        """
        Genera i segnali di trading usando la strategia fornita
        
        Args:
            strategy_func: Funzione che implementa la logica della strategia
            
        Returns:
            Series con i segnali (-1, 0, 1)
        """
        return strategy_func(self.data)
    
    def execute_trades(self, signals: pd.Series):
        """
        Esegue il backtesting basato sui segnali
        
        Args:
            signals: Series con i segnali di trading
        """
        current_position = Position.NEUTRAL
        
        for i in range(1, len(self.data)):
            signal = signals.iloc[i]
            
            # Chiudi posizione esistente
            if current_position != Position.NEUTRAL and signal != current_position.value:
                self._close_position(i)
                current_position = Position.NEUTRAL
            
            # Apri nuova posizione
            if current_position == Position.NEUTRAL and signal != 0:
                position_type = Position.LONG if signal == 1 else Position.SHORT
                self._open_position(i, position_type)
                current_position = position_type
            
            # Aggiorna equity curve
            self.equity_curve.append(self._calculate_equity(i))
    
    def _calculate_trade_size(self) -> float:
        """Calcola la dimensione della posizione"""
        return self.current_capital * self.position_size
    
    def _open_position(self, index: int, position_type: Position):
        """Apre una nuova posizione"""
        price = self.data['Close'].iloc[index]
        adjusted_price = price * (1 + self.slippage * position_type.value)
        size = self._calculate_trade_size() / adjusted_price
        
        self.positions.append({
            'entry_time': self.data.index[index],
            'entry_price': adjusted_price,
            'position_type': position_type,
            'size': size
        })
        
        # Applica costi di transazione
        self.current_capital -= abs(size * adjusted_price) * self.transaction_costs
    
    def _close_position(self, index: int):
        """Chiude la posizione corrente"""
        if not self.positions:
            return
            
        position = self.positions[-1]
        exit_price = self.data['Close'].iloc[index]
        adjusted_price = exit_price * (1 - self.slippage * position['position_type'].value)
        
        # Calcola P&L
        pnl = (adjusted_price - position['entry_price']) * position['size'] * position['position_type'].value
        self.current_capital += pnl
        
        # Registra trade
        self.trades.append(Trade(
            entry_time=position['entry_time'],
            exit_time=self.data.index[index],
            entry_price=position['entry_price'],
            exit_price=adjusted_price,
            position=position['position_type'],
            size=position['size'],
            pnl=pnl
        ))
        
        # Applica costi di transazione
        self.current_capital -= abs(position['size'] * adjusted_price) * self.transaction_costs
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calcola le metriche di performance"""
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        self.metrics = {
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor()
        }
        
        return self.metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calcola Sharpe Ratio annualizzato"""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self) -> float:
        """Calcola Maximum Drawdown"""
        equity_curve = pd.Series(self.equity_curve)
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        return abs(drawdowns.min())
    
    def _calculate_win_rate(self) -> float:
        """Calcola Win Rate"""
        if not self.trades:
            return 0.0
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        return winning_trades / len(self.trades)
    
    def _calculate_profit_factor(self) -> float:
        """Calcola Profit Factor"""
        if not self.trades:
            return 0.0
        gross_profits = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        gross_losses = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))
        return gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    def plot_results(self):
        """Visualizza i risultati del backtesting"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Equity Curve
        equity_curve = pd.Series(self.equity_curve, index=self.data.index)
        equity_curve.plot(ax=ax1, title='Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        
        # Drawdown
        drawdown = (equity_curve - equity_curve.expanding().max()) / equity_curve.expanding().max()
        drawdown.plot(ax=ax2, title='Drawdown', color='red')
        ax2.set_ylabel('Drawdown (%)')
        
        # Trade Distribution
        pnls = [trade.pnl for trade in self.trades]
        ax3.hist(pnls, bins=50, alpha=0.75)
        ax3.set_title('Trade P&L Distribution')
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

    def run_monte_carlo(self, n_simulations: int = 1000) -> Dict[str, np.ndarray]:
        """
        Esegue simulazione Monte Carlo per valutare la robustezza della strategia
        
        Args:
            n_simulations: Numero di simulazioni
            
        Returns:
            Dict con risultati delle simulazioni
        """
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        simulation_results = {
            'final_equity': np.zeros(n_simulations),
            'max_drawdown': np.zeros(n_simulations),
            'sharpe_ratio': np.zeros(n_simulations)
        }
        
        for i in range(n_simulations):
            # Genera scenario
            simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
            simulated_equity = self.initial_capital * (1 + simulated_returns).cumprod()
            
            # Calcola metriche
            simulation_results['final_equity'][i] = simulated_equity[-1]
            simulation_results['max_drawdown'][i] = self._calculate_max_drawdown_from_equity(simulated_equity)
            simulation_results['sharpe_ratio'][i] = np.sqrt(252) * simulated_returns.mean() / simulated_returns.std()
        
        return simulation_results
    
    def _calculate_max_drawdown_from_equity(self, equity: np.ndarray) -> float:
        """Calcola Maximum Drawdown da una serie di equity"""
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return abs(drawdown.min())