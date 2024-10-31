import numpy as np
import pandas as pd
from typing import List, Dict, Union, Callable
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class BacktestResults:
    """Classe per salvare i risultati del backtesting"""
    returns: pd.Series
    positions: pd.Series
    equity_curve: pd.Series
    metrics: Dict[str, float]
    trades: pd.DataFrame

class VectorizedBacktester:
    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.001,
                 position_size: float = 1.0):
        """
        Args:
            initial_capital: Capitale iniziale
            commission: Commissioni per trade
            slippage: Slippage stimato
            position_size: Dimensione posizione (1.0 = 100% del capitale)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        
    def _calculate_metrics(self,
                          returns: pd.Series,
                          positions: pd.Series,
                          trades: pd.DataFrame) -> Dict[str, float]:
        """Calcola le metriche di performance"""
        # Metriche base
        total_return = (returns + 1).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        daily_std = returns.std()
        annualized_std = daily_std * np.sqrt(252)
        
        # Sharpe Ratio
        risk_free_rate = 0.02  # tasso risk-free annualizzato
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / negative_returns.std()
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown)
        
        # Trade metrics
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_std': annualized_std,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'number_of_trades': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
    def _generate_signals(self,
                         predictions: np.ndarray,
                         threshold: float = 0.0) -> pd.Series:
        """Genera segnali di trading dai predictions"""
        return pd.Series(np.where(predictions > threshold, 1,
                                np.where(predictions < -threshold, -1, 0)))
    
    def backtest(self,
                prices: pd.Series,
                predictions: np.ndarray,
                threshold: float = 0.0) -> BacktestResults:
        """Esegue il backtesting vectorized"""
        # Genera segnali
        signals = self._generate_signals(predictions, threshold)
        
        # Calcola returns
        price_returns = prices.pct_change()
        
        # Calcola posizioni
        positions = signals.shift(1)  # Implementa i trade al giorno successivo
        
        # Calcola returns della strategia
        strategy_returns = positions * price_returns
        strategy_returns = strategy_returns - self.commission * abs(positions.diff())  # Commissioni
        strategy_returns = strategy_returns * (1 - self.slippage)  # Slippage
        
        # Calcola equity curve
        equity_curve = (1 + strategy_returns).cumprod() * self.initial_capital
        
        # Identifica trades
        trades = pd.DataFrame()
        trades['entry_date'] = pd.Series(dtype='datetime64[ns]')
        trades['exit_date'] = pd.Series(dtype='datetime64[ns]')
        trades['direction'] = pd.Series(dtype='int64')
        trades['entry_price'] = pd.Series(dtype='float64')
        trades['exit_price'] = pd.Series(dtype='float64')
        trades['pnl'] = pd.Series(dtype='float64')
        
        position_changes = positions.diff()
        entry_dates = position_changes[position_changes != 0].index
        
        for i in range(len(entry_dates)-1):
            entry_date = entry_dates[i]
            exit_date = entry_dates[i+1]
            direction = positions[entry_date]
            entry_price = prices[entry_date]
            exit_price = prices[exit_date]
            pnl = direction * (exit_price - entry_price) / entry_price
            
            trades = pd.concat([trades, pd.DataFrame({
                'entry_date': [entry_date],
                'exit_date': [exit_date],
                'direction': [direction],
                'entry_price': [entry_price],
                'exit_price': [exit_price],
                'pnl': [pnl]
            })])
        
        # Calcola metriche
        metrics = self._calculate_metrics(strategy_returns, positions, trades)
        
        return BacktestResults(
            returns=strategy_returns,
            positions=positions,
            equity_curve=equity_curve,
            metrics=metrics,
            trades=trades
        )

class WalkForwardOptimizer:
    def __init__(self,
                 model_class: object,
                 param_grid: Dict,
                 train_window: int = 252,
                 test_window: int = 63,
                 step_size: int = 21):
        """
        Args:
            model_class: Classe del modello da ottimizzare
            param_grid: Griglia di parametri da testare
            train_window: Dimensione finestra di training (giorni)
            test_window: Dimensione finestra di test (giorni)
            step_size: Dimensione step per walk forward (giorni)
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        
    def _generate_windows(self, data: pd.DataFrame) -> List[tuple]:
        """Genera finestre per walk forward optimization"""
        windows = []
        start = 0
        while start + self.train_window + self.test_window <= len(data):
            train_start = start
            train_end = start + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window
            
            windows.append((
                (train_start, train_end),
                (test_start, test_end)
            ))
            
            start += self.step_size
            
        return windows
    
    def optimize(self,
                data: pd.DataFrame,
                target_col: str,
                feature_cols: List[str]) -> Dict:
        """Esegue walk forward optimization"""
        windows = self._generate_windows(data)
        results = []
        
        for train_idx, test_idx in windows:
            train_data = data.iloc[train_idx[0]:train_idx[1]]
            test_data = data.iloc[test_idx[0]:test_idx[1]]
            
            best_sharpe = -np.inf
            best_params = None
            
            # Grid search su ogni finestra
            for params in self._generate_param_combinations():
                model = self.model_class(**params)
                model.fit(
                    train_data[feature_cols],
                    train_data[target_col]
                )
                
                predictions = model.predict(test_data[feature_cols])
                
                # Backtest sulla finestra di test
                backtester = VectorizedBacktester()
                results = backtester.backtest(
                    test_data[target_col],
                    predictions
                )
                
                if results.metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = results.metrics['sharpe_ratio']
                    best_params = params
            
            results.append({
                'window_start': data.index[test_idx[0]],
                'window_end': data.index[test_idx[1]],
                'best_params': best_params,
                'sharpe_ratio': best_sharpe
            })
        
        return pd.DataFrame(results)
    
    def _generate_param_combinations(self) -> List[Dict]:
        """Genera tutte le combinazioni di parametri"""
        import itertools
        
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]

class BacktestAnalyzer:
    @staticmethod
    def plot_equity_curve(results: BacktestResults) -> None:
        """Visualizza equity curve"""
        plt.figure(figsize=(12, 6))
        results.equity_curve.plot()
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show()
        
    @staticmethod
    def plot_drawdown(results: BacktestResults) -> None:
        """Visualizza drawdown"""
        equity = results.equity_curve
        rolling_max = equity.expanding().max()
        drawdown = equity / rolling_max - 1
        
        plt.figure(figsize=(12, 6))
        drawdown.plot()
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.show()
        
    @staticmethod
    def plot_monthly_returns(results: BacktestResults) -> None:
        """Visualizza returns mensili"""
        monthly_returns = results.returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        plt.figure(figsize=(12, 6))
        monthly_returns.plot(kind='bar')
        plt.title('Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Return')
        plt.grid(True)
        plt.show()
        
    @staticmethod
    def print_metrics(results: BacktestResults) -> None:
        """Stampa metriche principali"""
        metrics = results.metrics
        print("\n=== Performance Metrics ===")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Number of Trades: {metrics['number_of_trades']}")

# Esempio di utilizzo
if __name__ == "__main__":
    """
    # Dati di esempio
    data = pd.DataFrame({
        'close': [100, 101, 99, 102, 103, 98],
        'volume': [1000, 1100, 900, 1200, 1300, 800]
    }, index=pd.date_range('2023-01-01', periods=6))
    
    # Modello esempio
    class SimpleModel:
        def __init__(self, window=2):
            self.window = window
            
        def fit(self, X, y):
            pass
            
        def predict(self, X):
            return np.random.randn(len(X))
    
    # Grid di parametri
    param_grid = {
        'window': [2, 3, 4]
    }
    
    # Walk Forward Optimization
    optimizer = WalkForwardOptimizer(
        SimpleModel,
        param_grid,
        train_window=3,
        test_window=2,
        step_size=1
    )
    
    optimization_results = optimizer.optimize(
        data,
        target_col='close',
        feature_cols=['volume']
    )
    
    # Backtesting
    backtester = VectorizedBacktester(
        initial_capital=100000,
        commission=0.001
    )
    
    # Genera prediction di esempio
    predictions = np.random.randn(len(data))
    
    results = backtester.backtest(
        data['close'],
        predictions
    )
    
    # Analisi risultati
    analyzer = BacktestAnalyzer()
    analyzer.print_metrics(results)
    analyzer.plot_equity_curve(results)
    analyzer.plot_drawdown(results)
    analyzer.plot_monthly_returns(results)
    """