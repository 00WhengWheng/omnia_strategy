import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Union, Tuple
from abc import ABC, abstractmethod
import datetime

class BaseBacktester(ABC):
    """Classe base astratta per i backtester"""
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        self.data = data
        self.initial_capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    @abstractmethod
    def run(self):
        pass
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calcola tutte le metriche di performance"""
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        metrics = {
            'total_return': self.calculate_total_return(),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns),
            'profit_factor': self.calculate_profit_factor(),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate(),
            'var_95': self.calculate_var(returns, 0.95),
            'expected_shortfall': self.calculate_expected_shortfall(returns, 0.95)
        }
        return metrics
    
    def calculate_total_return(self) -> float:
        """Calcola il rendimento totale"""
        if not self.equity_curve:
            return 0
        return (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calcola lo Sharpe Ratio"""
        excess_returns = returns - risk_free_rate/252  # Assuming daily data
        if len(excess_returns) < 2:
            return 0
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calcola il Sortino Ratio"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2:
            return 0
        return np.sqrt(252) * (excess_returns.mean() / downside_returns.std())
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calcola il Calmar Ratio"""
        max_dd = self.calculate_max_drawdown()
        if max_dd == 0:
            return 0
        annual_return = returns.mean() * 252
        return annual_return / abs(max_dd)
    
    def calculate_profit_factor(self) -> float:
        """Calcola il Profit Factor"""
        profits = sum(trade for trade in self.trades if trade > 0)
        losses = sum(abs(trade) for trade in self.trades if trade < 0)
        return profits / losses if losses != 0 else float('inf')
    
    def calculate_max_drawdown(self) -> float:
        """Calcola il Maximum Drawdown"""
        peaks = pd.Series(self.equity_curve).expanding(min_periods=1).max()
        drawdowns = pd.Series(self.equity_curve) / peaks - 1
        return drawdowns.min()
    
    def calculate_win_rate(self) -> float:
        """Calcola il Win Rate"""
        if not self.trades:
            return 0
        winning_trades = sum(1 for trade in self.trades if trade > 0)
        return winning_trades / len(self.trades)
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calcola il Value at Risk"""
        return returns.quantile(1 - confidence)
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calcola l'Expected Shortfall (Conditional VaR)"""
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

class WalkForwardBacktester(BaseBacktester):
    """Implementa il Walk-Forward Analysis"""
    
    def __init__(self, data: pd.DataFrame, train_size: float = 0.6, 
                 validation_size: float = 0.2, initial_capital: float = 100000):
        super().__init__(data, initial_capital)
        self.train_size = train_size
        self.validation_size = validation_size
        
    def run(self, model, optimization_func):
        n = len(self.data)
        train_end = int(n * self.train_size)
        validation_end = int(n * (self.train_size + self.validation_size))
        
        # Training period
        train_data = self.data[:train_end]
        optimal_params = optimization_func(model, train_data)
        
        # Validation period
        validation_data = self.data[train_end:validation_end]
        model.set_params(**optimal_params)
        validation_results = model.backtest(validation_data)
        
        # Test period
        test_data = self.data[validation_end:]
        test_results = model.backtest(test_data)
        
        return validation_results, test_results

class TimeSeriesCVBacktester(BaseBacktester):
    """Implementa Time Series Cross-Validation"""
    
    def __init__(self, data: pd.DataFrame, n_splits: int = 5, 
                 test_size: int = 252, initial_capital: float = 100000):
        super().__init__(data, initial_capital)
        self.n_splits = n_splits
        self.test_size = test_size
        
    def run(self, model):
        results = []
        for i in range(self.n_splits):
            test_start = len(self.data) - (i + 1) * self.test_size
            test_end = len(self.data) - i * self.test_size
            
            train_data = self.data[:test_start]
            test_data = self.data[test_start:test_end]
            
            model.fit(train_data)
            fold_results = model.backtest(test_data)
            results.append(fold_results)
            
        return results

class RollingWindowBacktester(BaseBacktester):
    """Implementa Rolling Window Analysis"""
    
    def __init__(self, data: pd.DataFrame, window_size: int = 252, 
                 step_size: int = 21, initial_capital: float = 100000):
        super().__init__(data, initial_capital)
        self.window_size = window_size
        self.step_size = step_size
        
    def run(self, model):
        results = []
        for i in range(0, len(self.data) - self.window_size, self.step_size):
            window_data = self.data[i:i + self.window_size]
            model.fit(window_data)
            window_results = model.backtest(window_data)
            results.append(window_results)
            
        return results

class MonteCarloBacktester(BaseBacktester):
    """Implementa Monte Carlo Simulation"""
    
    def __init__(self, data: pd.DataFrame, n_simulations: int = 1000, 
                 initial_capital: float = 100000):
        super().__init__(data, initial_capital)
        self.n_simulations = n_simulations
        
    def run(self, model, returns_generator):
        results = []
        for _ in range(self.n_simulations):
            simulated_data = returns_generator(self.data)
            simulation_results = model.backtest(simulated_data)
            results.append(simulation_results)
            
        return results

def create_backtesting_report(backtest_results: Dict[str, float], 
                            trades: List[float], 
                            equity_curve: List[float]) -> pd.DataFrame:
    """
    Crea un report dettagliato del backtesting
    """
    report = pd.DataFrame()
    
    # Performance metrics
    report['Total Return'] = [backtest_results['total_return']]
    report['Sharpe Ratio'] = [backtest_results['sharpe_ratio']]
    report['Sortino Ratio'] = [backtest_results['sortino_ratio']]
    report['Calmar Ratio'] = [backtest_results['calmar_ratio']]
    report['Profit Factor'] = [backtest_results['profit_factor']]
    
    # Risk metrics
    report['Max Drawdown'] = [backtest_results['max_drawdown']]
    report['Win Rate'] = [backtest_results['win_rate']]
    report['VaR (95%)'] = [backtest_results['var_95']]
    report['Expected Shortfall'] = [backtest_results['expected_shortfall']]
    
    # Additional statistics
    report['Number of Trades'] = [len(trades)]
    report['Average Trade'] = [np.mean(trades)]
    report['Trade Std Dev'] = [np.std(trades)]
    
    return report

def plot_equity_curve(equity_curve: List[float], 
                     drawdowns: List[float], 
                     trades: List[Tuple[datetime.datetime, float]]):
    """
    Visualizza la curva dell'equity con drawdown e trade markers
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(equity_curve, label='Equity')
    ax1.set_title('Equity Curve')
    
    # Add trade markers
    for date, price in trades:
        ax1.scatter(date, price, color='g' if price > 0 else 'r')
    
    # Plot drawdowns
    ax2.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdowns')
    
    plt.tight_layout()
    return fig

"""
come utilizzarlo:

walk forward
backtester = WalkForwardBacktester(data, train_size=0.6, validation_size=0.2)
validation_results, test_results = backtester.run(your_model, optimization_function)

time series cv
backtester = TimeSeriesCVBacktester(data, n_splits=5, test_size=252)
results = backtester.run(your_model)

Rolling window
backtester = RollingWindowBacktester(data, window_size=252, step_size=21)
results = backtester.run(your_model)

Monte Carlo
backtester = MonteCarloBacktester(data, n_simulations=1000)
results = backtester.run(your_model, returns_generator_function)
"""