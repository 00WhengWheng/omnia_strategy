import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

@dataclass
class TradeMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_mae: float
    average_mfe: float
    average_duration: timedelta
    risk_reward_ratio: float
    payoff_ratio: float
    profit_per_month: float
    sharpe_ratio: float
    sortino_ratio: float

class TradeAnalytics:
    def __init__(self, config: Dict):
        """Inizializza Trade Analytics Engine"""
        self.config = config
        
        # Storage for analyses
        self.trade_metrics = None
        self.trade_distributions = {}
        self.trade_patterns = {}
        self.time_analysis = {}
        self.risk_analysis = {}
        
        # Performance tracking
        self.equity_curve = pd.Series()
        self.drawdown_curve = pd.Series()
        self.monthly_returns = pd.Series()
        
        # Analysis settings
        self.min_trades = config.get('min_trades', 30)
        self.confidence_level = config.get('confidence_level', 0.95)
        
    def analyze_trades(self, trades: List[Dict],
                      portfolio_value: float) -> Dict:
        """Analisi completa dei trade"""
        if len(trades) < self.min_trades:
            return self._generate_limited_analysis(trades)
            
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate base metrics
        self.trade_metrics = self._calculate_trade_metrics(trades_df, portfolio_value)
        
        # Analyze trade distributions
        self.trade_distributions = self._analyze_distributions(trades_df)
        
        # Analyze trade patterns
        self.trade_patterns = self._analyze_patterns(trades_df)
        
        # Time-based analysis
        self.time_analysis = self._analyze_time_patterns(trades_df)
        
        # Risk analysis
        self.risk_analysis = self._analyze_risk_metrics(trades_df, portfolio_value)
        
        return self._generate_analysis_report()
        
    def _calculate_trade_metrics(self, trades: pd.DataFrame,
                               portfolio_value: float) -> TradeMetrics:
        """Calcola metriche base dei trade"""
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] <= 0]
        
        total_trades = len(trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        win_rate = winning_count / total_trades if total_trades > 0 else 0
        
        total_profits = winning_trades['pnl'].sum()
        total_losses = abs(losing_trades['pnl'].sum())
        profit_factor = total_profits / total_losses if total_losses != 0 else np.inf
        
        return TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=winning_trades['pnl'].mean() if winning_count > 0 else 0,
            average_loss=losing_trades['pnl'].mean() if losing_count > 0 else 0,
            largest_win=winning_trades['pnl'].max() if winning_count > 0 else 0,
            largest_loss=losing_trades['pnl'].min() if losing_count > 0 else 0,
            average_mae=trades['mae'].mean(),
            average_mfe=trades['mfe'].mean(),
            average_duration=pd.Timedelta(trades['duration'].mean()),
            risk_reward_ratio=self._calculate_risk_reward_ratio(trades),
            payoff_ratio=self._calculate_payoff_ratio(trades),
            profit_per_month=self._calculate_monthly_profit(trades, portfolio_value),
            sharpe_ratio=self._calculate_sharpe_ratio(trades),
            sortino_ratio=self._calculate_sortino_ratio(trades)
        )
        
    def _analyze_distributions(self, trades: pd.DataFrame) -> Dict:
        """Analizza distribuzioni dei trade"""
        distributions = {}
        
        # PnL distribution
        distributions['pnl'] = {
            'mean': trades['pnl'].mean(),
            'std': trades['pnl'].std(),
            'skew': trades['pnl'].skew(),
            'kurtosis': trades['pnl'].kurtosis(),
            'percentiles': trades['pnl'].describe(percentiles=[.05, .25, .50, .75, .95])
        }
        
        # Duration distribution
        distributions['duration'] = {
            'mean': trades['duration'].mean(),
            'std': trades['duration'].std(),
            'percentiles': trades['duration'].describe(percentiles=[.05, .25, .50, .75, .95])
        }
        
        # MAE/MFE distribution
        distributions['mae_mfe'] = {
            'mae_mean': trades['mae'].mean(),
            'mae_std': trades['mae'].std(),
            'mfe_mean': trades['mfe'].mean(),
            'mfe_std': trades['mfe'].std()
        }
        
        return distributions
        
    def _analyze_patterns(self, trades: pd.DataFrame) -> Dict:
        """Analizza pattern nei trade"""
        patterns = {}
        
        # Winning/losing streaks
        patterns['streaks'] = self._analyze_streaks(trades)
        
        # Trade clustering
        patterns['clustering'] = self._analyze_trade_clustering(trades)
        
        # Sequential patterns
        patterns['sequences'] = self._analyze_trade_sequences(trades)
        
        # Strategy patterns
        patterns['strategies'] = self._analyze_strategy_patterns(trades)
        
        return patterns
        
    def _analyze_time_patterns(self, trades: pd.DataFrame) -> Dict:
        """Analizza pattern temporali"""
        time_analysis = {}
        
        # Hourly analysis
        time_analysis['hourly'] = self._analyze_hourly_performance(trades)
        
        # Daily analysis
        time_analysis['daily'] = self._analyze_daily_performance(trades)
        
        # Monthly analysis
        time_analysis['monthly'] = self._analyze_monthly_performance(trades)
        
        # Seasonality
        time_analysis['seasonality'] = self._analyze_seasonality(trades)
        
        return time_analysis
        
    def _analyze_risk_metrics(self, trades: pd.DataFrame,
                            portfolio_value: float) -> Dict:
        """Analizza metriche di rischio"""
        risk_metrics = {}
        
        # Value at Risk
        risk_metrics['var'] = self._calculate_var(trades, portfolio_value)
        
        # Expected Shortfall
        risk_metrics['expected_shortfall'] = self._calculate_expected_shortfall(trades)
        
        # Risk-adjusted returns
        risk_metrics['risk_adjusted'] = self._calculate_risk_adjusted_metrics(trades)
        
        # Drawdown analysis
        risk_metrics['drawdown'] = self._analyze_drawdowns(trades)
        
        return risk_metrics
        
    def _analyze_streaks(self, trades: pd.DataFrame) -> Dict:
        """Analizza serie vincenti/perdenti"""
        streaks = []
        current_streak = 0
        
        for pnl in trades['pnl']:
            if pnl > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    if current_streak != 0:
                        streaks.append(current_streak)
                    current_streak = 1
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    if current_streak != 0:
                        streaks.append(current_streak)
                    current_streak = -1
                    
        if current_streak != 0:
            streaks.append(current_streak)
            
        return {
            'max_winning_streak': max([s for s in streaks if s > 0], default=0),
            'max_losing_streak': abs(min([s for s in streaks if s < 0], default=0)),
            'avg_winning_streak': np.mean([s for s in streaks if s > 0]) if any(s > 0 for s in streaks) else 0,
            'avg_losing_streak': abs(np.mean([s for s in streaks if s < 0])) if any(s < 0 for s in streaks) else 0,
            'streak_distribution': pd.Series(streaks).value_counts().to_dict()
        }
        
    def _analyze_trade_clustering(self, trades: pd.DataFrame) -> Dict:
        """Analizza clustering dei trade"""
        # Time between trades
        trades['time_between'] = trades['entry_time'].diff()
        
        # Trade density
        trade_density = self._calculate_trade_density(trades)
        
        # Clustering coefficient
        clustering = self._calculate_clustering_coefficient(trades)
        
        return {
            'avg_time_between': trades['time_between'].mean(),
            'trade_density': trade_density,
            'clustering_coefficient': clustering
        }
        
    def _analyze_trade_sequences(self, trades: pd.DataFrame) -> Dict:
        """Analizza sequenze di trade"""
        # Win/loss sequences
        sequences = self._find_trade_sequences(trades)
        
        # Sequence probabilities
        probabilities = self._calculate_sequence_probabilities(sequences)
        
        # Markov chain analysis
        markov = self._analyze_markov_chain(trades)
        
        return {
            'sequences': sequences,
            'probabilities': probabilities,
            'markov_chain': markov
        }
        
    def generate_report(self, format: str = 'dict') -> Union[Dict, str]:
        """Genera report di analisi"""
        if format == 'dict':
            return self._generate_analysis_report()
        elif format == 'html':
            return self._generate_html_report()
        elif format == 'pdf':
            return self._generate_pdf_report()
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def plot_analytics(self) -> Dict:
        """Genera grafici di analisi"""
        plots = {}
        
        # Equity curve
        plots['equity_curve'] = self._plot_equity_curve()
        
        # PnL distribution
        plots['pnl_distribution'] = self._plot_pnl_distribution()
        
        # MAE/MFE analysis
        plots['mae_mfe'] = self._plot_mae_mfe_analysis()
        
        # Time analysis
        plots['time_analysis'] = self._plot_time_analysis()
        
        return plots
        
    def _plot_equity_curve(self):
        """Plot della equity curve"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.equity_curve, label='Equity')
        ax.fill_between(self.drawdown_curve.index, 0, 
                       self.drawdown_curve.values, alpha=0.3, color='red')
        ax.set_title('Equity Curve with Drawdowns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.legend()
        return fig
        
    def _plot_pnl_distribution(self):
        """Plot della distribuzione PnL"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=self.trade_distributions['pnl'], ax=ax)
        ax.set_title('PnL Distribution')
        ax.set_xlabel('PnL')
        ax.set_ylabel('Frequency')
        return fig
        
    def _plot_mae_mfe_analysis(self):
        """Plot analisi MAE/MFE"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.trade_distributions['mae_mfe']['mae_mean'],
                  self.trade_distributions['mae_mfe']['mfe_mean'])
        ax.set_title('MAE vs MFE Analysis')
        ax.set_xlabel('Maximum Adverse Excursion')
        ax.set_ylabel('Maximum Favorable Excursion')
        return fig
        
    def _plot_time_analysis(self):
        """Plot analisi temporale"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Hourly performance
        sns.barplot(data=self.time_analysis['hourly'], ax=axes[0,0])
        axes[0,0].set_title('Hourly Performance')
        
        # Daily performance
        sns.barplot(data=self.time_analysis['daily'], ax=axes[0,1])
        axes[0,1].set_title('Daily Performance')
        
        # Monthly performance
        sns.barplot(data=self.time_analysis['monthly'], ax=axes[1,0])
        axes[1,0].set_title('Monthly Performance')
        
        # Seasonality
        sns.lineplot(data=self.time_analysis['seasonality'], ax=axes[1,1])
        axes[1,1].set_title('Seasonality Analysis')
        
        plt.tight_layout()
        return fig
