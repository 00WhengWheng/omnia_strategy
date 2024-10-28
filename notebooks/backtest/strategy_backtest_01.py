{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Strategy Backtest Analysis\n",
    "\n",
    "## 1. Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import sys\n",
    "sys.path.append('../..')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from src.backtesting.engine import BacktestEngine\n",
    "from src.strategies.trend_following import TrendFollowingStrategy\n",
    "from src.analysis.trade_analytics import TradeAnalytics\n",
    "\n",
    "%matplotlib inline\n",
    "plt.style.use('seaborn')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Configurazione Backtest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Configurazione backtest\n",
    "config = {\n",
    "    'initial_capital': 100000,\n",
    "    'commission': 0.001,\n",
    "    'slippage': 0.0001,\n",
    "    'position_size': 0.02,\n",
    "    'max_positions': 5\n",
    "}\n",
    "\n",
    "# Inizializza engine\n",
    "engine = BacktestEngine(config)\n",
    "\n",
    "# Inizializza strategia\n",
    "strategy = TrendFollowingStrategy({\n",
    "    'fast_ma': 20,\n",
    "    'slow_ma': 50,\n",
    "    'stop_loss': 0.02,\n",
    "    'take_profit': 0.04\n",
    "})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Esecuzione Backtest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Carica dati\n",
    "data = pd.read_csv('../../data/raw/market_data.csv', index_col='timestamp', parse_dates=True)\n",
    "\n",
    "# Esegui backtest\n",
    "results = engine.run_backtest(strategy, data)\n",
    "\n",
    "# Mostra risultati base\n",
    "print('Total Return: {:.2%}'.format(results.metrics['total_return']))\n",
    "print('Sharpe Ratio: {:.2f}'.format(results.metrics['sharpe_ratio']))\n",
    "print('Max Drawdown: {:.2%}'.format(results.metrics['max_drawdown']))\n",
    "print('Win Rate: {:.2%}'.format(results.metrics['win_rate']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Analisi Performance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Plot equity curve\n",
    "plt.figure(figsize=(15, 6))\n",
    "plt.plot(results.equity_curve, label='Equity')\n",
    "plt.fill_between(results.drawdown_curve.index, 0, \n",
    "                 results.drawdown_curve.values, alpha=0.3, color='red')\n",
    "plt.title('Equity Curve and Drawdowns')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "# Distribuzione returns\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(results.daily_returns, kde=True)\n",
    "plt.title('Distribution of Daily Returns')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Analisi Trade"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Inizializza trade analyzer\n",
    "trade_analyzer = TradeAnalytics({})\n",
    "\n",
    "# Analizza trades\n",
    "trade_analysis = trade_analyzer.analyze_trades(results.trades)\n",
    "\n",
    "# Plot trade metrics\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
    "\n",
    "# PnL distribution\n",
    "sns.histplot(data=[t.pnl for t in results.trades], ax=axes[0,0])\n",
    "axes[0,0].set_title('PnL Distribution')\n",
    "\n",
    "# Trade duration\n",
    "sns.histplot(data=[t.duration.total_seconds()/3600 for t in results.trades], ax=axes[0,1])\n",
    "axes[0,1].set_title('Trade Duration (hours)')\n",
    "\n",
    "# MAE vs MFE\n",
    "axes[1,0].scatter([t.mae for t in results.trades], \n",
    "                 [t.mfe for t in results.trades])\n",
    "axes[1,0].set_title('MAE vs MFE')\n",
    "\n",
    "# Win rate by month\n",
    "monthly_wr = trade_analysis['time_analysis']['monthly']['win_rate']\n",
    "monthly_wr.plot(kind='bar', ax=axes[1,1])\n",
    "axes[1,1].set_title('Monthly Win Rate')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Risk Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Risk metrics\n",
    "risk_metrics = trade_analysis['risk_analysis']\n",
    "\n",
    "# Plot risk metrics\n",
    "fig, axes = plt.subplots(2, 1, figsize=(15, 10))\n",
    "\n",
    "# Rolling Sharpe ratio\n",
    "results.metrics['rolling_sharpe'].plot(ax=axes[0])\n",
    "axes[0].set_title('Rolling Sharpe Ratio')\n",
    "\n",
    "# Rolling VaR\n",
    "results.metrics['rolling_var'].plot(ax=axes[1])\n",
    "axes[1].set_title('Rolling Value at Risk')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
