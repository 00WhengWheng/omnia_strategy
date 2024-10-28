{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Strategy Research and Development\n",
    "\n",
    "## 1. Inizializzazione e Import"
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
    "from datetime import datetime, timedelta\n",
    "\n",
    "from src.strategies.trend_following import TrendFollowingStrategy\n",
    "from src.analysis.technical import TechnicalAnalyzer\n",
    "from src.analysis.correlation import CorrelationAnalyzer\n",
    "\n",
    "%matplotlib inline\n",
    "plt.style.use('seaborn')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Caricamento e Preparazione Dati"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Carica dati di mercato\n",
    "data = pd.read_csv('../../data/raw/market_data.csv', index_col='timestamp', parse_dates=True)\n",
    "\n",
    "# Mostra prime righe\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Analisi Tecnica"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Inizializza analizzatore tecnico\n",
    "tech_analyzer = TechnicalAnalyzer({})\n",
    "\n",
    "# Calcola indicatori\n",
    "analysis = tech_analyzer.analyze(data)\n",
    "\n",
    "# Plot indicatori principali\n",
    "fig, axes = plt.subplots(3, 1, figsize=(15, 10))\n",
    "\n",
    "# Prezzo e medie mobili\n",
    "axes[0].plot(data.index, data['close'], label='Price')\n",
    "axes[0].plot(data.index, analysis['sma_20'], label='SMA 20')\n",
    "axes[0].plot(data.index, analysis['sma_50'], label='SMA 50')\n",
    "axes[0].set_title('Price and Moving Averages')\n",
    "axes[0].legend()\n",
    "\n",
    "# RSI\n",
    "axes[1].plot(data.index, analysis['rsi'])\n",
    "axes[1].axhline(y=70, color='r', linestyle='--')\n",
    "axes[1].axhline(y=30, color='g', linestyle='--')\n",
    "axes[1].set_title('RSI')\n",
    "\n",
    "# Volume\n",
    "axes[2].bar(data.index, data['volume'])\n",
    "axes[2].set_title('Volume')\n",
    "\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Analisi delle Correlazioni"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Inizializza analizzatore correlazioni\n",
    "corr_analyzer = CorrelationAnalyzer({})\n",
    "\n",
    "# Calcola correlazioni\n",
    "correlations = corr_analyzer.analyze(data)\n",
    "\n",
    "# Plot heatmap correlazioni\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(correlations['correlation_matrix'], annot=True, cmap='coolwarm')\n",
    "plt.title('Correlation Matrix')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Test Strategia"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Inizializza strategia\n",
    "strategy = TrendFollowingStrategy({\n",
    "    'fast_ma': 20,\n",
    "    'slow_ma': 50,\n",
    "    'stop_loss': 0.02,\n",
    "    'take_profit': 0.04\n",
    "})\n",
    "\n",
    "# Genera segnali\n",
    "signals = strategy.generate_signals(data)\n",
    "\n",
    "# Plot segnali\n",
    "plt.figure(figsize=(15, 6))\n",
    "plt.plot(data.index, data['close'])\n",
    "\n",
    "# Plot entry points\n",
    "longs = signals[signals['direction'] == 'long']\n",
    "shorts = signals[signals['direction'] == 'short']\n",
    "\n",
    "plt.scatter(longs.index, data.loc[longs.index]['close'], \n",
    "           marker='^', color='g', label='Long')\n",
    "plt.scatter(shorts.index, data.loc[shorts.index]['close'], \n",
    "           marker='v', color='r', label='Short')\n",
    "\n",
    "plt.title('Strategy Signals')\n",
    "plt.legend()\n",
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
