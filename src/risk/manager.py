from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
from ..core.constants import TimeFrame

@dataclass
class RiskLimits:
    max_position_size: float          # Massima dimensione singola posizione
    max_portfolio_risk: float         # Massimo rischio portfolio
    max_correlation: float            # Massima correlazione tra posizioni
    max_sector_exposure: float        # Massima esposizione per settore
    max_drawdown: float              # Massimo drawdown accettabile
    position_scaling: bool            # Scaling dinamico delle posizioni
    risk_scaling: bool               # Scaling basato su volatilità
    
@dataclass
class PortfolioState:
    total_equity: float
    open_positions: Dict
    realized_pnl: float
    unrealized_pnl: float
    current_drawdown: float
    peak_equity: float
    position_history: List
    risk_metrics: Dict

class RiskManager:
    def __init__(self, config: Dict):
        """Inizializza Risk Manager"""
        self.config = config
        
        # Risk Limits
        self.limits = RiskLimits(
            max_position_size=config.get('max_position_size', 0.1),
            max_portfolio_risk=config.get('max_portfolio_risk', 0.2),
            max_correlation=config.get('max_correlation', 0.7),
            max_sector_exposure=config.get('max_sector_exposure', 0.3),
            max_drawdown=config.get('max_drawdown', 0.2),
            position_scaling=config.get('position_scaling', True),
            risk_scaling=config.get('risk_scaling', True)
        )
        
        # Portfolio State
        self.portfolio = PortfolioState(
            total_equity=config.get('initial_equity', 100000),
            open_positions={},
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            current_drawdown=0.0,
            peak_equity=config.get('initial_equity', 100000),
            position_history=[],
            risk_metrics={}
        )
        
        # Risk Monitoring
        self.risk_metrics = {
            'var': pd.Series(),
            'expected_shortfall': pd.Series(),
            'sharpe_ratio': pd.Series(),
            'correlation_matrix': pd.DataFrame()
        }
        
        # Performance Tracking
        self.performance_metrics = pd.DataFrame()
        
    def calculate_position_size(self, signal: Dict, risk_per_trade: float) -> float:
        """Calcola dimensione della posizione"""
        # Base position size basata su equity e risk
        base_size = self.portfolio.total_equity * risk_per_trade
        
        # Aggiusta per volatilità se abilitato
        if self.limits.risk_scaling:
            volatility_scalar = self._calculate_volatility_scalar(signal)
            base_size *= volatility_scalar
            
        # Aggiusta per correlazione portfolio
        correlation_scalar = self._calculate_correlation_scalar(signal)
        base_size *= correlation_scalar
        
        # Applica scaling dinamico se abilitato
        if self.limits.position_scaling:
            performance_scalar = self._calculate_performance_scalar()
            base_size *= performance_scalar
            
        # Applica limiti
        max_size = self.portfolio.total_equity * self.limits.max_position_size
        position_size = min(base_size, max_size)
        
        return position_size
        
    def validate_trade(self, signal: Dict, position_size: float) -> bool:
        """Valida un potenziale trade contro i limiti di rischio"""
        # Verifica limiti portfolio
        if not self._check_portfolio_limits(position_size):
            return False
            
        # Verifica correlazione
        if not self._check_correlation_limits(signal):
            return False
            
        # Verifica esposizione settoriale
        if not self._check_sector_exposure(signal):
            return False
            
        # Verifica drawdown
        if not self._check_drawdown_limits():
            return False
            
        return True
        
    def update_portfolio(self, trade: Dict) -> None:
        """Aggiorna stato del portfolio dopo un trade"""
        # Aggiorna posizioni
        if trade['action'] == 'open':
            self._open_position(trade)
        elif trade['action'] == 'close':
            self._close_position(trade)
            
        # Aggiorna equity
        self._update_equity()
        
        # Aggiorna metriche di rischio
        self._update_risk_metrics()
        
        # Aggiorna performance
        self._update_performance_metrics()
        
    def _calculate_volatility_scalar(self, signal: Dict) -> float:
        """Calcola fattore di scala basato sulla volatilità"""
        current_vol = signal.get('volatility', 1.0)
        avg_vol = self.risk_metrics.get('average_volatility', 1.0)
        
        if avg_vol == 0:
            return 1.0
            
        vol_ratio = current_vol / avg_vol
        
        # Riduce size quando volatilità è alta
        if vol_ratio > 1:
            return 1 / vol_ratio
            
        return 1.0
        
    def _calculate_correlation_scalar(self, signal: Dict) -> float:
        """Calcola fattore di scala basato sulla correlazione"""
        if not self.portfolio.open_positions:
            return 1.0
            
        correlations = []
        for position in self.portfolio.open_positions.values():
            corr = self._calculate_correlation(signal, position)
            correlations.append(abs(corr))
            
        avg_correlation = np.mean(correlations)
        
        # Riduce size per alte correlazioni
        if avg_correlation > self.limits.max_correlation:
            return 1 - (avg_correlation - self.limits.max_correlation)
            
        return 1.0
        
    def _calculate_performance_scalar(self) -> float:
        """Calcola fattore di scala basato sulla performance"""
        if not self.position_history:
            return 1.0
            
        # Calcola win rate recente
        recent_trades = self.position_history[-20:]
        win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades)
        
        # Calcola profit factor
        profit_factor = self._calculate_profit_factor(recent_trades)
        
        # Scala basato su performance
        performance_scalar = (win_rate * 0.5 + min(profit_factor, 2) / 2 * 0.5)
        
        return np.clip(performance_scalar, 0.5, 1.5)
        
    def _check_portfolio_limits(self, position_size: float) -> bool:
        """Verifica limiti del portfolio"""
        # Verifica risk totale
        current_risk = sum(pos['risk'] for pos in self.portfolio.open_positions.values())
        total_risk = (current_risk + position_size) / self.portfolio.total_equity
        
        if total_risk > self.limits.max_portfolio_risk:
            return False
            
        # Verifica numero posizioni
        max_positions = self.config.get('max_positions', 10)
        if len(self.portfolio.open_positions) >= max_positions:
            return False
            
        return True
        
    def _check_correlation_limits(self, signal: Dict) -> bool:
        """Verifica limiti di correlazione"""
        if not self.portfolio.open_positions:
            return True
            
        for position in self.portfolio.open_positions.values():
            correlation = abs(self._calculate_correlation(signal, position))
            if correlation > self.limits.max_correlation:
                return False
                
        return True
        
    def _check_sector_exposure(self, signal: Dict) -> bool:
        """Verifica limiti di esposizione settoriale"""
        sector = signal.get('sector', 'unknown')
        
        # Calcola esposizione corrente del settore
        sector_exposure = sum(
            pos['size'] for pos in self.portfolio.open_positions.values()
            if pos.get('sector') == sector
        )
        
        sector_exposure = sector_exposure / self.portfolio.total_equity
        
        return sector_exposure < self.limits.max_sector_exposure
        
    def _check_drawdown_limits(self) -> bool:
        """Verifica limiti di drawdown"""
        if self.portfolio.current_drawdown > self.limits.max_drawdown:
            return False
        return True
        
    def _update_equity(self) -> None:
        """Aggiorna equity del portfolio"""
        current_equity = (
            self.portfolio.total_equity +
            self.portfolio.realized_pnl +
            self.portfolio.unrealized_pnl
        )
        
        # Aggiorna peak equity
        if current_equity > self.portfolio.peak_equity:
            self.portfolio.peak_equity = current_equity
            
        # Aggiorna drawdown
        self.portfolio.current_drawdown = max(0, 1 - current_equity / self.portfolio.peak_equity)
        
        # Aggiorna equity totale
        self.portfolio.total_equity = current_equity
        
    def _update_risk_metrics(self) -> None:
        """Aggiorna metriche di rischio"""
        returns = self._calculate_returns()
        
        # Value at Risk
        self.risk_metrics['var'] = self._calculate_var(returns)
        
        # Expected Shortfall
        self.risk_metrics['expected_shortfall'] = self._calculate_es(returns)
        
        # Sharpe Ratio
        self.risk_metrics['sharpe_ratio'] = self._calculate_sharpe(returns)
        
        # Correlation Matrix
        self.risk_metrics['correlation_matrix'] = self._calculate_correlation_matrix()
        
    def get_risk_report(self) -> Dict:
        """Genera report completo di rischio"""
        return {
            'portfolio_state': {
                'total_equity': self.portfolio.total_equity,
                'open_positions': len(self.portfolio.open_positions),
                'realized_pnl': self.portfolio.realized_pnl,
                'unrealized_pnl': self.portfolio.unrealized_pnl,
                'current_drawdown': self.portfolio.current_drawdown
            },
            'risk_metrics': {
                'var_95': self.risk_metrics['var'].iloc[-1] if len(self.risk_metrics['var']) > 0 else None,
                'es_95': self.risk_metrics['expected_shortfall'].iloc[-1] if len(self.risk_metrics['expected_shortfall']) > 0 else None,
                'sharpe_ratio': self.risk_metrics['sharpe_ratio'].iloc[-1] if len(self.risk_metrics['sharpe_ratio']) > 0 else None
            },
            'position_analysis': self._analyze_positions(),
            'risk_limits': {
                'portfolio_risk_used': self._calculate_portfolio_risk_usage(),
                'sector_exposure': self._calculate_sector_exposures(),
                'correlation_status': self._analyze_correlations()
            }
        }
        
    def _analyze_positions(self) -> Dict:
        """Analizza posizioni correnti"""
        return {
            'position_sizes': {
                pos_id: pos['size'] / self.portfolio.total_equity
                for pos_id, pos in self.portfolio.open_positions.items()
            },
            'risk_contribution': self._calculate_risk_contribution(),
            'position_performance': self._calculate_position_performance()
        }
        
    def _calculate_risk_contribution(self) -> Dict:
        """Calcola contributo al rischio di ogni posizione"""
        if not self.portfolio.open_positions:
            return {}
            
        total_risk = sum(pos['risk'] for pos in self.portfolio.open_positions.values())
        
        return {
            pos_id: pos['risk'] / total_risk
            for pos_id, pos in self.portfolio.open_positions.items()
        }
