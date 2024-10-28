from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class RiskParameters:
    max_position_size: float
    max_portfolio_risk: float
    max_correlation: float
    max_sector_exposure: float
    position_sizing_method: str
    use_var: bool
    var_confidence: float
    max_leverage: float
    risk_free_rate: float
    stress_test_scenarios: List[str]

class RiskEngine:
    def __init__(self, risk_params: RiskParameters):
        """Inizializza Risk Management Engine"""
        self.params = risk_params
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.positions = {}
        self.portfolio_exposure = 0.0
        self.sector_exposure = {}
        self.correlation_matrix = pd.DataFrame()
        
        # Risk metrics
        self.var_history = []
        self.stress_test_results = {}
        self.risk_metrics = {}
        
    def validate_trade(self, 
                      trade: Dict,
                      portfolio_state: Dict,
                      market_data: pd.DataFrame) -> bool:
        """Valida un potenziale trade contro tutti i criteri di rischio"""
        try:
            # Check position size
            if not self._check_position_size(trade, portfolio_state):
                return False
                
            # Check portfolio risk
            if not self._check_portfolio_risk(trade, portfolio_state):
                return False
                
            # Check correlation
            if not self._check_correlation_limits(trade, portfolio_state):
                return False
                
            # Check sector exposure
            if not self._check_sector_exposure(trade, portfolio_state):
                return False
                
            # Check VaR limits
            if self.params.use_var:
                if not self._check_var_limits(trade, portfolio_state, market_data):
                    return False
                    
            # Check leverage
            if not self._check_leverage_limits(trade, portfolio_state):
                return False
                
            # Run stress tests
            if not self._run_stress_tests(trade, portfolio_state, market_data):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {str(e)}")
            return False
            
    def calculate_position_size(self,
                              signal: Dict,
                              portfolio_value: float,
                              market_data: pd.DataFrame) -> float:
        """Calcola la dimensione ottimale della posizione"""
        if self.params.position_sizing_method == 'fixed_fraction':
            return self._fixed_fraction_sizing(signal, portfolio_value)
        elif self.params.position_sizing_method == 'kelly':
            return self._kelly_criterion_sizing(signal, portfolio_value, market_data)
        elif self.params.position_sizing_method == 'risk_parity':
            return self._risk_parity_sizing(signal, portfolio_value, market_data)
        elif self.params.position_sizing_method == 'vol_targeting':
            return self._volatility_targeting_sizing(signal, portfolio_value, market_data)
        else:
            return self._default_position_sizing(signal, portfolio_value)
            
    def update_risk_metrics(self,
                          portfolio_state: Dict,
                          market_data: pd.DataFrame) -> Dict:
        """Aggiorna tutte le metriche di rischio"""
        metrics = {}
        
        # Portfolio risk metrics
        metrics['portfolio'] = self._calculate_portfolio_risk_metrics(
            portfolio_state, market_data)
            
        # VaR metrics
        if self.params.use_var:
            metrics['var'] = self._calculate_var_metrics(
                portfolio_state, market_data)
                
        # Stress test metrics
        metrics['stress_test'] = self._run_stress_test_scenarios(
            portfolio_state, market_data)
            
        # Correlation metrics
        metrics['correlation'] = self._calculate_correlation_metrics(
            portfolio_state, market_data)
            
        # Exposure metrics
        metrics['exposure'] = self._calculate_exposure_metrics(portfolio_state)
        
        self.risk_metrics = metrics
        return metrics
        
    def _check_position_size(self, trade: Dict, portfolio_state: Dict) -> bool:
        """Verifica i limiti di dimensione della posizione"""
        position_value = trade['quantity'] * trade['price']
        portfolio_value = portfolio_state['equity']
        
        # Check absolute size
        if position_value > portfolio_value * self.params.max_position_size:
            return False
            
        # Check relative to portfolio
        current_exposure = sum(pos['value'] for pos in portfolio_state['positions'].values())
        new_exposure = current_exposure + position_value
        
        if new_exposure > portfolio_value:
            return False
            
        return True
        
    def _check_portfolio_risk(self, trade: Dict, portfolio_state: Dict) -> bool:
        """Verifica i limiti di rischio del portfolio"""
        # Calculate current portfolio risk
        current_risk = self._calculate_portfolio_risk(portfolio_state)
        
        # Calculate marginal risk contribution
        marginal_risk = self._calculate_marginal_risk(trade, portfolio_state)
        
        # Check total risk
        total_risk = current_risk + marginal_risk
        
        return total_risk <= self.params.max_portfolio_risk
        
    def _check_correlation_limits(self, trade: Dict, portfolio_state: Dict) -> bool:
        """Verifica i limiti di correlazione"""
        if not portfolio_state['positions']:
            return True
            
        # Calculate correlations with existing positions
        correlations = []
        for pos in portfolio_state['positions'].values():
            corr = self._calculate_correlation(trade['symbol'], pos['symbol'])
            correlations.append(abs(corr))
            
        # Check max correlation
        max_correlation = max(correlations)
        return max_correlation <= self.params.max_correlation
        
    def _check_sector_exposure(self, trade: Dict, portfolio_state: Dict) -> bool:
        """Verifica i limiti di esposizione settoriale"""
        sector = trade.get('sector', 'unknown')
        
        # Calculate current sector exposure
        current_exposure = self.sector_exposure.get(sector, 0.0)
        
        # Calculate new exposure
        new_exposure = current_exposure + (trade['quantity'] * trade['price'])
        portfolio_value = portfolio_state['equity']
        
        return (new_exposure / portfolio_value) <= self.params.max_sector_exposure
        
    def _check_var_limits(self, 
                         trade: Dict,
                         portfolio_state: Dict,
                         market_data: pd.DataFrame) -> bool:
        """Verifica i limiti di Value at Risk"""
        # Calculate current VaR
        current_var = self._calculate_var(portfolio_state, market_data)
        
        # Calculate marginal VaR
        marginal_var = self._calculate_marginal_var(trade, portfolio_state, market_data)
        
        # Check total VaR
        total_var = current_var + marginal_var
        portfolio_value = portfolio_state['equity']
        
        return (total_var / portfolio_value) <= self.params.max_portfolio_risk
        
    def _run_stress_tests(self,
                         trade: Dict,
                         portfolio_state: Dict,
                         market_data: pd.DataFrame) -> bool:
        """Esegue stress test sul portfolio"""
        for scenario in self.params.stress_test_scenarios:
            # Apply scenario
            stressed_portfolio = self._apply_stress_scenario(
                scenario, trade, portfolio_state, market_data)
                
            # Check stress results
            if not self._validate_stress_test(stressed_portfolio):
                return False
                
        return True
        
    def _fixed_fraction_sizing(self, signal: Dict, portfolio_value: float) -> float:
        """Calcola position size usando fixed fraction"""
        risk_amount = portfolio_value * self.params.max_position_size
        price = signal['price']
        stop_loss = signal.get('stop_loss')
        
        if stop_loss:
            risk_per_share = abs(price - stop_loss)
            if risk_per_share > 0:
                return risk_amount / risk_per_share
                
        return (risk_amount / price)
        
    def _kelly_criterion_sizing(self,
                              signal: Dict,
                              portfolio_value: float,
                              market_data: pd.DataFrame) -> float:
        """Calcola position size usando Kelly Criterion"""
        win_rate = signal.get('win_rate', 0.5)
        win_loss_ratio = signal.get('win_loss_ratio', 1.0)
        
        kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
        kelly_fraction = max(0, min(kelly_fraction, self.params.max_position_size))
        
        return (portfolio_value * kelly_fraction) / signal['price']
        
    def _risk_parity_sizing(self,
                           signal: Dict,
                           portfolio_value: float,
                           market_data: pd.DataFrame) -> float:
        """Calcola position size usando Risk Parity"""
        # Calculate asset volatility
        returns = market_data['returns'].dropna()
        asset_vol = returns.std() * np.sqrt(252)
        
        # Calculate target risk contribution
        n_assets = len(self.positions) + 1
        target_risk = self.params.max_portfolio_risk / n_assets
        
        # Calculate position size
        if asset_vol > 0:
            position_value = target_risk * portfolio_value / asset_vol
            return position_value / signal['price']
        
        return 0.0
        
    def _volatility_targeting_sizing(self,
                                   signal: Dict,
                                   portfolio_value: float,
                                   market_data: pd.DataFrame) -> float:
        """Calcola position size usando Volatility Targeting"""
        # Calculate asset volatility
        returns = market_data['returns'].dropna()
        asset_vol = returns.std() * np.sqrt(252)
        
        # Target volatility
        target_vol = self.params.max_portfolio_risk
        
        # Calculate position size
        if asset_vol > 0:
            vol_scalar = target_vol / asset_vol
            position_value = portfolio_value * vol_scalar * self.params.max_position_size
            return position_value / signal['price']
            
        return 0.0
        
    def _calculate_portfolio_risk_metrics(self,
                                        portfolio_state: Dict,
                                        market_data: pd.DataFrame) -> Dict:
        """Calcola metriche di rischio del portfolio"""
        metrics = {}
        
        # Portfolio volatility
        metrics['volatility'] = self._calculate_portfolio_volatility(
            portfolio_state, market_data)
            
        # Beta
        metrics['beta'] = self._calculate_portfolio_beta(
            portfolio_state, market_data)
            
        # Tracking error
        metrics['tracking_error'] = self._calculate_tracking_error(
            portfolio_state, market_data)
            
        # Information ratio
        metrics['information_ratio'] = self._calculate_information_ratio(
            portfolio_state, market_data)
            
        return metrics
        
    def _calculate_var_metrics(self,
                             portfolio_state: Dict,
                             market_data: pd.DataFrame) -> Dict:
        """Calcola metriche VaR"""
        metrics = {}
        
        # Parametric VaR
        metrics['parametric_var'] = self._calculate_parametric_var(
            portfolio_state, market_data)
            
        # Historical VaR
        metrics['historical_var'] = self._calculate_historical_var(
            portfolio_state, market_data)
            
        # Conditional VaR (Expected Shortfall)
        metrics['conditional_var'] = self._calculate_conditional_var(
            portfolio_state, market_data)
            
        return metrics
        
    def _calculate_correlation_metrics(self,
                                     portfolio_state: Dict,
                                     market_data: pd.DataFrame) -> Dict:
        """Calcola metriche di correlazione"""
        metrics = {}
        
        # Asset correlations
        metrics['asset_correlations'] = self._calculate_asset_correlations(market_data)
        
        # Portfolio correlation
        metrics['portfolio_correlation'] = self._calculate_portfolio_correlation(
            portfolio_state, market_data)
            
        return metrics
