from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class PositionConfig:
    base_risk_per_trade: float
    position_sizing_method: str
    max_position_size: float
    min_position_size: float
    size_scaling: bool
    use_atr_sizing: bool
    use_volatility_scaling: bool
    use_kelly_criterion: bool
    portfolio_heat: float
    risk_scaling_factors: Dict[str, float]

class PositionSizingEngine:
    def __init__(self, config: PositionConfig):
        """Inizializza Position Sizing Engine"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.size_history = []
        self.risk_adjustments = []
        self.scaling_history = []
        
        # Sizing methods
        self.sizing_methods = {
            'fixed': self._fixed_sizing,
            'volatility': self._volatility_sizing,
            'atr': self._atr_sizing,
            'kelly': self._kelly_sizing,
            'risk_parity': self._risk_parity_sizing,
            'adaptive': self._adaptive_sizing
        }
        
    def calculate_position_size(self,
                              signal: Dict,
                              portfolio_state: Dict,
                              market_data: pd.DataFrame) -> float:
        """Calcola la dimensione ottimale della posizione"""
        try:
            # Base position size
            base_size = self._calculate_base_size(signal, portfolio_state)
            
            # Apply sizing method
            sizing_method = self.sizing_methods[self.config.position_sizing_method]
            position_size = sizing_method(base_size, signal, portfolio_state, market_data)
            
            # Apply risk adjustments
            position_size = self._apply_risk_adjustments(
                position_size, signal, portfolio_state)
                
            # Apply scaling factors
            if self.config.size_scaling:
                position_size = self._apply_scaling_factors(
                    position_size, signal, portfolio_state)
                    
            # Validate final size
            position_size = self._validate_position_size(
                position_size, portfolio_state)
                
            # Track sizing decision
            self._track_sizing_decision(
                position_size, signal, portfolio_state)
                
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    def _calculate_base_size(self,
                           signal: Dict,
                           portfolio_state: Dict) -> float:
        """Calcola la dimensione base della posizione"""
        portfolio_value = portfolio_state['equity']
        risk_amount = portfolio_value * self.config.base_risk_per_trade
        
        if signal.get('stop_loss'):
            risk_per_unit = abs(signal['price'] - signal['stop_loss'])
            if risk_per_unit > 0:
                return risk_amount / risk_per_unit
                
        return risk_amount / signal['price']
        
    def _volatility_sizing(self,
                          base_size: float,
                          signal: Dict,
                          portfolio_state: Dict,
                          market_data: pd.DataFrame) -> float:
        """Sizing basato sulla volatilità"""
        if not self.config.use_volatility_scaling:
            return base_size
            
        # Calculate volatility scalar
        current_vol = market_data['volatility'].iloc[-1]
        avg_vol = market_data['volatility'].rolling(20).mean().iloc[-1]
        
        if avg_vol > 0:
            vol_scalar = avg_vol / current_vol
        else:
            vol_scalar = 1.0
            
        # Apply boundaries
        vol_scalar = np.clip(vol_scalar, 0.5, 2.0)
        
        return base_size * vol_scalar
        
    def _atr_sizing(self,
                    base_size: float,
                    signal: Dict,
                    portfolio_state: Dict,
                    market_data: pd.DataFrame) -> float:
        """Sizing basato su Average True Range"""
        if not self.config.use_atr_sizing:
            return base_size
            
        portfolio_value = portfolio_state['equity']
        atr = market_data['atr'].iloc[-1]
        
        if atr > 0:
            risk_amount = portfolio_value * self.config.base_risk_per_trade
            atr_size = risk_amount / (atr * self.config.risk_scaling_factors.get('atr', 2.0))
            return min(base_size, atr_size)
            
        return base_size
        
    def _kelly_sizing(self,
                     base_size: float,
                     signal: Dict,
                     portfolio_state: Dict,
                     market_data: pd.DataFrame) -> float:
        """Sizing basato sul Kelly Criterion"""
        if not self.config.use_kelly_criterion:
            return base_size
            
        # Calculate Kelly fraction
        win_rate = signal.get('win_probability', 0.5)
        win_loss_ratio = signal.get('win_loss_ratio', 1.0)
        
        kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply half-Kelly for conservativeness
        kelly_fraction *= 0.5
        
        # Apply boundaries
        kelly_fraction = np.clip(kelly_fraction, 0, self.config.max_position_size)
        
        return base_size * kelly_fraction
        
    def _risk_parity_sizing(self,
                           base_size: float,
                           signal: Dict,
                           portfolio_state: Dict,
                           market_data: pd.DataFrame) -> float:
        """Sizing basato su Risk Parity"""
        portfolio_value = portfolio_state['equity']
        
        # Calculate asset volatility
        asset_vol = market_data['returns'].std() * np.sqrt(252)
        
        # Calculate portfolio volatility target
        target_vol = self.config.risk_scaling_factors.get('portfolio_vol', 0.15)
        
        if asset_vol > 0:
            # Calculate size to achieve equal risk contribution
            n_assets = len(portfolio_state['positions']) + 1
            target_risk = target_vol / np.sqrt(n_assets)
            size = (target_risk / asset_vol) * portfolio_value / signal['price']
            return min(base_size, size)
            
        return base_size
        
    def _adaptive_sizing(self,
                        base_size: float,
                        signal: Dict,
                        portfolio_state: Dict,
                        market_data: pd.DataFrame) -> float:
        """Sizing adattivo basato su multiple metriche"""
        # Start with base size
        size = base_size
        
        # Apply volatility scaling
        size = self._volatility_sizing(size, signal, portfolio_state, market_data)
        
        # Apply ATR scaling if enabled
        if self.config.use_atr_sizing:
            size = self._atr_sizing(size, signal, portfolio_state, market_data)
            
        # Apply Kelly criterion if enabled
        if self.config.use_kelly_criterion:
            size = self._kelly_sizing(size, signal, portfolio_state, market_data)
            
        # Apply performance scaling
        size *= self._calculate_performance_scalar(portfolio_state)
        
        return size
        
    def _apply_risk_adjustments(self,
                              size: float,
                              signal: Dict,
                              portfolio_state: Dict) -> float:
        """Applica aggiustamenti basati sul rischio"""
        # Portfolio heat adjustment
        heat = self._calculate_portfolio_heat(portfolio_state)
        if heat > self.config.portfolio_heat:
            size *= (self.config.portfolio_heat / heat)
            
        # Correlation adjustment
        corr_scalar = self._calculate_correlation_scalar(signal, portfolio_state)
        size *= corr_scalar
        
        # Market regime adjustment
        regime_scalar = self._calculate_regime_scalar(portfolio_state)
        size *= regime_scalar
        
        return size
        
    def _apply_scaling_factors(self,
                             size: float,
                             signal: Dict,
                             portfolio_state: Dict) -> float:
        """Applica fattori di scaling basati su varie metriche"""
        # Performance based scaling
        performance_scalar = self._calculate_performance_scalar(portfolio_state)
        size *= performance_scalar
        
        # Volatility based scaling
        volatility_scalar = signal.get('volatility_scalar', 1.0)
        size *= volatility_scalar
        
        # Confidence based scaling
        confidence_scalar = signal.get('confidence', 1.0)
        size *= confidence_scalar
        
        return size
        
    def _validate_position_size(self,
                              size: float,
                              portfolio_state: Dict) -> float:
        """Valida e aggiusta la dimensione finale della posizione"""
        portfolio_value = portfolio_state['equity']
        
        # Apply absolute limits
        max_size = portfolio_value * self.config.max_position_size
        min_size = portfolio_value * self.config.min_position_size
        
        # Check current exposure
        current_exposure = sum(pos['value'] for pos in portfolio_state['positions'].values())
        available_exposure = portfolio_value - current_exposure
        
        # Apply limits
        size = min(size, max_size, available_exposure)
        size = max(size, min_size)
        
        return size
        
    def _calculate_portfolio_heat(self, portfolio_state: Dict) -> float:
        """Calcola il portfolio heat"""
        portfolio_value = portfolio_state['equity']
        total_exposure = sum(pos['value'] for pos in portfolio_state['positions'].values())
        return total_exposure / portfolio_value
        
    def _calculate_correlation_scalar(self,
                                    signal: Dict,
                                    portfolio_state: Dict) -> float:
        """Calcola scalar basato sulle correlazioni"""
        if not portfolio_state['positions']:
            return 1.0
            
        # Get correlations with existing positions
        correlations = signal.get('correlations', {})
        if not correlations:
            return 1.0
            
        # Calculate average absolute correlation
        avg_correlation = np.mean([abs(c) for c in correlations.values()])
        
        # Reduce size for high correlations
        return 1.0 - (avg_correlation * 0.5)
        
    def _calculate_regime_scalar(self, portfolio_state: Dict) -> float:
        """Calcola scalar basato sul regime di mercato"""
        regime = portfolio_state.get('market_regime', 'normal')
        
        regime_scalars = {
            'crisis': 0.5,
            'high_vol': 0.7,
            'normal': 1.0,
            'low_vol': 1.2,
            'trending': 1.3
        }
        
        return regime_scalars.get(regime, 1.0)
        
    def _calculate_performance_scalar(self, portfolio_state: Dict) -> float:
        """Calcola scalar basato sulla performance"""
        # Get recent performance metrics
        win_rate = portfolio_state.get('win_rate', 0.5)
        profit_factor = portfolio_state.get('profit_factor', 1.0)
        
        # Calculate performance scalar
        performance_scalar = (win_rate * 0.5 + min(profit_factor / 2, 1.0) * 0.5)
        
        # Apply boundaries
        return np.clip(performance_scalar, 0.5, 1.5)
        
    def _track_sizing_decision(self,
                             size: float,
                             signal: Dict,
                             portfolio_state: Dict):
        """Traccia le decisioni di sizing"""
        decision = {
            'timestamp': datetime.now(),
            'symbol': signal['symbol'],
            'base_size': self._calculate_base_size(signal, portfolio_state),
            'final_size': size,
            'portfolio_heat': self._calculate_portfolio_heat(portfolio_state),
            'sizing_method': self.config.position_sizing_method,
            'adjustments': self.risk_adjustments,
            'scaling_factors': self.scaling_history
        }
        
        self.size_history.append(decision)

    def get_sizing_analytics(self) -> pd.DataFrame:
        """Ritorna analytics sulle decisioni di sizing"""
        return pd.DataFrame(self.size_history)
