from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import logging
from ..base import BaseAnalyzer

@dataclass
class EconomicIndicator:
    name: str
    value: float
    previous: float
    trend: str
    impact: float      # Impact score from -1 to 1
    confidence: float  # Data reliability score
    forecast: float
    timestamp: datetime
    metadata: Dict

@dataclass
class EconomicState:
    timestamp: datetime
    cycle_phase: str                   # Current economic cycle phase
    indicators: Dict[str, EconomicIndicator]  # Economic indicators
    monetary_policy: Dict              # Monetary policy analysis
    growth_metrics: Dict               # Growth-related metrics
    inflation_metrics: Dict            # Inflation-related metrics
    employment_metrics: Dict           # Employment-related metrics
    risk_metrics: Dict                 # Economic risk metrics
    forecasts: Dict                    # Economic forecasts
    signals: Dict                      # Market signals
    confidence: float
    metadata: Dict

class EconomicAnalyzer(BaseAnalyzer):
    """Economic Analysis Component"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.indicator_weights = self.config.get('indicator_weights', {
            'gdp': 0.20,
            'inflation': 0.15,
            'employment': 0.15,
            'interest_rates': 0.15,
            'industrial_production': 0.10,
            'retail_sales': 0.10,
            'housing': 0.08,
            'trade_balance': 0.07
        })
        
        # Economic cycle parameters
        self.cycle_phases = ['expansion', 'peak', 'contraction', 'trough']
        self.min_phase_duration = self.config.get('min_phase_duration', 2)  # quarters
        
        # Analysis thresholds
        self.recession_threshold = self.config.get('recession_threshold', -0.5)
        self.inflation_threshold = self.config.get('inflation_threshold', 2.0)
        self.growth_threshold = self.config.get('growth_threshold', 2.0)
        
        # Analysis cache
        self.indicator_history = {}
        self.analysis_history = []

    def analyze(self, data: Dict[str, pd.DataFrame]) -> EconomicState:
        """
        Perform comprehensive economic analysis
        
        Parameters:
        - data: Dictionary containing economic indicators and data
        
        Returns:
        - EconomicState object containing analysis results
        """
        try:
            # Analyze economic indicators
            indicators = self._analyze_indicators(data)
            
            # Determine economic cycle phase
            cycle_phase = self._determine_cycle_phase(indicators)
            
            # Analyze monetary policy
            monetary_policy = self._analyze_monetary_policy(data, indicators)
            
            # Analyze growth metrics
            growth_metrics = self._analyze_growth(data, indicators)
            
            # Analyze inflation metrics
            inflation_metrics = self._analyze_inflation(data, indicators)
            
            # Analyze employment metrics
            employment_metrics = self._analyze_employment(data, indicators)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                indicators, monetary_policy, growth_metrics)
            
            # Generate forecasts
            forecasts = self._generate_forecasts(
                indicators, cycle_phase, monetary_policy)
            
            # Generate market signals
            signals = self._generate_signals(
                indicators, cycle_phase, monetary_policy, risk_metrics)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                indicators, cycle_phase, monetary_policy, forecasts)
            
            # Generate metadata
            metadata = self._generate_metadata(data, indicators, cycle_phase)
            
            state = EconomicState(
                timestamp=datetime.now(),
                cycle_phase=cycle_phase,
                indicators=indicators,
                monetary_policy=monetary_policy,
                growth_metrics=growth_metrics,
                inflation_metrics=inflation_metrics,
                employment_metrics=employment_metrics,
                risk_metrics=risk_metrics,
                forecasts=forecasts,
                signals=signals,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update history
            self._update_history(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Economic analysis failed: {e}")
            raise

    def _analyze_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, EconomicIndicator]:
        """Analyze economic indicators"""
        indicators = {}
        
        # GDP analysis
        if 'gdp' in data:
            indicators['gdp'] = self._analyze_gdp(data['gdp'])
            
        # Inflation analysis
        if 'inflation' in data:
            indicators['inflation'] = self._analyze_inflation_indicator(
                data['inflation'])
            
        # Employment analysis
        if 'employment' in data:
            indicators['employment'] = self._analyze_employment_indicator(
                data['employment'])
            
        # Interest rates analysis
        if 'interest_rates' in data:
            indicators['interest_rates'] = self._analyze_interest_rates(
                data['interest_rates'])
            
        # Industrial production analysis
        if 'industrial_production' in data:
            indicators['industrial_production'] = self._analyze_industrial_production(
                data['industrial_production'])
            
        # Additional indicators
        self._analyze_additional_indicators(data, indicators)
        
        return indicators

    def _determine_cycle_phase(self, 
                             indicators: Dict[str, EconomicIndicator]) -> str:
        """Determine current economic cycle phase"""
        # Calculate composite growth indicator
        growth = self._calculate_composite_growth(indicators)
        
        # Calculate momentum
        momentum = self._calculate_economic_momentum(indicators)
        
        # Analyze turning points
        turning_points = self._analyze_turning_points(indicators)
        
        # Determine phase
        if growth > self.growth_threshold and momentum > 0:
            phase = 'expansion'
        elif growth > self.growth_threshold and momentum < 0:
            phase = 'peak'
        elif growth < self.recession_threshold and momentum < 0:
            phase = 'contraction'
        elif growth < self.recession_threshold and momentum > 0:
            phase = 'trough'
        else:
            phase = 'transition'
            
        return phase

    def _analyze_monetary_policy(self, data: Dict[str, pd.DataFrame],
                               indicators: Dict[str, EconomicIndicator]) -> Dict:
        """Analyze monetary policy stance and implications"""
        # Analyze interest rates
        rates_analysis = self._analyze_interest_rate_policy(
            data.get('interest_rates', pd.DataFrame()))
        
        # Analyze money supply
        money_supply = self._analyze_money_supply(
            data.get('money_supply', pd.DataFrame()))
        
        # Analyze central bank actions
        cb_actions = self._analyze_central_bank_actions(
            data.get('cb_actions', pd.DataFrame()))
        
        # Analyze policy stance
        policy_stance = self._determine_policy_stance(
            rates_analysis, money_supply, cb_actions)
        
        return {
            'rates_analysis': rates_analysis,
            'money_supply': money_supply,
            'cb_actions': cb_actions,
            'policy_stance': policy_stance,
            'market_impact': self._analyze_policy_impact(policy_stance)
        }

    def _analyze_growth(self, data: Dict[str, pd.DataFrame],
                       indicators: Dict[str, EconomicIndicator]) -> Dict:
        """Analyze economic growth metrics"""
        # GDP components analysis
        gdp_components = self._analyze_gdp_components(
            data.get('gdp_components', pd.DataFrame()))
        
        # Leading indicators
        leading_indicators = self._analyze_leading_indicators(
            data, indicators)
        
        # Growth momentum
        momentum = self._calculate_growth_momentum(data, indicators)
        
        # Growth forecasts
        forecasts = self._forecast_growth(data, indicators)
        
        return {
            'gdp_components': gdp_components,
            'leading_indicators': leading_indicators,
            'momentum': momentum,
            'forecasts': forecasts,
            'risk_factors': self._analyze_growth_risks(data, indicators)
        }

    def _analyze_inflation(self, data: Dict[str, pd.DataFrame],
                         indicators: Dict[str, EconomicIndicator]) -> Dict:
        """Analyze inflation metrics"""
        # Core inflation
        core_inflation = self._analyze_core_inflation(
            data.get('inflation', pd.DataFrame()))
        
        # Price pressures
        price_pressures = self._analyze_price_pressures(data)
        
        # Inflation expectations
        expectations = self._analyze_inflation_expectations(data)
        
        # Wage growth
        wage_growth = self._analyze_wage_growth(
            data.get('wages', pd.DataFrame()))
        
        return {
            'core_inflation': core_inflation,
            'price_pressures': price_pressures,
            'expectations': expectations,
            'wage_growth': wage_growth,
            'risks': self._analyze_inflation_risks(data, indicators)
        }

    def _generate_forecasts(self, indicators: Dict[str, EconomicIndicator],
                          cycle_phase: str,
                          monetary_policy: Dict) -> Dict:
        """Generate economic forecasts"""
        # GDP forecast
        gdp_forecast = self._forecast_gdp(indicators, cycle_phase)
        
        # Inflation forecast
        inflation_forecast = self._forecast_inflation(
            indicators, monetary_policy)
        
        # Employment forecast
        employment_forecast = self._forecast_employment(indicators)
        
        # Interest rates forecast
        rates_forecast = self._forecast_interest_rates(
            indicators, monetary_policy)
        
        return {
            'gdp': gdp_forecast,
            'inflation': inflation_forecast,
            'employment': employment_forecast,
            'interest_rates': rates_forecast,
            'confidence': self._calculate_forecast_confidence(
                indicators, cycle_phase)
        }

    def _generate_signals(self, indicators: Dict[str, EconomicIndicator],
                         cycle_phase: str,
                         monetary_policy: Dict,
                         risk_metrics: Dict) -> Dict:
        """Generate market signals based on economic analysis"""
        # Asset allocation signals
        asset_signals = self._generate_asset_allocation_signals(
            cycle_phase, monetary_policy)
        
        # Sector rotation signals
        sector_signals = self._generate_sector_signals(
            cycle_phase, indicators)
        
        # Risk signals
        risk_signals = self._generate_risk_signals(risk_metrics)
        
        # Timing signals
        timing_signals = self._generate_timing_signals(
            indicators, cycle_phase)
        
        return {
            'asset_allocation': asset_signals,
            'sector_rotation': sector_signals,
            'risk_positioning': risk_signals,
            'timing': timing_signals,
            'conviction': self._calculate_signal_conviction(
                indicators, cycle_phase)
        }

    def _calculate_risk_metrics(self, indicators: Dict[str, EconomicIndicator],
                              monetary_policy: Dict,
                              growth_metrics: Dict) -> Dict:
        """Calculate economic risk metrics"""
        return {
            'recession_probability': self._calculate_recession_probability(
                indicators, growth_metrics),
            'inflation_risk': self._calculate_inflation_risk(
                indicators, monetary_policy),
            'policy_risk': self._calculate_policy_risk(monetary_policy),
            'systemic_risk': self._calculate_systemic_risk(indicators),
            'market_impact': self._assess_market_impact(
                indicators, monetary_policy)
        }

    def _calculate_confidence(self, indicators: Dict[str, EconomicIndicator],
                            cycle_phase: str,
                            monetary_policy: Dict,
                            forecasts: Dict) -> float:
        """Calculate confidence in economic analysis"""
        # Calculate component confidences
        indicator_conf = self._calculate_indicator_confidence(indicators)
        cycle_conf = self._calculate_cycle_confidence(cycle_phase)
        policy_conf = self._calculate_policy_confidence(monetary_policy)
        forecast_conf = self._calculate_forecast_confidence(
            indicators, cycle_phase)
        
        # Weight components
        weights = {
            'indicators': 0.3,
            'cycle': 0.2,
            'policy': 0.3,
            'forecasts': 0.2
        }
        
        confidence = sum(
            conf * weights[name]
            for name, conf in [
                ('indicators', indicator_conf),
                ('cycle', cycle_conf),
                ('policy', policy_conf),
                ('forecasts', forecast_conf)
            ]
        )
        
        return np.clip(confidence, 0, 1)

    @property
    def required_data(self) -> List[str]:
        """Required economic data series"""
        return [
            'gdp', 'inflation', 'employment', 'interest_rates',
            'industrial_production', 'retail_sales', 'housing',
            'trade_balance'
        ]

    def get_analysis_summary(self) -> Dict:
        """Get summary of current economic state"""
        if not self.analysis_history:
            return {}
            
        latest = self.analysis_history[-1]
        return {
            'timestamp': latest.timestamp,
            'cycle_phase': latest.cycle_phase,
            'gdp_growth': latest.indicators['gdp'].value,
            'inflation': latest.indicators['inflation'].value,
            'policy_stance': latest.monetary_policy['policy_stance'],
            'risks': latest.risk_metrics,
            'confidence': latest.confidence
        }
