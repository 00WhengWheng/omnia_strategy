from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import logging
from ..base import BaseAnalyzer

@dataclass
class RiskMetrics:
    var: Dict[str, float]           # Value at Risk metrics
    expected_shortfall: float       # Expected Shortfall/CVaR
    beta: float                     # Market beta
    volatility: Dict[str, float]    # Volatility metrics
    correlation: Dict[str, float]   # Correlation metrics
    tail_risk: Dict[str, float]     # Tail risk measures

@dataclass
class MarketRiskState:
    timestamp: datetime
    metrics: RiskMetrics
    exposure: Dict[str, float]      # Market exposures
    stress_tests: Dict[str, float]  # Stress test results
    scenario_analysis: Dict         # Scenario analysis results
    risk_decomposition: Dict        # Risk factor decomposition
    risk_signals: Dict             # Risk signals
    risk_limits: Dict              # Risk limit monitoring
    hedging_recommendations: Dict   # Hedging suggestions
    confidence: float
    metadata: Dict

class MarketRiskAnalyzer(BaseAnalyzer):
    """Market Risk Analysis Component"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.confidence_levels = self.config.get('confidence_levels', [0.95, 0.99])
        self.var_window = self.config.get('var_window', 252)
        self.stress_scenarios = self.config.get('stress_scenarios', [
            'crisis', 'recession', 'inflation', 'recovery'
        ])
        
        # Risk limits
        self.risk_limits = {
            'var_limit': self.config.get('var_limit', 0.02),
            'volatility_limit': self.config.get('volatility_limit', 0.25),
            'beta_limit': self.config.get('beta_limit', 1.5),
            'correlation_limit': self.config.get('correlation_limit', 0.7)
        }
        
        # Analysis parameters
        self.lookback_period = self.config.get('lookback_period', 252)
        self.decay_factor = self.config.get('decay_factor', 0.94)
        
        # Initialize risk history
        self.risk_history = []
        self.exposure_history = pd.DataFrame()

    def analyze(self, data: Dict) -> MarketRiskState:
        """
        Perform comprehensive market risk analysis
        
        Parameters:
        - data: Dictionary containing:
            - returns: Asset returns
            - market_data: Market data
            - factor_data: Risk factor data
            - correlation_data: Correlation data
        
        Returns:
        - MarketRiskState object containing analysis results
        """
        try:
            # Calculate risk metrics
            metrics = self._calculate_risk_metrics(data)
            
            # Calculate market exposures
            exposure = self._calculate_exposures(data)
            
            # Perform stress tests
            stress_tests = self._perform_stress_tests(data, metrics)
            
            # Perform scenario analysis
            scenario_analysis = self._perform_scenario_analysis(data, metrics)
            
            # Decompose risk factors
            risk_decomposition = self._decompose_risk_factors(data, metrics)
            
            # Generate risk signals
            risk_signals = self._generate_risk_signals(
                metrics, exposure, stress_tests)
            
            # Monitor risk limits
            risk_limits = self._monitor_risk_limits(metrics, exposure)
            
            # Generate hedging recommendations
            hedging_recommendations = self._generate_hedging_recommendations(
                metrics, exposure, risk_signals)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                data, metrics, exposure)
            
            # Generate metadata
            metadata = self._generate_metadata(data)
            
            state = MarketRiskState(
                timestamp=datetime.now(),
                metrics=metrics,
                exposure=exposure,
                stress_tests=stress_tests,
                scenario_analysis=scenario_analysis,
                risk_decomposition=risk_decomposition,
                risk_signals=risk_signals,
                risk_limits=risk_limits,
                hedging_recommendations=hedging_recommendations,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update history
            self._update_history(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Market risk analysis failed: {e}")
            raise

    def _calculate_risk_metrics(self, data: Dict) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        returns = data['returns']
        market_data = data['market_data']
        
        # Calculate VaR
        var = self._calculate_var(returns)
        
        # Calculate Expected Shortfall
        es = self._calculate_expected_shortfall(returns)
        
        # Calculate beta
        beta = self._calculate_beta(returns, market_data)
        
        # Calculate volatility metrics
        volatility = self._calculate_volatility_metrics(returns)
        
        # Calculate correlation metrics
        correlation = self._calculate_correlation_metrics(
            returns, market_data)
        
        # Calculate tail risk metrics
        tail_risk = self._calculate_tail_risk(returns)
        
        return RiskMetrics(
            var=var,
            expected_shortfall=es,
            beta=beta,
            volatility=volatility,
            correlation=correlation,
            tail_risk=tail_risk
        )

    def _calculate_var(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk using multiple methods"""
        var_metrics = {}
        
        # Historical VaR
        for confidence in self.confidence_levels:
            var_metrics[f'historical_{confidence}'] = self._calculate_historical_var(
                returns, confidence)
        
        # Parametric VaR
        for confidence in self.confidence_levels:
            var_metrics[f'parametric_{confidence}'] = self._calculate_parametric_var(
                returns, confidence)
        
        # Monte Carlo VaR
        for confidence in self.confidence_levels:
            var_metrics[f'monte_carlo_{confidence}'] = self._calculate_monte_carlo_var(
                returns, confidence)
        
        return var_metrics

    def _calculate_expected_shortfall(self, returns: pd.Series) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Calculate cutoff index
        confidence = self.confidence_levels[0]  # Use primary confidence level
        cutoff_index = int(len(returns) * (1 - confidence))
        
        # Calculate ES
        es = sorted_returns[:cutoff_index].mean()
        
        return abs(es)

    def _calculate_exposures(self, data: Dict) -> Dict[str, float]:
        """Calculate market exposures"""
        # Factor exposures
        factor_exposure = self._calculate_factor_exposures(
            data['returns'], data['factor_data'])
        
        # Market exposure
        market_exposure = self._calculate_market_exposure(
            data['returns'], data['market_data'])
        
        # Sector exposures
        sector_exposure = self._calculate_sector_exposures(
            data['returns'], data['sector_data'])
        
        # Style exposures
        style_exposure = self._calculate_style_exposures(
            data['returns'], data['factor_data'])
        
        return {
            'factor': factor_exposure,
            'market': market_exposure,
            'sector': sector_exposure,
            'style': style_exposure,
            'total': self._calculate_total_exposure(
                factor_exposure, market_exposure, sector_exposure)
        }

    def _perform_stress_tests(self, data: Dict,
                            metrics: RiskMetrics) -> Dict[str, float]:
        """Perform stress tests"""
        stress_results = {}
        
        for scenario in self.stress_scenarios:
            # Historical stress test
            stress_results[f'historical_{scenario}'] = self._historical_stress_test(
                data, scenario)
            
            # Hypothetical stress test
            stress_results[f'hypothetical_{scenario}'] = self._hypothetical_stress_test(
                data, scenario)
            
            # Factor stress test
            stress_results[f'factor_{scenario}'] = self._factor_stress_test(
                data, scenario)
            
        return stress_results

    def _perform_scenario_analysis(self, data: Dict,
                                 metrics: RiskMetrics) -> Dict:
        """Perform scenario analysis"""
        # Market scenarios
        market_scenarios = self._analyze_market_scenarios(data)
        
        # Volatility scenarios
        volatility_scenarios = self._analyze_volatility_scenarios(data)
        
        # Correlation scenarios
        correlation_scenarios = self._analyze_correlation_scenarios(data)
        
        # Liquidity scenarios
        liquidity_scenarios = self._analyze_liquidity_scenarios(data)
        
        return {
            'market': market_scenarios,
            'volatility': volatility_scenarios,
            'correlation': correlation_scenarios,
            'liquidity': liquidity_scenarios,
            'combined': self._combine_scenarios(
                market_scenarios,
                volatility_scenarios,
                correlation_scenarios,
                liquidity_scenarios
            )
        }

    def _decompose_risk_factors(self, data: Dict,
                              metrics: RiskMetrics) -> Dict:
        """Decompose risk into factors"""
        # Factor contribution
        factor_contribution = self._calculate_factor_contribution(data)
        
        # Risk attribution
        risk_attribution = self._calculate_risk_attribution(data)
        
        # Component VaR
        component_var = self._calculate_component_var(data)
        
        # Marginal VaR
        marginal_var = self._calculate_marginal_var(data)
        
        return {
            'factor_contribution': factor_contribution,
            'risk_attribution': risk_attribution,
            'component_var': component_var,
            'marginal_var': marginal_var,
            'total_risk': self._calculate_total_risk(
                factor_contribution, risk_attribution)
        }

    def _generate_risk_signals(self, metrics: RiskMetrics,
                             exposure: Dict[str, float],
                             stress_tests: Dict[str, float]) -> Dict:
        """Generate risk signals"""
        # VaR signals
        var_signals = self._generate_var_signals(metrics.var)
        
        # Exposure signals
        exposure_signals = self._generate_exposure_signals(exposure)
        
        # Stress test signals
        stress_signals = self._generate_stress_signals(stress_tests)
        
        # Correlation signals
        correlation_signals = self._generate_correlation_signals(
            metrics.correlation)
        
        return {
            'var': var_signals,
            'exposure': exposure_signals,
            'stress': stress_signals,
            'correlation': correlation_signals,
            'aggregate': self._aggregate_risk_signals(
                var_signals,
                exposure_signals,
                stress_signals,
                correlation_signals
            )
        }

    def _monitor_risk_limits(self, metrics: RiskMetrics,
                           exposure: Dict[str, float]) -> Dict:
        """Monitor risk limits"""
        # Check VaR limits
        var_status = self._check_var_limits(metrics.var)
        
        # Check exposure limits
        exposure_status = self._check_exposure_limits(exposure)
        
        # Check volatility limits
        volatility_status = self._check_volatility_limits(metrics.volatility)
        
        # Check correlation limits
        correlation_status = self._check_correlation_limits(metrics.correlation)
        
        return {
            'var_status': var_status,
            'exposure_status': exposure_status,
            'volatility_status': volatility_status,
            'correlation_status': correlation_status,
            'breaches': self._identify_limit_breaches(
                var_status,
                exposure_status,
                volatility_status,
                correlation_status
            )
        }

    def _generate_hedging_recommendations(self, metrics: RiskMetrics,
                                        exposure: Dict[str, float],
                                        risk_signals: Dict) -> Dict:
        """Generate hedging recommendations"""
        # Portfolio hedging
        portfolio_hedging = self._recommend_portfolio_hedges(
            metrics, exposure)
        
        # Factor hedging
        factor_hedging = self._recommend_factor_hedges(exposure)
        
        # Tail risk hedging
        tail_hedging = self._recommend_tail_hedges(metrics)
        
        # Dynamic hedging
        dynamic_hedging = self._recommend_dynamic_hedges(
            metrics, risk_signals)
        
        return {
            'portfolio': portfolio_hedging,
            'factor': factor_hedging,
            'tail': tail_hedging,
            'dynamic': dynamic_hedging,
            'cost_analysis': self._analyze_hedging_costs(
                portfolio_hedging,
                factor_hedging,
                tail_hedging,
                dynamic_hedging
            )
        }

    def _calculate_confidence(self, data: Dict,
                            metrics: RiskMetrics,
                            exposure: Dict[str, float]) -> float:
        """Calculate confidence in risk analysis"""
        # Data confidence
        data_conf = self._calculate_data_confidence(data)
        
        # Model confidence
        model_conf = self._calculate_model_confidence(metrics)
        
        # Exposure confidence
        exposure_conf = self._calculate_exposure_confidence(exposure)
        
        # Signal confidence
        signal_conf = self._calculate_signal_confidence(metrics)
        
        # Weight components
        confidence = (
            data_conf * 0.3 +
            model_conf * 0.3 +
            exposure_conf * 0.2 +
            signal_conf * 0.2
        )
        
        return np.clip(confidence, 0, 1)

    @property
    def required_data(self) -> List[str]:
        """Required data for market risk analysis"""
        return [
            'returns',
            'market_data',
            'factor_data',
            'correlation_data',
            'liquidity_data'
        ]

    def get_risk_summary(self) -> Dict:
        """Get summary of current market risk state"""
        if not self.risk_history:
            return {}
            
        latest = self.risk_history[-1]
        return {
            'timestamp': latest.timestamp,
            'var': latest.metrics.var,
            'expected_shortfall': latest.metrics.expected_shortfall,
            'current_exposure': latest.exposure['total'],
            'risk_signals': latest.risk_signals['aggregate'],
            'limit_breaches': latest.risk_limits['breaches'],
            'confidence': latest.confidence
        }
