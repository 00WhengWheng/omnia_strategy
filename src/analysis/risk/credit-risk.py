from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import logging
from ..base import BaseAnalyzer

@dataclass
class CreditMetrics:
    default_probability: float
    recovery_rate: float
    credit_exposure: float
    credit_rating: str
    credit_spread: float
    transition_probability: Dict[str, float]  # Rating transition probabilities

@dataclass
class CounterpartyProfile:
    id: str
    credit_metrics: CreditMetrics
    financial_metrics: Dict[str, float]
    industry_metrics: Dict[str, float]
    historical_performance: Dict[str, List]
    risk_indicators: Dict[str, float]

@dataclass
class CreditRiskState:
    timestamp: datetime
    portfolio_metrics: Dict[str, float]
    counterparty_profiles: Dict[str, CounterpartyProfile]
    expected_loss: float
    unexpected_loss: float
    concentration_risk: Dict[str, float]
    correlation_matrix: pd.DataFrame
    stress_tests: Dict[str, float]
    credit_limits: Dict[str, float]
    risk_signals: Dict[str, float]
    confidence: float
    metadata: Dict

class CreditRiskAnalyzer(BaseAnalyzer):
    """Credit Risk Analysis Component"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.rating_scale = self.config.get('rating_scale', [
            'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D'
        ])
        
        self.default_thresholds = self.config.get('default_thresholds', {
            'zscore': -3.0,
            'debt_service': 1.5,
            'leverage': 0.8
        })
        
        # Risk limits
        self.risk_limits = {
            'single_counterparty': self.config.get('single_counterparty_limit', 0.1),
            'industry_concentration': self.config.get('industry_limit', 0.25),
            'rating_concentration': self.config.get('rating_limit', 0.3),
            'average_rating': self.config.get('min_avg_rating', 'BBB')
        }
        
        # Model parameters
        self.confidence_level = self.config.get('confidence_level', 0.99)
        self.lookback_period = self.config.get('lookback_period', 252)
        
        # Initialize history
        self.credit_history = []
        self.transition_matrix = self._initialize_transition_matrix()

    def analyze(self, data: Dict) -> CreditRiskState:
        """
        Perform comprehensive credit risk analysis
        
        Parameters:
        - data: Dictionary containing:
            - counterparty_data: Individual counterparty information
            - market_data: Market-based credit indicators
            - financial_data: Financial statements
            - industry_data: Industry risk factors
        
        Returns:
        - CreditRiskState object containing analysis results
        """
        try:
            # Analyze counterparties
            counterparty_profiles = self._analyze_counterparties(data)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                counterparty_profiles)
            
            # Calculate expected loss
            expected_loss = self._calculate_expected_loss(
                counterparty_profiles)
            
            # Calculate unexpected loss
            unexpected_loss = self._calculate_unexpected_loss(
                counterparty_profiles)
            
            # Analyze concentration risk
            concentration_risk = self._analyze_concentration_risk(
                counterparty_profiles)
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(
                counterparty_profiles)
            
            # Perform stress tests
            stress_tests = self._perform_stress_tests(
                counterparty_profiles, portfolio_metrics)
            
            # Monitor credit limits
            credit_limits = self._monitor_credit_limits(
                counterparty_profiles, concentration_risk)
            
            # Generate risk signals
            risk_signals = self._generate_risk_signals(
                portfolio_metrics, concentration_risk, stress_tests)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                data, counterparty_profiles, portfolio_metrics)
            
            # Generate metadata
            metadata = self._generate_metadata(data)
            
            state = CreditRiskState(
                timestamp=datetime.now(),
                portfolio_metrics=portfolio_metrics,
                counterparty_profiles=counterparty_profiles,
                expected_loss=expected_loss,
                unexpected_loss=unexpected_loss,
                concentration_risk=concentration_risk,
                correlation_matrix=correlation_matrix,
                stress_tests=stress_tests,
                credit_limits=credit_limits,
                risk_signals=risk_signals,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update history
            self._update_history(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Credit risk analysis failed: {e}")
            raise

    def _analyze_counterparties(self, data: Dict) -> Dict[str, CounterpartyProfile]:
        """Analyze individual counterparties"""
        profiles = {}
        
        for counterparty_id, counterparty_data in data['counterparty_data'].items():
            # Calculate credit metrics
            credit_metrics = self._calculate_credit_metrics(counterparty_data)
            
            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics(
                counterparty_data)
            
            # Calculate industry metrics
            industry_metrics = self._calculate_industry_metrics(
                counterparty_data, data['industry_data'])
            
            # Analyze historical performance
            historical_performance = self._analyze_historical_performance(
                counterparty_data)
            
            # Calculate risk indicators
            risk_indicators = self._calculate_risk_indicators(
                credit_metrics, financial_metrics, industry_metrics)
            
            profiles[counterparty_id] = CounterpartyProfile(
                id=counterparty_id,
                credit_metrics=credit_metrics,
                financial_metrics=financial_metrics,
                industry_metrics=industry_metrics,
                historical_performance=historical_performance,
                risk_indicators=risk_indicators
            )
            
        return profiles

    def _calculate_credit_metrics(self, data: Dict) -> CreditMetrics:
        """Calculate credit metrics for a counterparty"""
        # Calculate default probability
        pd = self._calculate_default_probability(data)
        
        # Estimate recovery rate
        recovery_rate = self._estimate_recovery_rate(data)
        
        # Calculate credit exposure
        exposure = self._calculate_credit_exposure(data)
        
        # Determine credit rating
        rating = self._determine_credit_rating(data)
        
        # Calculate credit spread
        spread = self._calculate_credit_spread(data)
        
        # Calculate transition probabilities
        transition_probs = self._calculate_transition_probabilities(data)
        
        return CreditMetrics(
            default_probability=pd,
            recovery_rate=recovery_rate,
            credit_exposure=exposure,
            credit_rating=rating,
            credit_spread=spread,
            transition_probability=transition_probs
        )

    def _calculate_portfolio_metrics(self, 
                                  profiles: Dict[str, CounterpartyProfile]) -> Dict:
        """Calculate portfolio-level credit metrics"""
        # Weighted average rating
        avg_rating = self._calculate_average_rating(profiles)
        
        # Portfolio default risk
        portfolio_pd = self._calculate_portfolio_pd(profiles)
        
        # Portfolio concentration
        concentration = self._calculate_portfolio_concentration(profiles)
        
        # Rating distribution
        rating_dist = self._calculate_rating_distribution(profiles)
        
        return {
            'average_rating': avg_rating,
            'portfolio_pd': portfolio_pd,
            'concentration': concentration,
            'rating_distribution': rating_dist,
            'risk_metrics': self._calculate_portfolio_risk_metrics(profiles)
        }

    def _calculate_expected_loss(self, 
                               profiles: Dict[str, CounterpartyProfile]) -> float:
        """Calculate portfolio expected loss"""
        total_el = 0
        
        for profile in profiles.values():
            # EL = PD * LGD * EAD
            el = (profile.credit_metrics.default_probability *
                 (1 - profile.credit_metrics.recovery_rate) *
                 profile.credit_metrics.credit_exposure)
            total_el += el
            
        return total_el

    def _calculate_unexpected_loss(self, 
                                 profiles: Dict[str, CounterpartyProfile]) -> float:
        """Calculate portfolio unexpected loss"""
        # Calculate variance of losses
        loss_variance = self._calculate_loss_variance(profiles)
        
        # Calculate correlation effect
        correlation_effect = self._calculate_correlation_effect(profiles)
        
        # Calculate unexpected loss
        ul = np.sqrt(loss_variance * (1 + correlation_effect))
        
        return ul

    def _analyze_concentration_risk(self, 
                                  profiles: Dict[str, CounterpartyProfile]) -> Dict:
        """Analyze portfolio concentration risk"""
        # Single name concentration
        single_name = self._analyze_single_name_concentration(profiles)
        
        # Sector concentration
        sector = self._analyze_sector_concentration(profiles)
        
        # Geographic concentration
        geographic = self._analyze_geographic_concentration(profiles)
        
        # Rating concentration
        rating = self._analyze_rating_concentration(profiles)
        
        return {
            'single_name': single_name,
            'sector': sector,
            'geographic': geographic,
            'rating': rating,
            'risk_contribution': self._calculate_concentration_risk_contribution(
                single_name, sector, geographic, rating)
        }

    def _perform_stress_tests(self, profiles: Dict[str, CounterpartyProfile],
                            portfolio_metrics: Dict) -> Dict:
        """Perform credit stress tests"""
        # Rating downgrade stress
        downgrade_stress = self._perform_downgrade_stress(profiles)
        
        # Default rate stress
        default_stress = self._perform_default_stress(profiles)
        
        # Sector stress
        sector_stress = self._perform_sector_stress(profiles)
        
        # Correlation stress
        correlation_stress = self._perform_correlation_stress(profiles)
        
        return {
            'downgrade': downgrade_stress,
            'default': default_stress,
            'sector': sector_stress,
            'correlation': correlation_stress,
            'combined': self._calculate_combined_stress(
                downgrade_stress,
                default_stress,
                sector_stress,
                correlation_stress
            )
        }

    def _monitor_credit_limits(self, profiles: Dict[str, CounterpartyProfile],
                             concentration_risk: Dict) -> Dict:
        """Monitor credit risk limits"""
        # Check single counterparty limits
        single_limits = self._check_single_counterparty_limits(profiles)
        
        # Check concentration limits
        concentration_limits = self._check_concentration_limits(
            concentration_risk)
        
        # Check rating limits
        rating_limits = self._check_rating_limits(profiles)
        
        # Check portfolio limits
        portfolio_limits = self._check_portfolio_limits(profiles)
        
        return {
            'single_counterparty': single_limits,
            'concentration': concentration_limits,
            'rating': rating_limits,
            'portfolio': portfolio_limits,
            'breaches': self._identify_limit_breaches(
                single_limits,
                concentration_limits,
                rating_limits,
                portfolio_limits
            )
        }

    def _generate_risk_signals(self, portfolio_metrics: Dict,
                             concentration_risk: Dict,
                             stress_tests: Dict) -> Dict:
        """Generate credit risk signals"""
        # Rating signals
        rating_signals = self._generate_rating_signals(portfolio_metrics)
        
        # Concentration signals
        concentration_signals = self._generate_concentration_signals(
            concentration_risk)
        
        # Stress signals
        stress_signals = self._generate_stress_signals(stress_tests)
        
        # Early warning signals
        warning_signals = self._generate_warning_signals(portfolio_metrics)
        
        return {
            'rating': rating_signals,
            'concentration': concentration_signals,
            'stress': stress_signals,
            'warning': warning_signals,
            'aggregate': self._aggregate_risk_signals(
                rating_signals,
                concentration_signals,
                stress_signals,
                warning_signals
            )
        }

    def get_risk_summary(self) -> Dict:
        """Get summary of current credit risk state"""
        if not self.credit_history:
            return {}
            
        latest = self.credit_history[-1]
        return {
            'timestamp': latest.timestamp,
            'expected_loss': latest.expected_loss,
            'unexpected_loss': latest.unexpected_loss,
            'worst_concentration': max(latest.concentration_risk.values()),
            'limit_breaches': latest.credit_limits.get('breaches', []),
            'risk_signals': latest.risk_signals['aggregate'],
            'confidence': latest.confidence
        }

    @property
    def required_data(self) -> List[str]:
        """Required data for credit risk analysis"""
        return [
            'counterparty_financials',
            'market_indicators',
            'rating_history',
            'industry_metrics',
            'macroeconomic_data'
        ]
