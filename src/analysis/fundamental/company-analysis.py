from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import logging
from ..base import BaseAnalyzer

@dataclass
class FinancialStatements:
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    period: str  # 'quarterly' or 'annual'
    last_update: datetime

@dataclass
class CompanyMetrics:
    profitability: Dict[str, float]
    liquidity: Dict[str, float]
    solvency: Dict[str, float]
    efficiency: Dict[str, float]
    valuation: Dict[str, float]
    growth: Dict[str, float]
    quality: Dict[str, float]

@dataclass
class CompanyState:
    timestamp: datetime
    company: str
    metrics: CompanyMetrics
    financial_health: Dict[str, float]
    market_position: Dict[str, any]
    competitive_analysis: Dict[str, any]
    valuation_analysis: Dict[str, float]
    risk_analysis: Dict[str, float]
    quality_score: float
    signals: Dict[str, float]
    forecast: Dict[str, any]
    confidence: float
    metadata: Dict

class CompanyAnalyzer(BaseAnalyzer):
    """Company Fundamental Analysis Component"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.metrics_weights = self.config.get('metrics_weights', {
            'profitability': 0.25,
            'liquidity': 0.15,
            'solvency': 0.15,
            'efficiency': 0.15,
            'growth': 0.15,
            'quality': 0.15
        })
        
        # Analysis thresholds
        self.warning_thresholds = self.config.get('warning_thresholds', {
            'current_ratio': 1.5,
            'debt_equity': 2.0,
            'interest_coverage': 2.0,
            'profit_margin': 0.05
        })
        
        # Valuation parameters
        self.discount_rate = self.config.get('discount_rate', 0.1)
        self.growth_rates = self.config.get('growth_rates', {
            'high': 0.15,
            'medium': 0.10,
            'low': 0.05
        })
        
        # Analysis cache
        self.company_history = {}
        self.peer_comparison = {}

    def analyze(self, company_data: Dict) -> CompanyState:
        """
        Perform comprehensive company analysis
        
        Parameters:
        - company_data: Dictionary containing:
            - financial_statements: FinancialStatements
            - market_data: Market prices and volumes
            - peer_data: Peer company data
            - industry_data: Industry metrics
        
        Returns:
        - CompanyState object containing analysis results
        """
        try:
            statements = company_data['financial_statements']
            market_data = company_data.get('market_data', pd.DataFrame())
            peer_data = company_data.get('peer_data', {})
            industry_data = company_data.get('industry_data', {})
            
            # Calculate financial metrics
            metrics = self._calculate_metrics(statements)
            
            # Analyze financial health
            financial_health = self._analyze_financial_health(
                metrics, statements)
            
            # Analyze market position
            market_position = self._analyze_market_position(
                metrics, market_data, peer_data, industry_data)
            
            # Analyze competitive position
            competitive_analysis = self._analyze_competitive_position(
                metrics, market_position, industry_data)
            
            # Perform valuation analysis
            valuation_analysis = self._perform_valuation(
                statements, metrics, market_data)
            
            # Analyze risks
            risk_analysis = self._analyze_risks(
                metrics, financial_health, market_position)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                metrics, financial_health, competitive_analysis)
            
            # Generate signals
            signals = self._generate_signals(
                metrics, valuation_analysis, quality_score, risk_analysis)
            
            # Generate forecasts
            forecast = self._generate_forecasts(
                statements, metrics, market_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                statements, metrics, market_data)
            
            # Generate metadata
            metadata = self._generate_metadata(company_data)
            
            state = CompanyState(
                timestamp=datetime.now(),
                company=company_data['company_id'],
                metrics=metrics,
                financial_health=financial_health,
                market_position=market_position,
                competitive_analysis=competitive_analysis,
                valuation_analysis=valuation_analysis,
                risk_analysis=risk_analysis,
                quality_score=quality_score,
                signals=signals,
                forecast=forecast,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update history
            self._update_history(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Company analysis failed: {e}")
            raise

    def _calculate_metrics(self, statements: FinancialStatements) -> CompanyMetrics:
        """Calculate comprehensive financial metrics"""
        # Profitability metrics
        profitability = self._calculate_profitability_metrics(statements)
        
        # Liquidity metrics
        liquidity = self._calculate_liquidity_metrics(statements)
        
        # Solvency metrics
        solvency = self._calculate_solvency_metrics(statements)
        
        # Efficiency metrics
        efficiency = self._calculate_efficiency_metrics(statements)
        
        # Valuation metrics
        valuation = self._calculate_valuation_metrics(statements)
        
        # Growth metrics
        growth = self._calculate_growth_metrics(statements)
        
        # Quality metrics
        quality = self._calculate_quality_metrics(
            profitability, efficiency, growth)
        
        return CompanyMetrics(
            profitability=profitability,
            liquidity=liquidity,
            solvency=solvency,
            efficiency=efficiency,
            valuation=valuation,
            growth=growth,
            quality=quality
        )

    def _calculate_profitability_metrics(self, 
                                       statements: FinancialStatements) -> Dict[str, float]:
        """Calculate profitability metrics"""
        income_stmt = statements.income_statement
        balance_sheet = statements.balance_sheet
        
        try:
            return {
                'gross_margin': (
                    income_stmt['gross_profit'] / income_stmt['revenue']
                ).iloc[-1],
                'operating_margin': (
                    income_stmt['operating_income'] / income_stmt['revenue']
                ).iloc[-1],
                'net_margin': (
                    income_stmt['net_income'] / income_stmt['revenue']
                ).iloc[-1],
                'roa': (
                    income_stmt['net_income'] / balance_sheet['total_assets'].mean()
                ).iloc[-1],
                'roe': (
                    income_stmt['net_income'] / balance_sheet['total_equity'].mean()
                ).iloc[-1],
                'roic': self._calculate_roic(statements)
            }
        except Exception as e:
            self.logger.error(f"Error calculating profitability metrics: {e}")
            return {}

    def _analyze_financial_health(self, metrics: CompanyMetrics,
                                statements: FinancialStatements) -> Dict:
        """Analyze company's financial health"""
        # Analyze trends
        trend_analysis = self._analyze_financial_trends(statements)
        
        # Analyze stability
        stability_analysis = self._analyze_financial_stability(statements)
        
        # Analyze red flags
        red_flags = self._detect_financial_red_flags(metrics, statements)
        
        # Credit analysis
        credit_analysis = self._analyze_credit_metrics(metrics, statements)
        
        return {
            'trends': trend_analysis,
            'stability': stability_analysis,
            'red_flags': red_flags,
            'credit_metrics': credit_analysis,
            'overall_health': self._calculate_health_score(
                trend_analysis, stability_analysis, red_flags)
        }

    def _perform_valuation(self, statements: FinancialStatements,
                          metrics: CompanyMetrics,
                          market_data: pd.DataFrame) -> Dict:
        """Perform company valuation analysis"""
        # DCF Valuation
        dcf_value = self._calculate_dcf_valuation(statements)
        
        # Multiple based valuation
        multiple_value = self._calculate_multiple_valuation(
            metrics, market_data)
        
        # Asset based valuation
        asset_value = self._calculate_asset_valuation(statements)
        
        # Earnings power value
        epv = self._calculate_earnings_power_value(statements)
        
        return {
            'dcf_value': dcf_value,
            'multiple_value': multiple_value,
            'asset_value': asset_value,
            'epv': epv,
            'composite_value': self._calculate_composite_value(
                dcf_value, multiple_value, asset_value, epv),
            'confidence_range': self._calculate_valuation_range(
                dcf_value, multiple_value, asset_value, epv)
        }

    def _analyze_competitive_position(self, metrics: CompanyMetrics,
                                    market_position: Dict,
                                    industry_data: Dict) -> Dict:
        """Analyze company's competitive position"""
        # Market share analysis
        market_share = self._analyze_market_share(market_position, industry_data)
        
        # Competitive advantages
        advantages = self._analyze_competitive_advantages(
            metrics, market_position)
        
        # Industry position
        industry_position = self._analyze_industry_position(
            metrics, industry_data)
        
        # Moat analysis
        moat = self._analyze_economic_moat(metrics, advantages)
        
        return {
            'market_share': market_share,
            'competitive_advantages': advantages,
            'industry_position': industry_position,
            'economic_moat': moat,
            'sustainability': self._analyze_competitive_sustainability(
                advantages, moat)
        }

    def _generate_forecasts(self, statements: FinancialStatements,
                          metrics: CompanyMetrics,
                          market_data: pd.DataFrame) -> Dict:
        """Generate company forecasts"""
        # Revenue forecast
        revenue_forecast = self._forecast_revenue(statements, metrics)
        
        # Earnings forecast
        earnings_forecast = self._forecast_earnings(
            statements, metrics, revenue_forecast)
        
        # Cash flow forecast
        cash_flow_forecast = self._forecast_cash_flows(
            statements, metrics, revenue_forecast)
        
        # Balance sheet forecast
        balance_sheet_forecast = self._forecast_balance_sheet(
            statements, metrics, revenue_forecast)
        
        return {
            'revenue': revenue_forecast,
            'earnings': earnings_forecast,
            'cash_flow': cash_flow_forecast,
            'balance_sheet': balance_sheet_forecast,
            'confidence': self._calculate_forecast_confidence(
                statements, metrics),
            'scenarios': self._generate_forecast_scenarios(
                revenue_forecast, earnings_forecast, cash_flow_forecast)
        }

    def _generate_signals(self, metrics: CompanyMetrics,
                         valuation: Dict,
                         quality_score: float,
                         risk_analysis: Dict) -> Dict:
        """Generate investment signals"""
        # Valuation signals
        valuation_signals = self._generate_valuation_signals(valuation)
        
        # Quality signals
        quality_signals = self._generate_quality_signals(
            metrics, quality_score)
        
        # Risk signals
        risk_signals = self._generate_risk_signals(risk_analysis)
        
        # Momentum signals
        momentum_signals = self._generate_momentum_signals(metrics)
        
        return {
            'valuation': valuation_signals,
            'quality': quality_signals,
            'risk': risk_signals,
            'momentum': momentum_signals,
            'composite': self._calculate_composite_signal(
                valuation_signals,
                quality_signals,
                risk_signals,
                momentum_signals
            )
        }

    def _calculate_quality_score(self, metrics: CompanyMetrics,
                               financial_health: Dict,
                               competitive_analysis: Dict) -> float:
        """Calculate overall company quality score"""
        # Financial quality
        financial_quality = self._calculate_financial_quality(metrics)
        
        # Business quality
        business_quality = self._calculate_business_quality(
            competitive_analysis)
        
        # Management quality
        management_quality = self._calculate_management_quality(
            metrics, financial_health)
        
        # Weight components
        quality_score = (
            financial_quality * 0.4 +
            business_quality * 0.4 +
            management_quality * 0.2
        )
        
        return np.clip(quality_score, 0, 1)

    def _calculate_confidence(self, statements: FinancialStatements,
                            metrics: CompanyMetrics,
                            market_data: pd.DataFrame) -> float:
        """Calculate confidence in analysis"""
        # Data quality confidence
        data_conf = self._calculate_data_confidence(statements)
        
        # Metric confidence
        metric_conf = self._calculate_metric_confidence(metrics)
        
        # Forecast confidence
        forecast_conf = self._calculate_forecast_confidence(
            statements, metrics)
        
        # Market data confidence
        market_conf = self._calculate_market_confidence(market_data)
        
        # Weight components
        confidence = (
            data_conf * 0.3 +
            metric_conf * 0.3 +
            forecast_conf * 0.2 +
            market_conf * 0.2
        )
        
        return np.clip(confidence, 0, 1)

    @property
    def required_data(self) -> Dict[str, List[str]]:
        """Required data fields for company analysis"""
        return {
            'income_statement': [
                'revenue', 'gross_profit', 'operating_income',
                'net_income', 'ebitda'
            ],
            'balance_sheet': [
                'total_assets', 'total_liabilities', 'total_equity',
                'current_assets', 'current_liabilities'
            ],
            'cash_flow': [
                'operating_cash_flow', 'investing_cash_flow',
                'financing_cash_flow', 'capex'
            ]
        }

    def get_analysis_summary(self) -> Dict:
        """Get summary of current company state"""
        if not self.company_history:
            return {}
            
        latest = list(self.company_history.values())[-1]
        return {
            'timestamp': latest.timestamp,
            'company': latest.company,
            'quality_score': latest.quality_score,
            'valuation': latest.valuation_analysis['composite_value'],
            'financial_health': latest.financial_health['overall_health'],
            'signals': latest.signals['composite'],
            'confidence': latest.confidence
        }
