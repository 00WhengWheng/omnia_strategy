from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import logging

@dataclass
class ComprehensiveAnalysis:
    timestamp: datetime
    market_analysis: Dict
    statistical_analysis: Dict
    fundamental_analysis: Dict
    behavioral_analysis: Dict
    geopolitical_analysis: Dict
    risk_analysis: Dict
    composite_score: float
    confidence: float
    metadata: Dict

class ComprehensiveAnalyzer:
    """Comprehensive Market Analysis System"""
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize all analyzers
        self.analyzers = self._initialize_analyzers()
        
        # Analysis cache
        self.analysis_history = pd.DataFrame()
        
    def _initialize_analyzers(self) -> Dict:
        """Initialize all analysis components"""
        return {
            'market': {
                'technical': TechnicalAnalyzer(self.config),
                'microstructure': MicrostructureAnalyzer(self.config),
                'sentiment': SentimentAnalyzer(self.config)
            },
            'statistical': {
                'timeseries': TimeSeriesAnalyzer(self.config),
                'arbitrage': StatArbitrageAnalyzer(self.config),
                'ml': MachineLearningAnalyzer(self.config)
            },
            'fundamental': {
                'economic': EconomicAnalyzer(self.config),
                'company': CompanyAnalyzer(self.config),
                'industry': IndustryAnalyzer(self.config)
            },
            'behavioral': {
                'psychology': MarketPsychologyAnalyzer(self.config),
                'patterns': BehavioralPatternAnalyzer(self.config)
            },
            'geopolitical': {
                'political': PoliticalAnalyzer(self.config),
                'global': GlobalEventAnalyzer(self.config)
            },
            'risk': {
                'market': MarketRiskAnalyzer(self.config),
                'credit': CreditRiskAnalyzer(self.config),
                'systemic': SystemicRiskAnalyzer(self.config)
            }
        }

    def analyze(self, data: Dict) -> ComprehensiveAnalysis:
        """Perform comprehensive analysis"""
        try:
            # Run all analyses
            market_analysis = self._run_market_analysis(data)
            statistical_analysis = self._run_statistical_analysis(data)
            fundamental_analysis = self._run_fundamental_analysis(data)
            behavioral_analysis = self._run_behavioral_analysis(data)
            geopolitical_analysis = self._run_geopolitical_analysis(data)
            risk_analysis = self._run_risk_analysis(data)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score({
                'market': market_analysis,
                'statistical': statistical_analysis,
                'fundamental': fundamental_analysis,
                'behavioral': behavioral_analysis,
                'geopolitical': geopolitical_analysis,
                'risk': risk_analysis
            })
            
            # Calculate confidence
            confidence = self._calculate_analysis_confidence({
                'market': market_analysis,
                'statistical': statistical_analysis,
                'fundamental': fundamental_analysis,
                'behavioral': behavioral_analysis,
                'geopolitical': geopolitical_analysis,
                'risk': risk_analysis
            })
            
            # Generate metadata
            metadata = self._generate_analysis_metadata({
                'market': market_analysis,
                'statistical': statistical_analysis,
                'fundamental': fundamental_analysis,
                'behavioral': behavioral_analysis,
                'geopolitical': geopolitical_analysis,
                'risk': risk_analysis
            })
            
            # Create comprehensive analysis
            analysis = ComprehensiveAnalysis(
                timestamp=datetime.now(),
                market_analysis=market_analysis,
                statistical_analysis=statistical_analysis,
                fundamental_analysis=fundamental_analysis,
                behavioral_analysis=behavioral_analysis,
                geopolitical_analysis=geopolitical_analysis,
                risk_analysis=risk_analysis,
                composite_score=composite_score,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update history
            self._update_history(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            raise

    def _run_market_analysis(self, data: Dict) -> Dict:
        """Run market analysis"""
        results = {}
        for name, analyzer in self.analyzers['market'].items():
            try:
                results[name] = analyzer.analyze(data)
            except Exception as e:
                self.logger.error(f"Market analysis {name} failed: {e}")
        return results

    def _calculate_composite_score(self, analyses: Dict) -> float:
        """Calculate composite analysis score"""
        weights = {
            'market': 0.25,
            'statistical': 0.20,
            'fundamental': 0.20,
            'behavioral': 0.15,
            'geopolitical': 0.10,
            'risk': 0.10
        }
        
        weighted_score = 0
        for category, analysis in analyses.items():
            category_score = self._calculate_category_score(analysis)
            weighted_score += category_score * weights[category]
            
        return np.clip(weighted_score, -1, 1)

    def _generate_analysis_metadata(self, analyses: Dict) -> Dict:
        """Generate analysis metadata"""
        return {
            'analysis_time': datetime.now(),
            'data_quality': self._assess_data_quality(analyses),
            'key_factors': self._identify_key_factors(analyses),
            'warnings': self._generate_warnings(analyses),
            'opportunities': self._identify_opportunities(analyses),
            'risks': self._identify_risks(analyses)
        }

    def get_analysis_summary(self) -> Dict:
        """Get summary of current analysis"""
        if self.analysis_history.empty:
            return {}
            
        latest = self.analysis_history.iloc[-1]
        return {
            'timestamp': latest.timestamp,
            'composite_score': latest.composite_score,
            'confidence': latest.confidence,
            'key_factors': latest.metadata['key_factors'],
            'opportunities': latest.metadata['opportunities'],
            'risks': latest.metadata['risks']
        }
