from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from datetime import datetime
import logging
from abc import ABC, abstractmethod

@dataclass
class MarketContext:
    """Market context containing all analysis results"""
    timestamp: datetime
    timeframe: str
    market_regime: str
    volatility_state: Dict
    correlation_state: Dict
    liquidity_state: Dict
    technical_state: Dict
    statistical_state: Dict
    intermarket_state: Dict
    confidence: float
    metadata: Dict

class MarketAnalyzer:
    """Enhanced Market Analyzer"""
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize analysis components
        self.analyzers = {
            'regime': MarketRegimeAnalyzer(config),
            'volatility': VolatilityAnalyzer(config),
            'correlation': CorrelationAnalyzer(config),
            'liquidity': LiquidityAnalyzer(config),
            'technical': TechnicalAnalyzer(config),
            'statistical': StatisticalAnalyzer(config),
            'intermarket': IntermarketAnalyzer(config),
            'microstructure': MicrostructureAnalyzer(config)
        }
        
        # Analysis cache
        self.context_history = pd.DataFrame()
        self.feature_cache = {}
        
    def analyze(self, data: pd.DataFrame, 
                additional_data: Optional[Dict] = None) -> MarketContext:
        """Perform comprehensive market analysis"""
        try:
            # 1. Validate and prepare data
            clean_data = self._prepare_data(data)
            
            # 2. Run all analyzers
            analysis_results = self._run_analysis(clean_data, additional_data)
            
            # 3. Build market context
            context = self._build_market_context(analysis_results)
            
            # 4. Update history
            self._update_history(context)
            
            # 5. Generate insights
            self._generate_insights(context)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate data"""
        # Ensure required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required columns")
            
        # Calculate derived features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close']/data['close'].shift(1))
        data['volatility'] = data['returns'].rolling(20).std()
        data['volume_ma'] = data['volume'].rolling(20).mean()
        
        # Remove outliers
        data = self._remove_outliers(data)
        
        return data

    def _run_analysis(self, data: pd.DataFrame, 
                     additional_data: Optional[Dict]) -> Dict:
        """Run all analysis components"""
        results = {}
        
        # Run each analyzer
        for name, analyzer in self.analyzers.items():
            try:
                results[name] = analyzer.analyze(data, additional_data)
            except Exception as e:
                self.logger.error(f"Analyzer {name} failed: {e}")
                results[name] = None
                
        return results

    def _build_market_context(self, results: Dict) -> MarketContext:
        """Build comprehensive market context"""
        # Extract states from results
        regime_state = results['regime']
        volatility_state = results['volatility']
        correlation_state = results['correlation']
        liquidity_state = results['liquidity']
        technical_state = results['technical']
        statistical_state = results['statistical']
        intermarket_state = results['intermarket']
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(results)
        
        # Build metadata
        metadata = {
            'analysis_time': datetime.now(),
            'data_quality': self._assess_data_quality(results),
            'warning_flags': self._check_warning_flags(results),
            'opportunities': self._identify_opportunities(results)
        }
        
        return MarketContext(
            timestamp=datetime.now(),
            timeframe=self.config['timeframe'],
            market_regime=regime_state['current_regime'],
            volatility_state=volatility_state,
            correlation_state=correlation_state,
            liquidity_state=liquidity_state,
            technical_state=technical_state,
            statistical_state=statistical_state,
            intermarket_state=intermarket_state,
            confidence=confidence,
            metadata=metadata
        )

    def _generate_insights(self, context: MarketContext) -> None:
        """Generate trading insights from analysis"""
        insights = {
            'market_condition': self._evaluate_market_condition(context),
            'trading_opportunities': self._identify_trading_opportunities(context),
            'risk_factors': self._identify_risk_factors(context),
            'regime_changes': self._detect_regime_changes(context),
            'anomalies': self._detect_anomalies(context)
        }
        
        context.metadata['insights'] = insights

    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate overall confidence in analysis"""
        # Weight different components
        weights = {
            'regime': 0.2,
            'volatility': 0.15,
            'correlation': 0.15,
            'liquidity': 0.1,
            'technical': 0.15,
            'statistical': 0.15,
            'intermarket': 0.1
        }
        
        # Calculate weighted confidence
        confidence = 0
        for component, weight in weights.items():
            if results[component] and 'confidence' in results[component]:
                confidence += weight * results[component]['confidence']
                
        return min(max(confidence, 0), 1)

    def get_analysis_summary(self) -> Dict:
        """Get summary of current market analysis"""
        if self.context_history.empty:
            return {}
            
        latest = self.context_history.iloc[-1]
        
        return {
            'timestamp': latest['timestamp'],
            'market_regime': latest['market_regime'],
            'volatility_state': latest['volatility_state'],
            'key_opportunities': latest['metadata']['insights']['trading_opportunities'],
            'risk_factors': latest['metadata']['insights']['risk_factors'],
            'confidence': latest['confidence']
        }

    def _assess_data_quality(self, results: Dict) -> Dict:
        """Assess quality of analysis data"""
        quality_metrics = {
            'completeness': self._calculate_data_completeness(results),
            'consistency': self._check_data_consistency(results),
            'timeliness': self._assess_data_timeliness(results)
        }
        return quality_metrics

    def _identify_opportunities(self, results: Dict) -> List:
        """Identify potential trading opportunities"""
        opportunities = []
        
        # Check each component for opportunities
        for component, result in results.items():
            if result and 'opportunities' in result:
                opportunities.extend(result['opportunities'])
                
        return opportunities
