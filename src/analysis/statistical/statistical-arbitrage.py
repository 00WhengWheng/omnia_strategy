from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from ..base import BaseAnalyzer

@dataclass
class PairState:
    pair: Tuple[str, str]           # Asset pair
    beta: float                     # Hedge ratio
    spread: pd.Series              # Price spread
    zscore: float                  # Current z-score
    half_life: float               # Mean reversion half-life
    correlation: float             # Correlation coefficient
    cointegration_score: float     # Cointegration test result
    residuals: pd.Series          # Regression residuals
    metrics: Dict[str, float]      # Trading metrics

@dataclass
class ArbitrageState:
    timestamp: datetime
    pairs: List[PairState]         # Analyzed pairs
    signals: Dict[str, Dict]       # Trading signals
    clusters: Dict[str, List]      # Asset clusters
    opportunities: List[Dict]      # Current opportunities
    risk_metrics: Dict[str, float] # Risk measures
    performance: Dict[str, float]  # Performance metrics
    confidence: float
    metadata: Dict

class StatisticalArbitrageAnalyzer(BaseAnalyzer):
    """Statistical Arbitrage Analysis Component"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.lookback_window = self.config.get('lookback_window', 252)
        self.min_half_life = self.config.get('min_half_life', 1)
        self.max_half_life = self.config.get('max_half_life', 252)
        self.zscore_threshold = self.config.get('zscore_threshold', 2.0)
        self.min_correlation = self.config.get('min_correlation', 0.7)
        self.cointegration_pvalue = self.config.get('cointegration_pvalue', 0.05)
        
        # Clustering parameters
        self.n_clusters = self.config.get('n_clusters', 5)
        self.min_cluster_size = self.config.get('min_cluster_size', 3)
        
        # Analysis cache
        self.pair_history: Dict[Tuple[str, str], List[PairState]] = {}
        self.analysis_history: List[ArbitrageState] = []

    def analyze(self, data: Dict[str, pd.DataFrame]) -> ArbitrageState:
        """
        Perform statistical arbitrage analysis
        
        Parameters:
        - data: Dictionary of asset price DataFrames
        
        Returns:
        - ArbitrageState object containing analysis results
        """
        try:
            # Find tradeable pairs
            pairs = self._find_pairs(data)
            
            # Analyze each pair
            pair_states = []
            for pair in pairs:
                state = self._analyze_pair(
                    data[pair[0]], data[pair[1]], pair)
                if self._validate_pair(state):
                    pair_states.append(state)
            
            # Generate trading signals
            signals = self._generate_signals(pair_states)
            
            # Find asset clusters
            clusters = self._find_clusters(data)
            
            # Identify opportunities
            opportunities = self._identify_opportunities(pair_states)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(pair_states)
            
            # Calculate performance metrics
            performance = self._calculate_performance(pair_states)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                pair_states, signals, opportunities)
            
            # Generate metadata
            metadata = self._generate_metadata(
                data, pair_states, signals, opportunities)
            
            state = ArbitrageState(
                timestamp=datetime.now(),
                pairs=pair_states,
                signals=signals,
                clusters=clusters,
                opportunities=opportunities,
                risk_metrics=risk_metrics,
                performance=performance,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update history
            self._update_history(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Statistical arbitrage analysis failed: {e}")
            raise

    def _find_pairs(self, data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """Find potential trading pairs"""
        pairs = []
        assets = list(data.keys())
        
        # Calculate correlation matrix
        returns = pd.DataFrame({
            asset: data[asset]['returns']
            for asset in assets
        })
        corr_matrix = returns.corr()
        
        # Find highly correlated pairs
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) >= self.min_correlation:
                    pairs.append((assets[i], assets[j]))
        
        return pairs

    def _analyze_pair(self, asset1: pd.DataFrame, 
                     asset2: pd.DataFrame, 
                     pair: Tuple[str, str]) -> PairState:
        """Analyze a potential trading pair"""
        # Calculate hedge ratio
        beta = self._calculate_hedge_ratio(asset1['close'], asset2['close'])
        
        # Calculate spread
        spread = self._calculate_spread(
            asset1['close'], asset2['close'], beta)
        
        # Calculate z-score
        zscore = self._calculate_zscore(spread)
        
        # Calculate half-life
        half_life = self._calculate_half_life(spread)
        
        # Test for cointegration
        coint_score = self._test_cointegration(
            asset1['close'], asset2['close'])
        
        # Calculate residuals
        residuals = self._calculate_residuals(
            asset1['close'], asset2['close'], beta)
        
        # Calculate pair metrics
        metrics = self._calculate_pair_metrics(spread, zscore)
        
        return PairState(
            pair=pair,
            beta=beta,
            spread=spread,
            zscore=zscore,
            half_life=half_life,
            correlation=asset1['returns'].corr(asset2['returns']),
            cointegration_score=coint_score,
            residuals=residuals,
            metrics=metrics
        )

    def _calculate_hedge_ratio(self, price1: pd.Series, 
                             price2: pd.Series) -> float:
        """Calculate hedge ratio between two assets"""
        model = OLS(price1, price2).fit()
        return model.params[0]

    def _calculate_spread(self, price1: pd.Series, 
                         price2: pd.Series, 
                         beta: float) -> pd.Series:
        """Calculate price spread between two assets"""
        return price1 - beta * price2

    def _calculate_zscore(self, spread: pd.Series) -> float:
        """Calculate z-score of the spread"""
        return (spread[-1] - spread.mean()) / spread.std()

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life"""
        lagged_spread = spread.shift(1)
        delta_spread = spread - lagged_spread
        model = OLS(delta_spread[1:], lagged_spread[1:]).fit()
        half_life = -np.log(2) / model.params[0]
        return half_life

    def _test_cointegration(self, price1: pd.Series, 
                           price2: pd.Series) -> float:
        """Test for cointegration between two assets"""
        _, pvalue, _ = coint(price1, price2)
        return 1 - pvalue

    def _find_clusters(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List]:
        """Find clusters of related assets"""
        # Prepare return matrix
        returns = pd.DataFrame({
            asset: data[asset]['returns']
            for asset in data.keys()
        })
        
        # Standardize returns
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42
        ).fit(scaled_returns)
        
        # Organize clusters
        clusters = {}
        for i in range(self.n_clusters):
            cluster_assets = [
                asset for asset, label 
                in zip(data.keys(), kmeans.labels_) 
                if label == i
            ]
            if len(cluster_assets) >= self.min_cluster_size:
                clusters[f'cluster_{i}'] = cluster_assets
                
        return clusters

    def _generate_signals(self, pair_states: List[PairState]) -> Dict[str, Dict]:
        """Generate trading signals for pairs"""
        signals = {}
        
        for state in pair_states:
            # Entry signals
            entry_signals = self._generate_entry_signals(state)
            
            # Exit signals
            exit_signals = self._generate_exit_signals(state)
            
            # Position sizing
            position_sizes = self._calculate_position_sizes(state)
            
            signals[f"{state.pair[0]}_{state.pair[1]}"] = {
                'entry': entry_signals,
                'exit': exit_signals,
                'position_sizes': position_sizes,
                'confidence': self._calculate_signal_confidence(state)
            }
            
        return signals

    def _identify_opportunities(self, 
                              pair_states: List[PairState]) -> List[Dict]:
        """Identify current trading opportunities"""
        opportunities = []
        
        for state in pair_states:
            if self._is_opportunity(state):
                opportunity = {
                    'pair': state.pair,
                    'type': self._determine_opportunity_type(state),
                    'zscore': state.zscore,
                    'half_life': state.half_life,
                    'expected_return': self._calculate_expected_return(state),
                    'risk_metrics': self._calculate_opportunity_risk(state),
                    'confidence': self._calculate_opportunity_confidence(state)
                }
                opportunities.append(opportunity)
                
        return sorted(
            opportunities,
            key=lambda x: x['expected_return'],
            reverse=True
        )

    def _calculate_risk_metrics(self, pair_states: List[PairState]) -> Dict:
        """Calculate risk metrics for pairs"""
        return {
            'value_at_risk': self._calculate_var(pair_states),
            'correlation_risk': self._calculate_correlation_risk(pair_states),
            'liquidity_risk': self._calculate_liquidity_risk(pair_states),
            'divergence_risk': self._calculate_divergence_risk(pair_states),
            'regime_risk': self._calculate_regime_risk(pair_states)
        }

    def _validate_pair(self, state: PairState) -> bool:
        """Validate if pair meets trading criteria"""
        return (
            state.half_life >= self.min_half_life and
            state.half_life <= self.max_half_life and
            state.cointegration_score >= (1 - self.cointegration_pvalue) and
            abs(state.correlation) >= self.min_correlation
        )

    def _generate_entry_signals(self, state: PairState) -> Dict:
        """Generate entry signals for a pair"""
        if state.zscore > self.zscore_threshold:
            return {
                'action': 'short',
                'asset1_size': 1.0,
                'asset2_size': state.beta,
                'confidence': self._calculate_entry_confidence(state)
            }
        elif state.zscore < -self.zscore_threshold:
            return {
                'action': 'long',
                'asset1_size': 1.0,
                'asset2_size': state.beta,
                'confidence': self._calculate_entry_confidence(state)
            }
        return {'action': 'none'}

    def _calculate_expected_return(self, state: PairState) -> float:
        """Calculate expected return for a pair trade"""
        # Calculate mean reversion expectation
        mean_rev_return = abs(state.zscore) * state.spread.std()
        
        # Adjust for half-life
        time_adjustment = np.exp(-np.log(2) / state.half_life)
        
        # Consider historical performance
        hist_performance = self._calculate_historical_performance(state)
        
        return mean_rev_return * time_adjustment * hist_performance

    @property
    def required_columns(self) -> List[str]:
        """Required columns for statistical arbitrage analysis"""
        return ['close', 'returns', 'volume']

    def get_analysis_summary(self) -> Dict:
        """Get summary of current arbitrage state"""
        if not self.analysis_history:
            return {}
            
        latest = self.analysis_history[-1]
        return {
            'timestamp': latest.timestamp,
            'active_pairs': len(latest.pairs),
            'opportunities': len(latest.opportunities),
            'top_opportunity': latest.opportunities[0] if latest.opportunities else None,
            'risk_metrics': latest.risk_metrics,
            'confidence': latest.confidence
        }
