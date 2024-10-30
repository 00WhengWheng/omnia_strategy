from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import logging
from ..base import BaseAnalyzer

@dataclass
class OrderBookState:
    timestamp: datetime
    bids: List[Tuple[float, float]]  # price, volume
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float
    depth: Dict[str, float]
    imbalance: float

@dataclass
class MicrostructureState:
    timestamp: datetime
    spread_metrics: Dict[str, float]    # Spread and transaction cost metrics
    liquidity_metrics: Dict[str, float] # Liquidity measures
    order_flow: Dict[str, any]         # Order flow imbalance and toxicity
    efficiency: Dict[str, float]       # Market efficiency measures
    volume_profile: Dict[str, any]     # Volume analysis
    price_impact: Dict[str, float]     # Price impact measures
    trade_metrics: Dict[str, float]    # Trade-based metrics
    signals: Dict[str, float]          # Generated signals
    confidence: float
    metadata: Dict

class MicrostructureAnalyzer(BaseAnalyzer):
    """Market Microstructure Analysis Component"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.depth_levels = self.config.get('depth_levels', 10)
        self.vpin_blocks = self.config.get('vpin_blocks', 50)
        self.flow_toxicity_threshold = self.config.get('flow_toxicity_threshold', 0.8)
        self.liquidity_window = self.config.get('liquidity_window', 100)
        self.impact_estimation_method = self.config.get('impact_estimation', 'kyle')
        
        # Analysis cache
        self.order_book_history: List[OrderBookState] = []
        self.trade_history: pd.DataFrame = pd.DataFrame()
        self.analysis_history: List[MicrostructureState] = []

    def analyze(self, data: Dict[str, pd.DataFrame]) -> MicrostructureState:
        """
        Perform market microstructure analysis
        
        Parameters:
        - data: Dictionary containing:
            - order_book: Order book snapshots
            - trades: Trade data
            - quotes: Quote data
        
        Returns:
        - MicrostructureState object containing analysis results
        """
        try:
            # Validate input data
            self._validate_data(data)
            
            # Create order book snapshot
            order_book = self._create_order_book_state(data['order_book'])
            
            # Analyze spreads and transaction costs
            spread_metrics = self._analyze_spreads(data['quotes'], data['trades'])
            
            # Analyze liquidity
            liquidity_metrics = self._analyze_liquidity(order_book, data['trades'])
            
            # Analyze order flow
            order_flow = self._analyze_order_flow(data['trades'], order_book)
            
            # Analyze market efficiency
            efficiency = self._analyze_market_efficiency(data)
            
            # Analyze volume profile
            volume_profile = self._analyze_volume_profile(data['trades'])
            
            # Calculate price impact
            price_impact = self._calculate_price_impact(data['trades'], order_book)
            
            # Analyze trade metrics
            trade_metrics = self._analyze_trade_metrics(data['trades'])
            
            # Generate signals
            signals = self._generate_signals(
                spread_metrics, liquidity_metrics, order_flow, efficiency
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                spread_metrics, liquidity_metrics, order_flow, efficiency
            )
            
            # Generate metadata
            metadata = self._generate_metadata(data)
            
            state = MicrostructureState(
                timestamp=datetime.now(),
                spread_metrics=spread_metrics,
                liquidity_metrics=liquidity_metrics,
                order_flow=order_flow,
                efficiency=efficiency,
                volume_profile=volume_profile,
                price_impact=price_impact,
                trade_metrics=trade_metrics,
                signals=signals,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update history
            self._update_history(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Microstructure analysis failed: {e}")
            raise

    def _analyze_spreads(self, quotes: pd.DataFrame, 
                        trades: pd.DataFrame) -> Dict[str, float]:
        """Analyze various spread metrics"""
        # Calculate quoted spread
        quoted_spread = self._calculate_quoted_spread(quotes)
        
        # Calculate effective spread
        effective_spread = self._calculate_effective_spread(trades, quotes)
        
        # Calculate realized spread
        realized_spread = self._calculate_realized_spread(trades, quotes)
        
        # Calculate spread decomposition
        decomposition = self._decompose_spread(trades, quotes)
        
        return {
            'quoted_spread': quoted_spread,
            'effective_spread': effective_spread,
            'realized_spread': realized_spread,
            'adverse_selection': decomposition['adverse_selection'],
            'inventory_cost': decomposition['inventory_cost'],
            'processing_cost': decomposition['processing_cost']
        }

    def _analyze_liquidity(self, order_book: OrderBookState,
                          trades: pd.DataFrame) -> Dict[str, float]:
        """Analyze market liquidity"""
        # Calculate order book liquidity
        book_liquidity = self._calculate_book_liquidity(order_book)
        
        # Calculate resiliency
        resiliency = self._calculate_resiliency(order_book)
        
        # Calculate trade-based liquidity measures
        trade_liquidity = self._calculate_trade_liquidity(trades)
        
        # Calculate Amihud's ILLIQ
        illiq = self._calculate_amihud_illiq(trades)
        
        # Calculate Kyle's lambda
        kyle_lambda = self._calculate_kyle_lambda(trades)
        
        # Calculate market depth
        market_depth = self._calculate_market_depth(order_book)
        
        return {
            'book_liquidity': book_liquidity,
            'resiliency': resiliency,
            'trade_liquidity': trade_liquidity,
            'illiq': illiq,
            'kyle_lambda': kyle_lambda,
            'market_depth': market_depth
        }

    def _analyze_order_flow(self, trades: pd.DataFrame,
                           order_book: OrderBookState) -> Dict:
        """Analyze order flow dynamics"""
        # Calculate order flow imbalance
        flow_imbalance = self._calculate_flow_imbalance(trades)
        
        # Calculate VPIN (Volume-synchronized Probability of Informed Trading)
        vpin = self._calculate_vpin(trades)
        
        # Detect order flow toxicity
        toxicity = self._detect_flow_toxicity(trades, order_book)
        
        # Analyze trade direction
        trade_direction = self._analyze_trade_direction(trades)
        
        # Calculate order flow runs
        flow_runs = self._calculate_flow_runs(trades)
        
        return {
            'imbalance': flow_imbalance,
            'vpin': vpin,
            'toxicity': toxicity,
            'direction': trade_direction,
            'runs': flow_runs,
            'intensity': self._calculate_flow_intensity(trades)
        }

    def _analyze_market_efficiency(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze market efficiency measures"""
        quotes = data['quotes']
        trades = data['trades']
        
        # Calculate price efficiency ratio
        efficiency_ratio = self._calculate_efficiency_ratio(trades)
        
        # Calculate variance ratio
        variance_ratio = self._calculate_variance_ratio(trades)
        
        # Calculate autocorrelation
        autocorr = self._calculate_return_autocorrelation(trades)
        
        # Calculate relative quote adjustment
        quote_adjustment = self._calculate_quote_adjustment(quotes)
        
        return {
            'efficiency_ratio': efficiency_ratio,
            'variance_ratio': variance_ratio,
            'autocorrelation': autocorr,
            'quote_adjustment': quote_adjustment,
            'price_discovery': self._analyze_price_discovery(trades, quotes)
        }

    def _calculate_price_impact(self, trades: pd.DataFrame,
                              order_book: OrderBookState) -> Dict[str, float]:
        """Calculate price impact measures"""
        # Temporary price impact
        temp_impact = self._calculate_temporary_impact(trades)
        
        # Permanent price impact
        perm_impact = self._calculate_permanent_impact(trades)
        
        # Kyle's lambda estimation
        kyle_lambda = self._estimate_kyle_lambda(trades)
        
        # Price impact decay
        impact_decay = self._calculate_impact_decay(trades)
        
        return {
            'temporary_impact': temp_impact,
            'permanent_impact': perm_impact,
            'kyle_lambda': kyle_lambda,
            'impact_decay': impact_decay,
            'impact_asymmetry': self._calculate_impact_asymmetry(trades)
        }

    def _generate_signals(self, spread_metrics: Dict,
                         liquidity_metrics: Dict,
                         order_flow: Dict,
                         efficiency: Dict) -> Dict[str, float]:
        """Generate trading signals from microstructure analysis"""
        signals = {}
        
        # Liquidity signals
        signals['liquidity'] = self._generate_liquidity_signals(liquidity_metrics)
        
        # Order flow signals
        signals['order_flow'] = self._generate_flow_signals(order_flow)
        
        # Efficiency signals
        signals['efficiency'] = self._generate_efficiency_signals(efficiency)
        
        # Combine signals
        signals['composite'] = self._combine_signals(signals)
        
        return signals

    def _calculate_confidence(self, spread_metrics: Dict,
                            liquidity_metrics: Dict,
                            order_flow: Dict,
                            efficiency: Dict) -> float:
        """Calculate confidence in microstructure analysis"""
        weights = {
            'spread': 0.2,
            'liquidity': 0.3,
            'order_flow': 0.3,
            'efficiency': 0.2
        }
        
        # Calculate component confidences
        confidences = {
            'spread': self._calculate_spread_confidence(spread_metrics),
            'liquidity': self._calculate_liquidity_confidence(liquidity_metrics),
            'order_flow': self._calculate_flow_confidence(order_flow),
            'efficiency': self._calculate_efficiency_confidence(efficiency)
        }
        
        # Calculate weighted confidence
        total_confidence = sum(
            confidences[k] * weights[k] for k in weights.keys()
        )
        
        return np.clip(total_confidence, 0, 1)

    @property
    def required_columns(self) -> Dict[str, List[str]]:
        """Required columns for each data type"""
        return {
            'order_book': ['bid_price', 'bid_size', 'ask_price', 'ask_size'],
            'trades': ['price', 'volume', 'direction'],
            'quotes': ['bid', 'ask', 'bid_size', 'ask_size']
        }

    def get_analysis_summary(self) -> Dict:
        """Get summary of current microstructure state"""
        if not self.analysis_history:
            return {}
            
        latest = self.analysis_history[-1]
        return {
            'timestamp': latest.timestamp,
            'spread': latest.spread_metrics['effective_spread'],
            'liquidity': latest.liquidity_metrics['book_liquidity'],
            'flow_toxicity': latest.order_flow['toxicity'],
            'efficiency': latest.efficiency['efficiency_ratio'],
            'signals': latest.signals['composite'],
            'confidence': latest.confidence
        }
