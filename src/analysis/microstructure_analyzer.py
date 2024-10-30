from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import logging

@dataclass
class OrderBookSnapshot:
    timestamp: datetime
    bids: List[Tuple[float, float]]  # price, volume
    asks: List[Tuple[float, float]]  # price, volume
    mid_price: float
    spread: float
    bid_depth: float
    ask_depth: float
    imbalance: float

@dataclass
class MicrostructureState:
    timestamp: datetime
    spread_analysis: Dict
    depth_analysis: Dict
    liquidity_analysis: Dict
    order_flow_analysis: Dict
    price_impact: Dict
    volatility_analysis: Dict
    tick_analysis: Dict
    confidence: float
    metadata: Dict

class MicrostructureAnalyzer:
    """Market Microstructure Analysis Component"""
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.depth_levels = config.get('depth_levels', 10)
        self.tick_window = config.get('tick_window', 1000)
        self.volume_bars = config.get('volume_bars', 100)
        self.liquidity_threshold = config.get('liquidity_threshold', 0.1)
        
        # Analysis cache
        self.orderbook_history: List[OrderBookSnapshot] = []
        self.trade_history: pd.DataFrame = pd.DataFrame()
        self.analysis_history: List[MicrostructureState] = []

    def analyze(self, 
                orderbook_data: pd.DataFrame, 
                trade_data: pd.DataFrame,
                additional_data: Optional[Dict] = None) -> MicrostructureState:
        """Perform microstructure analysis"""
        try:
            # Update histories
            self._update_histories(orderbook_data, trade_data)
            
            # Create orderbook snapshot
            snapshot = self._create_orderbook_snapshot(orderbook_data)
            
            # Analyze spread dynamics
            spread_analysis = self._analyze_spread_dynamics(snapshot)
            
            # Analyze market depth
            depth_analysis = self._analyze_market_depth(snapshot)
            
            # Analyze liquidity
            liquidity_analysis = self._analyze_liquidity(snapshot, trade_data)
            
            # Analyze order flow
            order_flow_analysis = self._analyze_order_flow(trade_data)
            
            # Calculate price impact
            price_impact = self._calculate_price_impact(trade_data)
            
            # Analyze tick data
            tick_analysis = self._analyze_tick_data(trade_data)
            
            # Analyze microstructure volatility
            volatility_analysis = self._analyze_microstructure_volatility(
                snapshot, trade_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                spread_analysis,
                depth_analysis,
                liquidity_analysis,
                order_flow_analysis
            )
            
            # Generate metadata
            metadata = self._generate_metadata(
                snapshot, trade_data, additional_data)
            
            state = MicrostructureState(
                timestamp=datetime.now(),
                spread_analysis=spread_analysis,
                depth_analysis=depth_analysis,
                liquidity_analysis=liquidity_analysis,
                order_flow_analysis=order_flow_analysis,
                price_impact=price_impact,
                volatility_analysis=volatility_analysis,
                tick_analysis=tick_analysis,
                confidence=confidence,
                metadata=metadata
            )
            
            self.analysis_history.append(state)
            return state
            
        except Exception as e:
            self.logger.error(f"Microstructure analysis failed: {e}")
            raise

    def _analyze_spread_dynamics(self, snapshot: OrderBookSnapshot) -> Dict:
        """Analyze bid-ask spread dynamics"""
        # Calculate effective spread
        effective_spread = self._calculate_effective_spread(snapshot)
        
        # Calculate realized spread
        realized_spread = self._calculate_realized_spread()
        
        # Analyze spread distribution
        spread_stats = self._analyze_spread_distribution()
        
        # Analyze spread trends
        spread_trend = self._analyze_spread_trend()
        
        return {
            'current_spread': snapshot.spread,
            'effective_spread': effective_spread,
            'realized_spread': realized_spread,
            'spread_stats': spread_stats,
            'spread_trend': spread_trend,
            'abnormal_spread': self._detect_abnormal_spread(snapshot.spread)
        }

    def _analyze_market_depth(self, snapshot: OrderBookSnapshot) -> Dict:
        """Analyze market depth"""
        # Calculate depth imbalance
        depth_imbalance = self._calculate_depth_imbalance(snapshot)
        
        # Analyze price levels
        price_levels = self._analyze_price_levels(snapshot)
        
        # Calculate market impact
        market_impact = self._estimate_market_impact(snapshot)
        
        # Analyze depth distribution
        depth_distribution = self._analyze_depth_distribution(snapshot)
        
        return {
            'imbalance': depth_imbalance,
            'price_levels': price_levels,
            'market_impact': market_impact,
            'depth_distribution': depth_distribution,
            'total_depth': {
                'bid': snapshot.bid_depth,
                'ask': snapshot.ask_depth
            }
        }

    def _analyze_liquidity(self, 
                          snapshot: OrderBookSnapshot,
                          trade_data: pd.DataFrame) -> Dict:
        """Analyze market liquidity"""
        # Calculate various liquidity measures
        volume_weighted_prices = self._calculate_vwap_metrics()
        
        # Analyze order book resilience
        resilience = self._analyze_orderbook_resilience(snapshot)
        
        # Calculate liquidity ratios
        liquidity_ratios = self._calculate_liquidity_ratios(trade_data)
        
        # Analyze trade sizes
        trade_size_analysis = self._analyze_trade_sizes(trade_data)
        
        return {
            'vwap_metrics': volume_weighted_prices,
            'resilience': resilience,
            'liquidity_ratios': liquidity_ratios,
            'trade_size_analysis': trade_size_analysis,
            'current_liquidity': self._assess_current_liquidity(snapshot)
        }

    def _analyze_order_flow(self, trade_data: pd.DataFrame) -> Dict:
        """Analyze order flow"""
        # Classify trades
        trade_classification = self._classify_trades(trade_data)
        
        # Calculate order flow imbalance
        flow_imbalance = self._calculate_flow_imbalance(trade_classification)
        
        # Analyze trade patterns
        trade_patterns = self._analyze_trade_patterns(trade_data)
        
        # Detect toxic order flow
        toxic_flow = self._detect_toxic_flow(trade_data)
        
        return {
            'classification': trade_classification,
            'imbalance': flow_imbalance,
            'patterns': trade_patterns,
            'toxic_flow': toxic_flow,
            'flow_metrics': self._calculate_flow_metrics(trade_data)
        }

    def _calculate_price_impact(self, trade_data: pd.DataFrame) -> Dict:
        """Calculate price impact metrics"""
        # Temporary price impact
        temp_impact = self._calculate_temporary_impact(trade_data)
        
        # Permanent price impact
        perm_impact = self._calculate_permanent_impact(trade_data)
        
        # Total price impact
        total_impact = self._calculate_total_impact(trade_data)
        
        return {
            'temporary': temp_impact,
            'permanent': perm_impact,
            'total': total_impact,
            'impact_decay': self._analyze_impact_decay(trade_data)
        }

    def _analyze_tick_data(self, trade_data: pd.DataFrame) -> Dict:
        """Analyze tick by tick data"""
        # Analyze tick sizes
        tick_sizes = self._analyze_tick_sizes(trade_data)
        
        # Calculate tick frequency
        tick_frequency = self._calculate_tick_frequency(trade_data)
        
        # Analyze tick patterns
        tick_patterns = self._analyze_tick_patterns(trade_data)
        
        return {
            'sizes': tick_sizes,
            'frequency': tick_frequency,
            'patterns': tick_patterns,
            'statistics': self._calculate_tick_statistics(trade_data)
        }

    def _classify_trades(self, trade_data: pd.DataFrame) -> Dict:
        """Classify trades using tick rule and other methods"""
        classifications = {
            'tick': self._apply_tick_rule(trade_data),
            'quote': self._apply_quote_rule(trade_data),
            'lee_ready': self._apply_lee_ready_rule(trade_data)
        }
        
        # Combine classifications using majority vote
        consensus = pd.DataFrame(classifications).mode(axis=1)[0]
        
        return {
            'classifications': classifications,
            'consensus': consensus,
            'buy_volume': sum(trade_data['volume'][consensus == 'buy']),
            'sell_volume': sum(trade_data['volume'][consensus == 'sell'])
        }

    def _calculate_flow_metrics(self, trade_data: pd.DataFrame) -> Dict:
        """Calculate order flow metrics"""
        return {
            'buy_ratio': len(trade_data[trade_data['side'] == 'buy']) / len(trade_data),
            'volume_ratio': sum(trade_data[trade_data['side'] == 'buy']['volume']) / 
                          sum(trade_data['volume']),
            'trade_size_avg': trade_data['volume'].mean(),
            'trade_size_std': trade_data['volume'].std(),
            'trade_intensity': len(trade_data) / 
                             (trade_data.index[-1] - trade_data.index[0]).seconds
        }

    def _detect_toxic_flow(self, trade_data: pd.DataFrame) -> Dict:
        """Detect toxic order flow patterns"""
        # Calculate order flow toxicity using VPIN or similar metrics
        vpin = self._calculate_vpin(trade_data)
        
        # Detect predatory trading patterns
        predatory = self._detect_predatory_patterns(trade_data)
        
        # Analyze price pressure
        price_pressure = self._analyze_price_pressure(trade_data)
        
        return {
            'vpin': vpin,
            'predatory_patterns': predatory,
            'price_pressure': price_pressure,
            'toxicity_score': self._calculate_toxicity_score(
                vpin, predatory, price_pressure)
        }

    def get_analysis_summary(self) -> Dict:
        """Get summary of current microstructure state"""
        if not self.analysis_history:
            return {}
            
        latest = self.analysis_history[-1]
        return {
            'timestamp': latest.timestamp,
            'spread': latest.spread_analysis['current_spread'],
            'depth_imbalance': latest.depth_analysis['imbalance'],
            'liquidity_score': latest.liquidity_analysis['current_liquidity'],
            'order_flow_toxicity': latest.order_flow_analysis['toxic_flow'],
            'confidence': latest.confidence
        }
