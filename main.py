import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import concurrent.futures
from dataclasses import dataclass

# Import core components
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.regime import MarketRegimeAnalyzer
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.volatility import VolatilityAnalyzer
from src.analysis.trade_analytics import TradeAnalytics

# Import strategies
from src.strategies.breakout import BreakoutStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.trend_following import TrendFollowingStrategy

# Import risk management
from src.risk_management.manager import RiskManager
from src.risk_management.risk_engine import RiskEngine, RiskParameters

@dataclass
class SystemState:
    active_strategies: Dict
    analyzers: Dict
    risk_manager: RiskManager
    risk_engine: RiskEngine
    current_positions: Dict
    portfolio_value: float
    trade_history: List
    performance_metrics: Dict

class TradingSystem:
    def __init__(self, config_path: str = 'config/default.yaml'):
        """Initialize the trading system"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.state = self._initialize_system()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_system.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('TradingSystem')

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise

    def _initialize_system(self) -> SystemState:
        """Initialize all system components"""
        self.logger.info("Initializing trading system components...")
        
        # Initialize analyzers
        analyzers = {
            'correlation': CorrelationAnalyzer(self.config),
            'regime': MarketRegimeAnalyzer(self.config),
            'sentiment': SentimentAnalyzer(self.config),
            'volatility': VolatilityAnalyzer(self.config),
            'trade_analytics': TradeAnalytics(self.config)
        }
        
        # Initialize risk management
        risk_params = RiskParameters(
            max_position_size=self.config['risk']['max_position_size'],
            max_portfolio_risk=self.config['risk']['max_portfolio_heat'],
            max_correlation=self.config['risk']['correlation_threshold'],
            max_sector_exposure=0.3,  # Default value
            position_sizing_method='risk_parity',
            use_var=True,
            var_confidence=0.95,
            max_leverage=1.0,
            risk_free_rate=0.02,
            stress_test_scenarios=['crisis', 'recovery', 'high_vol']
        )
        
        risk_engine = RiskEngine(risk_params)
        risk_manager = RiskManager(self.config['risk'])
        
        return SystemState(
            active_strategies={},
            analyzers=analyzers,
            risk_manager=risk_manager,
            risk_engine=risk_engine,
            current_positions={},
            portfolio_value=self.config.get('initial_capital', 100000),
            trade_history=[],
            performance_metrics={}
        )

    def activate_strategies(self, strategy_names: List[str]) -> None:
        """Activate specific strategies"""
        strategy_classes = {
            'breakout': BreakoutStrategy,
            'mean_reversion': MeanReversionStrategy,
            'momentum': MomentumStrategy,
            'trend_following': TrendFollowingStrategy
        }
        
        for name in strategy_names:
            if name not in strategy_classes:
                self.logger.warning(f"Strategy {name} not found")
                continue
                
            try:
                strategy_config = self.config['strategy'].copy()
                strategy_config['name'] = name
                self.state.active_strategies[name] = strategy_classes[name](strategy_config)
                self.logger.info(f"Activated strategy: {name}")
            except Exception as e:
                self.logger.error(f"Failed to activate strategy {name}: {e}")

    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest with historical data"""
        self.logger.info("Starting backtest...")
        results = {
            'trades': [],
            'portfolio_value': [],
            'analysis_results': [],
            'risk_metrics': []
        }
        
        try:
            for timestamp, market_data in self._prepare_market_data(data):
                # Update analysis
                analysis_results = self._run_analysis(market_data)
                
                # Generate strategy signals
                signals = self._generate_strategy_signals(market_data, analysis_results)
                
                # Apply risk management
                filtered_signals = self._apply_risk_management(signals, market_data)
                
                # Execute trades
                trades = self._execute_trades(filtered_signals, market_data)
                
                # Update portfolio
                self._update_portfolio(trades, market_data)
                
                # Track results
                self._track_results(results, trades, analysis_results)
                
            # Calculate final performance metrics
            self._calculate_performance_metrics(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise

    def _prepare_market_data(self, data: pd.DataFrame):
        """Prepare market data for backtesting"""
        min_history = self.config['data']['min_history_required']
        for i in range(min_history, len(data)):
            yield data.index[i], data.iloc[max(0, i-min_history):i+1]

    def _run_analysis(self, market_data: pd.DataFrame) -> Dict:
        """Run all analysis components"""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_analyzer = {
                executor.submit(analyzer.analyze, market_data): name
                for name, analyzer in self.state.analyzers.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_analyzer):
                name = future_to_analyzer[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    self.logger.error(f"Analysis failed for {name}: {e}")
                    
        return results

    def _generate_strategy_signals(self, 
                                 market_data: pd.DataFrame,
                                 analysis_results: Dict) -> Dict:
        """Generate signals from active strategies"""
        signals = {}
        
        for name, strategy in self.state.active_strategies.items():
            try:
                signal = strategy.update(market_data)
                if signal:
                    signals[name] = signal
            except Exception as e:
                self.logger.error(f"Signal generation failed for {name}: {e}")
                
        return signals

    def _apply_risk_management(self, 
                             signals: Dict,
                             market_data: pd.DataFrame) -> Dict:
        """Apply risk management filters"""
        filtered_signals = {}
        portfolio_state = self._get_portfolio_state()
        
        for name, signal in signals.items():
            try:
                # Calculate position size
                position_size = self.state.risk_engine.calculate_position_size(
                    signal, self.state.portfolio_value, market_data)
                
                # Validate trade
                if self.state.risk_engine.validate_trade(
                    {'signal': signal, 'size': position_size},
                    portfolio_state,
                    market_data
                ):
                    signal.metadata['position_size'] = position_size
                    filtered_signals[name] = signal
                    
            except Exception as e:
                self.logger.error(f"Risk management failed for {name}: {e}")
                
        return filtered_signals

    def _execute_trades(self, 
                       signals: Dict,
                       market_data: pd.DataFrame) -> List:
        """Execute trading signals"""
        trades = []
        
        for name, signal in signals.items():
            try:
                trade = self._create_trade(signal, market_data)
                trades.append(trade)
            except Exception as e:
                self.logger.error(f"Trade execution failed for {name}: {e}")
                
        return trades

    def _update_portfolio(self, 
                         trades: List,
                         market_data: pd.DataFrame) -> None:
        """Update portfolio state"""
        try:
            for trade in trades:
                self.state.risk_manager.update_portfolio(trade)
                
            # Update portfolio value
            self.state.portfolio_value = self.state.risk_manager.portfolio.total_equity
            
            # Update positions
            self.state.current_positions = self.state.risk_manager.portfolio.open_positions
            
        except Exception as e:
            self.logger.error(f"Portfolio update failed: {e}")

    def _track_results(self,
                      results: Dict,
                      trades: List,
                      analysis_results: Dict) -> None:
        """Track backtest results"""
        results['trades'].extend(trades)
        results['portfolio_value'].append(self.state.portfolio_value)
        results['analysis_results'].append(analysis_results)
        results['risk_metrics'].append(
            self.state.risk_engine.get_risk_report()
        )

    def _calculate_performance_metrics(self, results: Dict) -> None:
        """Calculate final performance metrics"""
        self.state.performance_metrics = self.state.analyzers['trade_analytics'].analyze_trades(
            results['trades'],
            self.state.portfolio_value
        )

    def _get_portfolio_state(self) -> Dict:
        """Get current portfolio state"""
        return {
            'equity': self.state.portfolio_value,
            'positions': self.state.current_positions,
            'risk_metrics': self.state.risk_engine.risk_metrics
        }

if __name__ == "__main__":
    # Example usage
    system = TradingSystem()
    
    # Activate specific strategies
    system.activate_strategies(['momentum', 'trend_following'])
    
    # Load historical data
    data = pd.read_csv('your_data.csv')  # Replace with your data loading logic
    
    # Run backtest
    results = system.run_backtest(data)
    
    # Access results
    print(f"Final Portfolio Value: {system.state.portfolio_value}")
    print(f"Performance Metrics: {system.state.performance_metrics}")
