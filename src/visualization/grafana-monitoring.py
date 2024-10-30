from typing import Dict, List, Optional, Union
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from ..core.config import Config
from ..risk.manager import RiskManager
from ..core.engine import Engine

class TradingMonitor:
    def __init__(self, config: Config):
        """Initialize Trading Monitor"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # InfluxDB configuration
        self.influx_config = {
            'url': config.get('monitoring.influxdb_url', 'http://localhost:8086'),
            'token': config.get('monitoring.influxdb_token', 'your_token'),
            'org': config.get('monitoring.influxdb_org', 'your_org'),
            'bucket': config.get('monitoring.influxdb_bucket', 'trading_metrics')
        }
        
        self.client = self._initialize_client()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
        # Create dashboard if it doesn't exist
        self._setup_dashboard()

    def _initialize_client(self) -> InfluxDBClient:
        """Initialize InfluxDB client"""
        try:
            return InfluxDBClient(
                url=self.influx_config['url'],
                token=self.influx_config['token'],
                org=self.influx_config['org']
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize InfluxDB client: {e}")
            raise

    def _setup_dashboard(self) -> None:
        """Setup Grafana dashboard"""
        dashboard_path = Path(__file__).parent / 'grafana-dashboard.json'
        if dashboard_path.exists():
            # Implementation for dashboard provisioning would go here
            pass

    def monitor_analysis(self, analysis_results: Dict) -> None:
        """Monitor analysis components"""
        try:
            for analyzer_name, result in analysis_results.items():
                point = (
                    Point("analysis_metrics")
                    .tag("analyzer", analyzer_name)
                    .field("value", float(result.value))
                    .field("confidence", float(result.confidence))
                )
                
                # Add component-specific fields
                for name, value in result.components.items():
                    if isinstance(value, (int, float)):
                        point = point.field(name, float(value))
                
                self.write_api.write(
                    bucket=self.influx_config['bucket'],
                    record=point
                )
        except Exception as e:
            self.logger.error(f"Failed to write analysis metrics: {e}")

    def monitor_risk(self, risk_manager: RiskManager) -> None:
        """Monitor risk metrics"""
        try:
            risk_metrics = risk_manager.get_risk_report()
            
            point = (
                Point("risk_metrics")
                .tag("type", "portfolio")
                .field("total_exposure", float(risk_metrics['portfolio_state']['total_equity']))
                .field("open_positions", int(risk_metrics['portfolio_state']['open_positions']))
                .field("current_drawdown", float(risk_metrics['portfolio_state']['current_drawdown']))
                .field("var_95", float(risk_metrics['risk_metrics']['var_95'] or 0))
                .field("sharpe_ratio", float(risk_metrics['risk_metrics']['sharpe_ratio'] or 0))
            )
            
            self.write_api.write(
                bucket=self.influx_config['bucket'],
                record=point
            )
        except Exception as e:
            self.logger.error(f"Failed to write risk metrics: {e}")

    def monitor_strategies(self, strategies: Dict) -> None:
        """Monitor strategy performance"""
        try:
            for name, strategy in strategies.items():
                metrics = strategy.get_performance_metrics()
                
                point = (
                    Point("strategy_metrics")
                    .tag("strategy", name)
                    .field("total_trades", int(metrics.get('total_trades', 0)))
                    .field("win_rate", float(metrics.get('win_rate', 0)))
                    .field("profit_factor", float(metrics.get('profit_factor', 0)))
                    .field("avg_trade_duration", float(metrics.get('avg_trade_duration', 0)))
                    .field("max_drawdown", float(metrics.get('max_drawdown', 0)))
                )
                
                self.write_api.write(
                    bucket=self.influx_config['bucket'],
                    record=point
                )
        except Exception as e:
            self.logger.error(f"Failed to write strategy metrics: {e}")

    def monitor_engine(self, engine: Engine) -> None:
        """Monitor trading engine metrics"""
        try:
            engine_metrics = engine.get_metrics()
            
            point = (
                Point("engine_metrics")
                .tag("type", "execution")
                .field("orders_processed", int(engine_metrics.get('orders_processed', 0)))
                .field("fills_processed", int(engine_metrics.get('fills_processed', 0)))
                .field("active_orders", int(engine_metrics.get('active_orders', 0)))
                .field("latency_ms", float(engine_metrics.get('latency_ms', 0)))
            )
            
            self.write_api.write(
                bucket=self.influx_config['bucket'],
                record=point
            )
        except Exception as e:
            self.logger.error(f"Failed to write engine metrics: {e}")

    def monitor_backtest(self, backtest_results: Dict) -> None:
        """Monitor backtest results"""
        try:
            # Write portfolio value timeseries
            for timestamp, value in backtest_results['portfolio_value'].items():
                point = (
                    Point("backtest_metrics")
                    .tag("metric", "portfolio_value")
                    .field("value", float(value))
                    .time(timestamp)
                )
                self.write_api.write(
                    bucket=self.influx_config['bucket'],
                    record=point
                )
                
            # Write trade metrics
            for trade in backtest_results['trades']:
                point = (
                    Point("backtest_trades")
                    .tag("strategy", trade['strategy'])
                    .tag("symbol", trade['symbol'])
                    .field("pnl", float(trade['pnl']))
                    .field("return", float(trade['return']))
                    .time(trade['timestamp'])
                )
                self.write_api.write(
                    bucket=self.influx_config['bucket'],
                    record=point
                )
        except Exception as e:
            self.logger.error(f"Failed to write backtest metrics: {e}")

    def close(self) -> None:
        """Close InfluxDB client connection"""
        try:
            self.client.close()
        except Exception as e:
            self.logger.error(f"Failed to close InfluxDB client: {e}")
