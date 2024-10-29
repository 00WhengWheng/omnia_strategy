import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from typing import Dict, List, Union
import logging
from datetime import datetime

class GrafanaFinanceMetrics:
    def __init__(self, config: Dict):
        """Initialize Grafana Finance Integration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # InfluxDB configuration
        self.influx_config = {
            'url': config.get('influxdb_url', 'http://localhost:8086'),
            'token': config.get('influxdb_token', 'your_token'),
            'org': config.get('influxdb_org', 'your_org'),
            'bucket': config.get('influxdb_bucket', 'trading_metrics')
        }
        
        # Initialize InfluxDB client
        self.client = self._initialize_client()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def _initialize_client(self) -> InfluxDBClient:
        """Initialize InfluxDB client"""
        try:
            client = InfluxDBClient(
                url=self.influx_config['url'],
                token=self.influx_config['token'],
                org=self.influx_config['org']
            )
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize InfluxDB client: {e}")
            raise

    def write_market_data(self, data: pd.DataFrame, symbol: str) -> None:
        """Write market data to InfluxDB"""
        try:
            for index, row in data.iterrows():
                point = (
                    Point("market_data")
                    .tag("symbol", symbol)
                    .field("open", float(row['open']))
                    .field("high", float(row['high']))
                    .field("low", float(row['low']))
                    .field("close", float(row['close']))
                    .field("volume", float(row['volume']))
                    .time(index, WritePrecision.NS)
                )
                self.write_api.write(
                    bucket=self.influx_config['bucket'],
                    record=point
                )
        except Exception as e:
            self.logger.error(f"Failed to write market data: {e}")

    def write_portfolio_metrics(self, metrics: Dict) -> None:
        """Write portfolio metrics to InfluxDB"""
        try:
            point = (
                Point("portfolio_metrics")
                .tag("type", "portfolio")
                .field("total_value", float(metrics['total_value']))
                .field("cash", float(metrics['cash']))
                .field("equity", float(metrics['equity']))
                .field("margin", float(metrics.get('margin', 0)))
                .field("unrealized_pnl", float(metrics['unrealized_pnl']))
                .field("realized_pnl", float(metrics['realized_pnl']))
                .field("drawdown", float(metrics.get('drawdown', 0)))
                .time(datetime.utcnow(), WritePrecision.NS)
            )
            self.write_api.write(
                bucket=self.influx_config['bucket'],
                record=point
            )
        except Exception as e:
            self.logger.error(f"Failed to write portfolio metrics: {e}")

    def write_risk_metrics(self, metrics: Dict) -> None:
        """Write risk metrics to InfluxDB"""
        try:
            point = (
                Point("risk_metrics")
                .tag("type", "risk")
                .field("var", float(metrics.get('var', 0)))
                .field("expected_shortfall", float(metrics.get('expected_shortfall', 0)))
                .field("sharpe_ratio", float(metrics.get('sharpe_ratio', 0)))
                .field("sortino_ratio", float(metrics.get('sortino_ratio', 0)))
                .field("volatility", float(metrics.get('volatility', 0)))
                .field("beta", float(metrics.get('beta', 0)))
                .time(datetime.utcnow(), WritePrecision.NS)
            )
            self.write_api.write(
                bucket=self.influx_config['bucket'],
                record=point
            )
        except Exception as e:
            self.logger.error(f"Failed to write risk metrics: {e}")

    def write_strategy_metrics(self, strategy_name: str, metrics: Dict) -> None:
        """Write strategy-specific metrics to InfluxDB"""
        try:
            point = (
                Point("strategy_metrics")
                .tag("strategy", strategy_name)
                .field("win_rate", float(metrics.get('win_rate', 0)))
                .field("profit_factor", float(metrics.get('profit_factor', 0)))
                .field("avg_win", float(metrics.get('avg_win', 0)))
                .field("avg_loss", float(metrics.get('avg_loss', 0)))
                .field("total_trades", int(metrics.get('total_trades', 0)))
                .field("active_trades", int(metrics.get('active_trades', 0)))
                .time(datetime.utcnow(), WritePrecision.NS)
            )
            self.write_api.write(
                bucket=self.influx_config['bucket'],
                record=point
            )
        except Exception as e:
            self.logger.error(f"Failed to write strategy metrics: {e}")

    def write_trade_signals(self, signals: List[Dict]) -> None:
        """Write trade signals to InfluxDB"""
        try:
            for signal in signals:
                point = (
                    Point("trade_signals")
                    .tag("strategy", signal['strategy'])
                    .tag("symbol", signal['symbol'])
                    .tag("direction", signal['direction'])
                    .field("strength", float(signal['strength']))
                    .field("confidence", float(signal['confidence']))
                    .field("entry_price", float(signal['entry_price']))
                    .time(datetime.utcnow(), WritePrecision.NS)
                )
                self.write_api.write(
                    bucket=self.influx_config['bucket'],
                    record=point
                )
        except Exception as e:
            self.logger.error(f"Failed to write trade signals: {e}")

    def close(self) -> None:
        """Close InfluxDB client connection"""
        try:
            self.client.close()
        except Exception as e:
            self.logger.error(f"Failed to close InfluxDB client: {e}")
