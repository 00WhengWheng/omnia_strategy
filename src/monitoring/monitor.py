from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import threading
import queue
import json
from pathlib import Path
import psutil
import websockets
import asyncio
from dataclasses import dataclass

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    process_uptime: float
    thread_count: int
    queue_sizes: Dict[str, int]

@dataclass
class TradingMetrics:
    equity: float
    positions: int
    orders: int
    daily_pnl: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    margin_usage: float

class MonitoringSystem:
    def __init__(self, config: Dict):
        """Inizializza il sistema di monitoring"""
        self.config = config
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.metrics_dir = Path(config.get('metrics_dir', 'metrics'))
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self.loggers = self._setup_loggers()
        
        # Metrics storage
        self.system_metrics: List[SystemMetrics] = []
        self.trading_metrics: List[TradingMetrics] = []
        
        # Alert settings
        self.alert_levels = {
            'critical': config.get('critical_threshold', 0.9),
            'warning': config.get('warning_threshold', 0.7),
            'info': config.get('info_threshold', 0.5)
        }
        
        # Alert queues
        self.alert_queue = queue.Queue()
        self.notification_queue = queue.Queue()
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread = None
        self.alert_thread = None
        
        # WebSocket server for real-time monitoring
        self.ws_server = None
        self.connected_clients = set()
        
    def start(self):
        """Avvia il sistema di monitoring"""
        self.is_running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        
        # Start alert thread
        self.alert_thread = threading.Thread(target=self._alert_loop)
        self.alert_thread.start()
        
        # Start WebSocket server
        asyncio.get_event_loop().run_until_complete(self._start_ws_server())
        
        self.log_system_event("Monitoring system started")
        
    def stop(self):
        """Ferma il sistema di monitoring"""
        self.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join()
        if self.alert_thread:
            self.alert_thread.join()
            
        # Close WebSocket server
        if self.ws_server:
            self.ws_server.close()
            
        self.log_system_event("Monitoring system stopped")
        
    def _setup_loggers(self) -> Dict[str, logging.Logger]:
        """Configura i diversi logger"""
        loggers = {}
        
        # System logger
        system_logger = logging.getLogger('system')
        system_logger.setLevel(logging.INFO)
        system_handler = logging.FileHandler(self.log_dir / 'system.log')
        system_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        system_logger.addHandler(system_handler)
        loggers['system'] = system_logger
        
        # Trading logger
        trading_logger = logging.getLogger('trading')
        trading_logger.setLevel(logging.INFO)
        trading_handler = logging.FileHandler(self.log_dir / 'trading.log')
        trading_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        trading_logger.addHandler(trading_handler)
        loggers['trading'] = trading_logger
        
        # Error logger
        error_logger = logging.getLogger('error')
        error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(self.log_dir / 'error.log')
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n'
        ))
        error_logger.addHandler(error_handler)
        loggers['error'] = error_logger
        
        return loggers
        
    def _monitoring_loop(self):
        """Loop principale di monitoring"""
        while self.is_running:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Collect trading metrics
                trading_metrics = self._collect_trading_metrics()
                self.trading_metrics.append(trading_metrics)
                
                # Check alert conditions
                self._check_alert_conditions(system_metrics, trading_metrics)
                
                # Store metrics
                self._store_metrics()
                
                # Broadcast to connected clients
                self._broadcast_metrics(system_metrics, trading_metrics)
                
                # Clean old metrics
                self._cleanup_old_metrics()
                
                # Sleep until next collection
                time.sleep(self.config.get('monitoring_interval', 1))
                
            except Exception as e:
                self.log_error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)
                
    def _alert_loop(self):
        """Loop di gestione alert"""
        while self.is_running:
            try:
                # Process alerts
                while not self.alert_queue.empty():
                    alert = self.alert_queue.get()
                    self._process_alert(alert)
                    
                # Process notifications
                while not self.notification_queue.empty():
                    notification = self.notification_queue.get()
                    self._send_notification(notification)
                    
                time.sleep(0.1)
                
            except Exception as e:
                self.log_error(f"Error in alert loop: {str(e)}")
                time.sleep(5)
                
    async def _start_ws_server(self):
        """Avvia il server WebSocket per monitoring real-time"""
        self.ws_server = await websockets.serve(
            self._handle_ws_connection,
            self.config.get('ws_host', 'localhost'),
            self.config.get('ws_port', 8765)
        )
        
    async def _handle_ws_connection(self, websocket, path):
        """Gestisce connessioni WebSocket"""
        self.connected_clients.add(websocket)
        try:
            while self.is_running:
                message = await websocket.recv()
                await self._handle_ws_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)
            
    def _collect_system_metrics(self) -> SystemMetrics:
        """Raccoglie metriche di sistema"""
        process = psutil.Process()
        
        return SystemMetrics(
            cpu_usage=process.cpu_percent(),
            memory_usage=process.memory_percent(),
            disk_usage=psutil.disk_usage('/').percent,
            network_latency=self._measure_network_latency(),
            process_uptime=time.time() - process.create_time(),
            thread_count=process.num_threads(),
            queue_sizes=self._get_queue_sizes()
        )
        
    def _collect_trading_metrics(self) -> TradingMetrics:
        """Raccoglie metriche di trading"""
        # Get metrics from trading system
        trading_system = self.config.get('trading_system')
        
        return TradingMetrics(
            equity=trading_system.get_equity(),
            positions=len(trading_system.positions),
            orders=len(trading_system.open_orders),
            daily_pnl=trading_system.get_daily_pnl(),
            max_drawdown=trading_system.get_max_drawdown(),
            win_rate=trading_system.get_win_rate(),
            sharpe_ratio=trading_system.get_sharpe_ratio(),
            margin_usage=trading_system.get_margin_usage()
        )
        
    def _check_alert_conditions(self, system_metrics: SystemMetrics,
                              trading_metrics: TradingMetrics):
        """Verifica condizioni per alert"""
        # System alerts
        if system_metrics.cpu_usage > self.alert_levels['critical']:
            self._create_alert('critical', 'High CPU Usage', system_metrics.cpu_usage)
            
        if system_metrics.memory_usage > self.alert_levels['warning']:
            self._create_alert('warning', 'High Memory Usage', system_metrics.memory_usage)
            
        # Trading alerts
        if trading_metrics.margin_usage > self.alert_levels['warning']:
            self._create_alert('warning', 'High Margin Usage', trading_metrics.margin_usage)
            
        if trading_metrics.max_drawdown > self.alert_levels['critical']:
            self._create_alert('critical', 'Excessive Drawdown', trading_metrics.max_drawdown)
            
    def _create_alert(self, level: str, message: str, value: float):
        """Crea un nuovo alert"""
        alert = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message,
            'value': value
        }
        
        self.alert_queue.put(alert)
        self.log_alert(alert)
        
    def _process_alert(self, alert: Dict):
        """Processa un alert"""
        # Log alert
        self.log_alert(alert)
        
        # Create notification
        notification = {
            'type': 'alert',
            'data': alert
        }
        
        # Add to notification queue
        self.notification_queue.put(notification)
        
    def log_system_event(self, message: str):
        """Logga un evento di sistema"""
        self.loggers['system'].info(message)
        
    def log_trading_event(self, message: str):
        """Logga un evento di trading"""
        self.loggers['trading'].info(message)
        
    def log_error(self, message: str):
        """Logga un errore"""
        self.loggers['error'].error(message)
        
    def log_alert(self, alert: Dict):
        """Logga un alert"""
        logger = self.loggers['system']
        level = alert['level'].upper()
        message = f"ALERT [{level}]: {alert['message']} (Value: {alert['value']})"
        
        if level == 'CRITICAL':
            logger.critical(message)
        elif level == 'WARNING':
            logger.warning(message)
        else:
            logger.info(message)
