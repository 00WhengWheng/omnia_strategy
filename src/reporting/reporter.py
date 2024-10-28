from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import jinja2
import pdfkit
import json
import asyncio
from dataclasses import dataclass
import logging

@dataclass
class ReportConfig:
    report_type: str
    frequency: str
    format: str
    recipients: List[str]
    template: str
    include_charts: bool
    include_metrics: bool
    include_alerts: bool
    custom_metrics: Dict

class ReportingSystem:
    def __init__(self, config: Dict):
        """Inizializza il sistema di reporting"""
        self.config = config
        self.report_dir = Path(config.get('report_dir', 'reports'))
        self.template_dir = Path(config.get('template_dir', 'templates'))
        
        # Create directories
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize template engine
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir)
        )
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Schedule configurations
        self.schedules = {
            'daily': self._generate_daily_reports,
            'weekly': self._generate_weekly_reports,
            'monthly': self._generate_monthly_reports,
            'custom': self._generate_custom_reports
        }
        
        # Report types
        self.report_types = {
            'performance': self._generate_performance_report,
            'risk': self._generate_risk_report,
            'trading': self._generate_trading_report,
            'system': self._generate_system_report,
            'custom': self._generate_custom_report
        }
        
    async def start_scheduler(self):
        """Avvia lo scheduler dei report"""
        while True:
            current_time = datetime.now()
            
            # Check daily reports
            if current_time.hour == self.config.get('daily_report_hour', 0):
                await self.schedules['daily']()
                
            # Check weekly reports
            if current_time.weekday() == 6 and current_time.hour == 0:
                await self.schedules['weekly']()
                
            # Check monthly reports
            if current_time.day == 1 and current_time.hour == 0:
                await self.schedules['monthly']()
                
            await asyncio.sleep(3600)  # Check every hour
            
    async def generate_report(self, report_config: ReportConfig) -> str:
        """Genera un report specifico"""
        try:
            # Get report generator
            report_generator = self.report_types.get(report_config.report_type)
            if not report_generator:
                raise ValueError(f"Invalid report type: {report_config.report_type}")
                
            # Generate report data
            report_data = await report_generator(report_config)
            
            # Generate report using template
            report_html = self._render_template(report_config.template, report_data)
            
            # Convert to requested format
            report_path = self._save_report(
                report_html,
                report_config.report_type,
                report_config.format
            )
            
            # Send report if recipients specified
            if report_config.recipients:
                await self._send_report(report_path, report_config.recipients)
                
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
            
    async def _generate_performance_report(self, config: ReportConfig) -> Dict:
        """Genera report di performance"""
        # Get performance data
        performance_data = await self._get_performance_data()
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(performance_data)
        
        # Generate charts if requested
        charts = {}
        if config.include_charts:
            charts = await self._generate_performance_charts(performance_data)
            
        return {
            'type': 'performance',
            'data': performance_data,
            'metrics': metrics,
            'charts': charts,
            'timestamp': datetime.now()
        }
        
    async def _generate_risk_report(self, config: ReportConfig) -> Dict:
        """Genera report di rischio"""
        # Get risk data
        risk_data = await self._get_risk_data()
        
        # Calculate risk metrics
        metrics = self._calculate_risk_metrics(risk_data)
        
        # Generate charts if requested
        charts = {}
        if config.include_charts:
            charts = await self._generate_risk_charts(risk_data)
            
        return {
            'type': 'risk',
            'data': risk_data,
            'metrics': metrics,
            'charts': charts,
            'timestamp': datetime.now()
        }
        
    async def _generate_trading_report(self, config: ReportConfig) -> Dict:
        """Genera report di trading"""
        # Get trading data
        trading_data = await self._get_trading_data()
        
        # Calculate trading metrics
        metrics = self._calculate_trading_metrics(trading_data)
        
        # Generate charts if requested
        charts = {}
        if config.include_charts:
            charts = await self._generate_trading_charts(trading_data)
            
        return {
            'type': 'trading',
            'data': trading_data,
            'metrics': metrics,
            'charts': charts,
            'timestamp': datetime.now()
        }
        
    def _generate_performance_charts(self, data: pd.DataFrame) -> Dict:
        """Genera grafici di performance"""
        charts = {}
        
        # Equity curve
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=data.index,
            y=data['equity'],
            name='Equity'
        ))
        charts['equity_curve'] = fig_equity
        
        # Drawdown chart
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=data.index,
            y=data['drawdown'],
            name='Drawdown',
            fill='tozeroy'
        ))
        charts['drawdown'] = fig_dd
        
        # Monthly returns heatmap
        monthly_returns = self._calculate_monthly_returns(data)
        fig_monthly = px.imshow(
            monthly_returns,
            labels=dict(x="Month", y="Year", color="Return %")
        )
        charts['monthly_returns'] = fig_monthly
        
        return charts
        
    def _generate_risk_charts(self, data: pd.DataFrame) -> Dict:
        """Genera grafici di rischio"""
        charts = {}
        
        # VaR chart
        fig_var = go.Figure()
        fig_var.add_trace(go.Scatter(
            x=data.index,
            y=data['var'],
            name='Value at Risk'
        ))
        charts['var'] = fig_var
        
        # Risk allocation pie chart
        fig_alloc = go.Figure(data=[go.Pie(
            labels=data['category'],
            values=data['risk_allocation']
        )])
        charts['risk_allocation'] = fig_alloc
        
        return charts
        
    def _generate_trading_charts(self, data: pd.DataFrame) -> Dict:
        """Genera grafici di trading"""
        charts = {}
        
        # Win/Loss ratio chart
        fig_wl = go.Figure()
        fig_wl.add_trace(go.Bar(
            x=['Wins', 'Losses'],
            y=[data['wins'].sum(), data['losses'].sum()]
        ))
        charts['win_loss'] = fig_wl
        
        # Trade duration histogram
        fig_duration = px.histogram(
            data,
            x='duration',
            title='Trade Duration Distribution'
        )
        charts['duration'] = fig_duration
        
        return charts
        
    def _render_template(self, template_name: str, data: Dict) -> str:
        """Renderizza il template con i dati"""
        template = self.template_env.get_template(template_name)
        return template.render(**data)
        
    def _save_report(self, content: str, report_type: str, format: str) -> str:
        """Salva il report nel formato richiesto"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{report_type}_{timestamp}.{format}"
        filepath = self.report_dir / filename
        
        if format == 'html':
            with open(filepath, 'w') as f:
                f.write(content)
        elif format == 'pdf':
            pdfkit.from_string(content, str(filepath))
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(content, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return str(filepath)
        
    async def _send_report(self, report_path: str, recipients: List[str]):
        """Invia il report ai destinatari"""
        # Implement email sending logic here
        pass
        
    def _calculate_monthly_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcola i rendimenti mensili"""
        returns = data['returns']
        monthly_returns = returns.groupby([
            returns.index.year,
            returns.index.month
        ]).sum().unstack()
        return monthly_returns
        
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calcola le metriche di performance"""
        returns = data['returns']
        
        return {
            'total_return': returns.sum(),
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252),
            'max_drawdown': data['drawdown'].min(),
            'win_rate': len(returns[returns > 0]) / len(returns),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum())
        }
        
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """Calcola le metriche di rischio"""
        returns = data['returns']
        
        return {
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
            'beta': self._calculate_beta(returns),
            'correlation': self._calculate_correlation_matrix(data),
            'tail_risk': self._calculate_tail_risk(returns)
        }
        
    def _calculate_trading_metrics(self, data: pd.DataFrame) -> Dict:
        """Calcola le metriche di trading"""
        return {
            'total_trades': len(data),
            'winning_trades': len(data[data['pnl'] > 0]),
            'losing_trades': len(data[data['pnl'] < 0]),
            'average_win': data[data['pnl'] > 0]['pnl'].mean(),
            'average_loss': data[data['pnl'] < 0]['pnl'].mean(),
            'largest_win': data['pnl'].max(),
            'largest_loss': data['pnl'].min(),
            'average_duration': data['duration'].mean()
        }
