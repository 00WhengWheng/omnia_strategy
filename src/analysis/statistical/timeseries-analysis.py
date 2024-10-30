from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
import logging
from ..base import BaseAnalyzer

@dataclass
class TimeSeriesState:
    timestamp: datetime
    stationarity: Dict[str, any]      # Stationarity tests and transformations
    decomposition: Dict[str, pd.Series] # Trend, seasonal, residual components
    patterns: Dict[str, any]          # Detected patterns and cycles
    forecast: Dict[str, any]          # Forecasting results
    statistics: Dict[str, float]      # Statistical measures
    correlations: Dict[str, float]    # Correlation analysis
    anomalies: List[Dict]            # Detected anomalies
    confidence: float
    metadata: Dict

class TimeSeriesAnalyzer(BaseAnalyzer):
    """Time Series Analysis Component"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.lookback_window = self.config.get('lookback_window', 252)
        self.cycle_periods = self.config.get('cycle_periods', [5, 10, 21, 63, 252])
        self.seasonality_periods = self.config.get('seasonality_periods', [5, 21, 63])
        self.forecast_horizon = self.config.get('forecast_horizon', 5)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
        # Statistical parameters
        self.anomaly_threshold = self.config.get('anomaly_threshold', 2.5)
        self.min_pattern_significance = self.config.get('min_pattern_significance', 0.05)
        
        # Initialize components
        self.scaler = StandardScaler()
        
        # Analysis cache
        self.analysis_history = []

    def analyze(self, data: pd.DataFrame) -> TimeSeriesState:
        """
        Perform comprehensive time series analysis
        
        Parameters:
        - data: DataFrame with time series data
        
        Returns:
        - TimeSeriesState object containing analysis results
        """
        try:
            # Validate and prepare data
            clean_data = self._prepare_data(data)
            
            # Analyze stationarity
            stationarity = self._analyze_stationarity(clean_data)
            
            # Decompose time series
            decomposition = self._decompose_series(clean_data)
            
            # Detect patterns
            patterns = self._detect_patterns(clean_data, decomposition)
            
            # Generate forecasts
            forecast = self._generate_forecasts(clean_data, decomposition)
            
            # Calculate statistics
            statistics = self._calculate_statistics(clean_data)
            
            # Analyze correlations
            correlations = self._analyze_correlations(clean_data)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(clean_data, statistics)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                stationarity, decomposition, forecast, statistics)
            
            # Generate metadata
            metadata = self._generate_metadata(
                clean_data, stationarity, decomposition, forecast)
            
            state = TimeSeriesState(
                timestamp=datetime.now(),
                stationarity=stationarity,
                decomposition=decomposition,
                patterns=patterns,
                forecast=forecast,
                statistics=statistics,
                correlations=correlations,
                anomalies=anomalies,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update history
            self._update_history(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Time series analysis failed: {e}")
            raise

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate time series data"""
        # Check for missing values
        if data.isnull().any().any():
            data = self._handle_missing_values(data)
            
        # Check for outliers
        data = self._handle_outliers(data)
        
        # Calculate returns if dealing with prices
        if 'close' in data.columns:
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close']/data['close'].shift(1))
            
        return data

    def _analyze_stationarity(self, data: pd.DataFrame) -> Dict:
        """Analyze time series stationarity"""
        # Augmented Dickey-Fuller test
        adf_test = adfuller(data['close'])
        
        # KPSS test
        kpss_test = kpss(data['close'])
        
        # Check for unit root
        unit_root = self._check_unit_root(data)
        
        # Determine transformation needed
        transformation = self._determine_transformation(data)
        
        return {
            'adf_test': {
                'statistic': adf_test[0],
                'p_value': adf_test[1],
                'critical_values': adf_test[4]
            },
            'kpss_test': {
                'statistic': kpss_test[0],
                'p_value': kpss_test[1],
                'critical_values': kpss_test[3]
            },
            'unit_root': unit_root,
            'transformation': transformation,
            'is_stationary': adf_test[1] < 0.05
        }

    def _decompose_series(self, data: pd.DataFrame) -> Dict:
        """Decompose time series into components"""
        # Seasonal decomposition
        decomp = seasonal_decompose(
            data['close'],
            period=self.seasonality_periods[0],
            extrapolate_trend='freq'
        )
        
        # Cycle analysis
        cycles = self._analyze_cycles(data)
        
        # Trend analysis
        trend_analysis = self._analyze_trend(decomp.trend)
        
        # Seasonality analysis
        seasonality = self._analyze_seasonality(decomp.seasonal)
        
        return {
            'trend': decomp.trend,
            'seasonal': decomp.seasonal,
            'residual': decomp.resid,
            'cycles': cycles,
            'trend_analysis': trend_analysis,
            'seasonality_analysis': seasonality
        }

    def _detect_patterns(self, data: pd.DataFrame,
                        decomposition: Dict) -> Dict:
        """Detect patterns in time series"""
        # Autocorrelation analysis
        autocorr = self._analyze_autocorrelation(data)
        
        # Regime detection
        regimes = self._detect_regimes(data)
        
        # Pattern recognition
        patterns = self._recognize_patterns(data)
        
        # Change point detection
        change_points = self._detect_change_points(data)
        
        return {
            'autocorrelation': autocorr,
            'regimes': regimes,
            'patterns': patterns,
            'change_points': change_points,
            'significance': self._calculate_pattern_significance(patterns)
        }

    def _generate_forecasts(self, data: pd.DataFrame,
                          decomposition: Dict) -> Dict:
        """Generate time series forecasts"""
        # ARIMA forecast
        arima_forecast = self._generate_arima_forecast(data)
        
        # Statistical forecast
        stat_forecast = self._generate_statistical_forecast(data)
        
        # Component-based forecast
        comp_forecast = self._generate_component_forecast(decomposition)
        
        # Combine forecasts
        combined_forecast = self._combine_forecasts(
            arima_forecast, stat_forecast, comp_forecast)
        
        return {
            'arima': arima_forecast,
            'statistical': stat_forecast,
            'component': comp_forecast,
            'combined': combined_forecast,
            'confidence_intervals': self._calculate_forecast_intervals(combined_forecast)
        }

    def _analyze_cycles(self, data: pd.DataFrame) -> Dict:
        """Analyze cyclic patterns in time series"""
        cycles = {}
        
        for period in self.cycle_periods:
            # Fourier analysis
            fourier = self._fourier_analysis(data, period)
            
            # Cycle detection
            detected_cycles = self._detect_cycles(data, period)
            
            # Period significance
            significance = self._calculate_cycle_significance(detected_cycles)
            
            cycles[period] = {
                'fourier': fourier,
                'detected': detected_cycles,
                'significance': significance
            }
            
        return cycles

    def _analyze_autocorrelation(self, data: pd.DataFrame) -> Dict:
        """Analyze autocorrelation in time series"""
        # Calculate ACF and PACF
        acf_values = acf(data['returns'].dropna(), nlags=40)
        pacf_values = pacf(data['returns'].dropna(), nlags=40)
        
        # Test for serial correlation
        serial_corr = self._test_serial_correlation(data)
        
        # Long memory analysis
        long_memory = self._analyze_long_memory(data)
        
        return {
            'acf': acf_values,
            'pacf': pacf_values,
            'serial_correlation': serial_corr,
            'long_memory': long_memory,
            'significance': self._calculate_correlation_significance(acf_values)
        }

    def _detect_regimes(self, data: pd.DataFrame) -> Dict:
        """Detect regime changes in time series"""
        # Markov regime detection
        markov = self._markov_regime_detection(data)
        
        # Threshold regime detection
        threshold = self._threshold_regime_detection(data)
        
        # Volatility regime detection
        volatility = self._volatility_regime_detection(data)
        
        return {
            'markov': markov,
            'threshold': threshold,
            'volatility': volatility,
            'transitions': self._analyze_regime_transitions(markov)
        }

    def _generate_arima_forecast(self, data: pd.DataFrame) -> Dict:
        """Generate ARIMA forecast"""
        try:
            # Determine optimal ARIMA parameters
            params = self._determine_arima_parameters(data)
            
            # Fit ARIMA model
            model = ARIMA(
                data['close'],
                order=(params['p'], params['d'], params['q'])
            ).fit()
            
            # Generate forecast
            forecast = model.forecast(steps=self.forecast_horizon)
            
            return {
                'forecast': forecast,
                'parameters': params,
                'aic': model.aic,
                'residuals': model.resid
            }
            
        except Exception as e:
            self.logger.error(f"ARIMA forecast failed: {e}")
            return None

    def _calculate_confidence(self, stationarity: Dict,
                            decomposition: Dict,
                            forecast: Dict,
                            statistics: Dict) -> float:
        """Calculate confidence in time series analysis"""
        # Stationarity confidence
        stationarity_conf = self._calculate_stationarity_confidence(stationarity)
        
        # Forecast confidence
        forecast_conf = self._calculate_forecast_confidence(forecast)
        
        # Pattern confidence
        pattern_conf = self._calculate_pattern_confidence(decomposition)
        
        # Statistical confidence
        stat_conf = self._calculate_statistical_confidence(statistics)
        
        # Combine confidences
        weights = {
            'stationarity': 0.25,
            'forecast': 0.30,
            'pattern': 0.25,
            'statistical': 0.20
        }
        
        confidence = sum(
            conf * weights[name]
            for name, conf in [
                ('stationarity', stationarity_conf),
                ('forecast', forecast_conf),
                ('pattern', pattern_conf),
                ('statistical', stat_conf)
            ]
        )
        
        return np.clip(confidence, 0, 1)

    @property
    def required_columns(self) -> List[str]:
        """Required columns for time series analysis"""
        return ['close', 'volume', 'high', 'low']

    def get_analysis_summary(self) -> Dict:
        """Get summary of current time series state"""
        if not self.analysis_history:
            return {}
            
        latest = self.analysis_history[-1]
        return {
            'timestamp': latest.timestamp,
            'is_stationary': latest.stationarity['is_stationary'],
            'current_regime': latest.patterns['regimes']['current'],
            'forecast': latest.forecast['combined'],
            'anomalies': len(latest.anomalies),
            'confidence': latest.confidence
        }
