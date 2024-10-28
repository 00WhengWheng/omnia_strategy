from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

@dataclass
class WFAConfig:
    train_size: int
    test_size: int
    step_size: int
    min_samples: int
    optimization_metric: str
    anchored: bool  # True for anchored walk forward, False for rolling
    parallel: bool
    n_jobs: int
    parameter_ranges: Dict
    validation_percent: float

@dataclass
class WFAResult:
    start_date: datetime
    end_date: datetime
    train_metrics: Dict
    test_metrics: Dict
    parameters: Dict
    optimization_score: float
    robustness_score: float
    consistency_score: float

class WalkForwardAnalyzer:
    def __init__(self, config: WFAConfig):
        # Initialize the walk-forward analyzer
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage for results
        self.results: List[WFAResult] = []
        self.parameter_history: Dict[str, List] = {}
        self.performance_history: List[Dict] = []
        
        # Metrics tracking
        self.optimization_results = pd.DataFrame()
        self.validation_results = pd.DataFrame()
        self.robustness_metrics = {}
        
    def run_analysis(self,
                    strategy,
                    data: pd.DataFrame,
                    optimizer) -> Dict:
        # Run the walk-forward analysis
        try:
            # Validate data
            if not self._validate_data(data):
                raise ValueError("Invalid data for walk-forward analysis")
                
            # Generate analysis windows
            windows = self._generate_windows(data)
            
            # Run analysis on each window
            if self.config.parallel:
                results = self._run_parallel_analysis(windows, strategy, optimizer)
            else:
                results = self._run_sequential_analysis(windows, strategy, optimizer)
                
            # Analyze results
            analysis = self._analyze_results(results)
            
            # Generate report
            report = self._generate_report(analysis)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Walk-forward analysis failed: {str(e)}")
            raise
            
    def _generate_windows(self, data: pd.DataFrame) -> List[Tuple]:
        # Generate analysis windows
        windows = []
        start_idx = 0
        
        while (start_idx + self.config.train_size + self.config.test_size) <= len(data):
            if self.config.anchored:
                # Anchored walk forward
                train_start = 0
                train_end = start_idx + self.config.train_size
            else:
                # Rolling walk forward
                train_start = start_idx
                train_end = start_idx + self.config.train_size
                
            test_start = train_end
            test_end = test_start + self.config.test_size
            
            windows.append((
                train_start,
                train_end,
                test_start,
                test_end
            ))
            
            start_idx += self.config.step_size
            
        return windows
        
    def _run_parallel_analysis(self,
                             windows: List[Tuple],
                             strategy,
                             optimizer) -> List[WFAResult]:
        # Run parallel analysis 
        with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = []
            
            for window in windows:
                future = executor.submit(
                    self._analyze_window,
                    window,
                    strategy,
                    optimizer
                )
                futures.append(future)
                
            results = [future.result() for future in futures]
            
        return results
        
    def _run_sequential_analysis(self,
                               windows: List[Tuple],
                               strategy,
                               optimizer) -> List[WFAResult]:
        # Run sequential analysis
        results = []
        
        for window in windows:
            result = self._analyze_window(window, strategy, optimizer)
            results.append(result)
            
        return results
        
    def _analyze_window(self,
                       window: Tuple,
                       strategy,
                       optimizer) -> WFAResult:
       # Analyze a single window
        train_start, train_end, test_start, test_end = window
        
        # Split data
        train_data = self.data.iloc[train_start:train_end]
        test_data = self.data.iloc[test_start:test_end]
        
        # Validate data splits
        if not self._validate_split(train_data, test_data):
            raise ValueError("Invalid data split")
            
        # Optimize on training data
        optimization_result = optimizer.optimize(
            strategy,
            train_data,
            self.config.parameter_ranges,
            self.config.optimization_metric
        )
        
        # Validate parameters
        if self.config.validation_percent > 0:
            validation_result = self._validate_parameters(
                strategy,
                train_data,
                optimization_result.parameters,
                self.config.validation_percent
            )
        else:
            validation_result = None
            
        # Test optimized parameters
        test_result = self._test_parameters(
            strategy,
            test_data,
            optimization_result.parameters
        )
        
        return WFAResult(
            start_date=train_data.index[0],
            end_date=test_data.index[-1],
            train_metrics=optimization_result.metrics,
            test_metrics=test_result.metrics,
            parameters=optimization_result.parameters,
            optimization_score=optimization_result.performance,
            robustness_score=self._calculate_robustness_score(
                optimization_result, test_result, validation_result),
            consistency_score=self._calculate_consistency_score(
                optimization_result, test_result)
        )
        
    def _validate_parameters(self,
                           strategy,
                           train_data: pd.DataFrame,
                           parameters: Dict,
                           validation_percent: float) -> Dict:
        # Validate parameters on a subset of training data
        # Split training data for validation
        split_idx = int(len(train_data) * (1 - validation_percent))
        train_subset = train_data.iloc[:split_idx]
        validation_subset = train_data.iloc[split_idx:]
        
        # Test parameters on validation set
        validation_result = strategy.backtest(
            validation_subset,
            parameters
        )
        
        return validation_result
        
    def _calculate_robustness_score(self,
                                  train_result: Dict,
                                  test_result: Dict,
                                  validation_result: Optional[Dict] = None) -> float:
        # Calculate robustness score
        # Compare key metrics
        metrics_to_compare = [
            'sharpe_ratio',
            'sortino_ratio',
            'win_rate',
            'profit_factor'
        ]
        
        scores = []
        
        for metric in metrics_to_compare:
            train_value = train_result.metrics.get(metric, 0)
            test_value = test_result.metrics.get(metric, 0)
            
            if validation_result:
                val_value = validation_result.metrics.get(metric, 0)
                consistency = 1 - (abs(train_value - val_value) + 
                                abs(val_value - test_value)) / (3 * max(abs(train_value), 1e-6))
            else:
                consistency = 1 - abs(train_value - test_value) / max(abs(train_value), 1e-6)
                
            scores.append(consistency)
            
        return np.mean(scores)
        
    def _calculate_consistency_score(self,
                                  train_result: Dict,
                                  test_result: Dict) -> float:
        # Calculate consistency score
        # Compare return distributions
        train_returns = train_result.metrics.get('returns', pd.Series())
        test_returns = test_result.metrics.get('returns', pd.Series())
        
        if len(train_returns) == 0 or len(test_returns) == 0:
            return 0.0
            
        # Calculate distribution similarity
        ks_statistic, _ = stats.ks_2samp(train_returns, test_returns)
        distribution_similarity = 1 - ks_statistic
        
        # Compare performance stability
        performance_stability = self._calculate_performance_stability(
            train_result.metrics,
            test_result.metrics
        )
        
        return (distribution_similarity + performance_stability) / 2
        
    def _analyze_results(self, results: List[WFAResult]) -> Dict:
        # Analyze the walk-forward results
        analysis = {
            'performance': self._analyze_performance(results),
            'parameters': self._analyze_parameters(results),
            'robustness': self._analyze_robustness(results),
            'consistency': self._analyze_consistency(results)
        }
        
        return analysis
        
    def _analyze_performance(self, results: List[WFAResult]) -> Dict:
        # Analyze the performance of the strategy
        train_metrics = pd.DataFrame([r.train_metrics for r in results])
        test_metrics = pd.DataFrame([r.test_metrics for r in results])
        
        return {
            'train': {
                'mean': train_metrics.mean(),
                'std': train_metrics.std(),
                'worst': train_metrics.min(),
                'best': train_metrics.max()
            },
            'test': {
                'mean': test_metrics.mean(),
                'std': test_metrics.std(),
                'worst': test_metrics.min(),
                'best': test_metrics.max()
            },
            'degradation': self._calculate_performance_degradation(
                train_metrics, test_metrics)
        }
        
    def _analyze_parameters(self, results: List[WFAResult]) -> Dict:
        # Analyze the parameters of the strategy
        parameters = pd.DataFrame([r.parameters for r in results])
        
        return {
            'stability': self._calculate_parameter_stability(parameters),
            'ranges': parameters.agg(['min', 'max', 'mean', 'std']).to_dict(),
            'correlations': self._calculate_parameter_correlations(
                parameters, results)
        }
        
    def _analyze_robustness(self, results: List[WFAResult]) -> Dict:
        # Analyze the robustness of the strategy
        robustness_scores = [r.robustness_score for r in results]
        
        return {
            'mean_score': np.mean(robustness_scores),
            'std_score': np.std(robustness_scores),
            'worst_score': np.min(robustness_scores),
            'best_score': np.max(robustness_scores),
            'distribution': pd.Series(robustness_scores).describe().to_dict()
        }
        
    def _analyze_consistency(self, results: List[WFAResult]) -> Dict:
        # Analyze the consistency of the strategy
        consistency_scores = [r.consistency_score for r in results]
        
        return {
            'mean_score': np.mean(consistency_scores),
            'std_score': np.std(consistency_scores),
            'worst_score': np.min(consistency_scores),
            'best_score': np.max(consistency_scores),
            'distribution': pd.Series(consistency_scores).describe().to_dict()
        }
