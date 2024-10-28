from typing import Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import optuna
from scipy.optimize import differential_evolution
from concurrent.futures import ProcessPoolExecutor
import joblib
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class OptimizationResult:
    parameters: Dict
    performance: float
    metrics: Dict
    timestamp: datetime
    metadata: Dict

class StrategyOptimizer:
    def __init__(self, config: Dict):
        """Inizializza il sistema di ottimizzazione"""
        self.config = config
        self.results_dir = Path(config.get('results_dir', 'optimization_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Optimization settings
        self.n_trials = config.get('n_trials', 100)
        self.n_jobs = config.get('n_jobs', -1)
        self.random_seed = config.get('random_seed', 42)
        
        # Validation settings
        self.validation_percentage = config.get('validation_percentage', 0.3)
        self.min_samples = config.get('min_samples', 252)
        
        # Cross-validation settings
        self.n_splits = config.get('n_splits', 5)
        self.walk_forward = config.get('walk_forward', True)
        
    def optimize_strategy(self,
                         strategy,
                         data: pd.DataFrame,
                         parameter_space: Dict,
                         objective_function: str = 'sharpe_ratio',
                         method: str = 'optuna',
                         constraints: Optional[Dict] = None) -> OptimizationResult:
        """Ottimizza i parametri della strategia"""
        
        # Validate data
        if not self._validate_data(data):
            raise ValueError("Invalid data for optimization")
            
        # Prepare data splits
        train_data, val_data = self._prepare_data_splits(data)
        
        # Define objective
        objective = self._create_objective(
            strategy,
            train_data,
            parameter_space,
            objective_function,
            constraints
        )
        
        # Run optimization
        if method == 'optuna':
            best_params = self._optimize_with_optuna(objective, parameter_space)
        elif method == 'differential_evolution':
            best_params = self._optimize_with_de(objective, parameter_space)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
            
        # Validate results
        validation_metrics = self._validate_parameters(
            strategy, best_params, val_data)
        
        # Create results
        results = OptimizationResult(
            parameters=best_params,
            performance=validation_metrics[objective_function],
            metrics=validation_metrics,
            timestamp=datetime.now(),
            metadata={'method': method, 'splits': self.n_splits}
        )
        
        # Save results
        self._save_results(results)
        
        return results
        
    def walk_forward_optimization(self,
                                strategy,
                                data: pd.DataFrame,
                                parameter_space: Dict,
                                window_size: int,
                                step_size: int,
                                objective_function: str = 'sharpe_ratio') -> List[OptimizationResult]:
        """Esegue ottimizzazione walk-forward"""
        results = []
        
        # Create windows
        windows = self._create_walk_forward_windows(
            data, window_size, step_size)
            
        # Optimize each window
        for train_data, test_data in windows:
            window_result = self.optimize_strategy(
                strategy,
                train_data,
                parameter_space,
                objective_function
            )
            
            # Validate on test data
            test_metrics = self._validate_parameters(
                strategy, window_result.parameters, test_data)
                
            window_result.metrics['test'] = test_metrics
            results.append(window_result)
            
        return results
        
    def _create_objective(self,
                         strategy,
                         data: pd.DataFrame,
                         parameter_space: Dict,
                         objective_function: str,
                         constraints: Optional[Dict]) -> Callable:
        """Crea la funzione obiettivo per l'ottimizzazione"""
        
        def objective(trial):
            # Generate parameters
            params = {}
            for param, space in parameter_space.items():
                if space['type'] == 'int':
                    params[param] = trial.suggest_int(
                        param, space['low'], space['high'])
                elif space['type'] == 'float':
                    params[param] = trial.suggest_float(
                        param, space['low'], space['high'])
                elif space['type'] == 'categorical':
                    params[param] = trial.suggest_categorical(
                        param, space['choices'])
                    
            # Apply constraints if any
            if constraints and not self._check_constraints(params, constraints):
                return float('-inf')
                
            # Run cross-validation
            cv_metrics = self._cross_validate(strategy, params, data)
            
            # Return objective metric
            return cv_metrics[objective_function]
            
        return objective
        
    def _cross_validate(self,
                       strategy,
                       parameters: Dict,
                       data: pd.DataFrame) -> Dict:
        """Esegue cross-validation delle performance"""
        if self.walk_forward:
            splits = self._create_walk_forward_splits(data)
        else:
            splits = self._create_time_series_splits(data)
            
        metrics = []
        
        # Run strategy on each split
        for train_idx, test_idx in splits:
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Train strategy
            strategy.set_parameters(parameters)
            strategy.train(train_data)
            
            # Test strategy
            split_metrics = strategy.evaluate(test_data)
            metrics.append(split_metrics)
            
        # Aggregate metrics
        return self._aggregate_metrics(metrics)
        
    def _optimize_with_optuna(self,
                            objective: Callable,
                            parameter_space: Dict) -> Dict:
        """Ottimizzazione usando Optuna"""
        study = optuna.create_study(direction='maximize')
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        return study.best_params
        
    def _optimize_with_de(self,
                         objective: Callable,
                         parameter_space: Dict) -> Dict:
        """Ottimizzazione usando Differential Evolution"""
        bounds = []
        param_names = []
        
        for param, space in parameter_space.items():
            if space['type'] in ['int', 'float']:
                bounds.append((space['low'], space['high']))
                param_names.append(param)
                
        def de_objective(x):
            params = dict(zip(param_names, x))
            return -objective(params)  # Negative because DE minimizes
            
        result = differential_evolution(
            de_objective,
            bounds,
            seed=self.random_seed,
            workers=self.n_jobs
        )
        
        return dict(zip(param_names, result.x))
        
    def _validate_parameters(self,
                           strategy,
                           parameters: Dict,
                           data: pd.DataFrame) -> Dict:
        """Valida i parametri ottimizzati"""
        strategy.set_parameters(parameters)
        return strategy.evaluate(data)
        
    def _create_walk_forward_windows(self,
                                   data: pd.DataFrame,
                                   window_size: int,
                                   step_size: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Crea finestre per walk-forward optimization"""
        windows = []
        start_idx = 0
        
        while start_idx + window_size < len(data):
            end_idx = start_idx + window_size
            test_idx = min(end_idx + step_size, len(data))
            
            train_data = data.iloc[start_idx:end_idx]
            test_data = data.iloc[end_idx:test_idx]
            
            windows.append((train_data, test_data))
            start_idx += step_size
            
        return windows
        
    def _create_walk_forward_splits(self,
                                  data: pd.DataFrame) -> List[Tuple[np.array, np.array]]:
        """Crea split per walk-forward cross-validation"""
        splits = []
        split_size = len(data) // self.n_splits
        
        for i in range(self.n_splits - 1):
            train_end = (i + 1) * split_size
            test_end = train_end + split_size
            
            train_idx = np.arange(i * split_size, train_end)
            test_idx = np.arange(train_end, test_end)
            
            splits.append((train_idx, test_idx))
            
        return splits
        
    def _aggregate_metrics(self, metrics: List[Dict]) -> Dict:
        """Aggrega metriche da multiple valutazioni"""
        aggregated = {}
        
        for key in metrics[0].keys():
            values = [m[key] for m in metrics]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
        return aggregated
        
    def _check_constraints(self,
                         parameters: Dict,
                         constraints: Dict) -> bool:
        """Verifica i vincoli sui parametri"""
        for constraint, func in constraints.items():
            if not func(parameters):
                return False
        return True
        
    def _save_results(self, results: OptimizationResult):
        """Salva i risultati dell'ottimizzazione"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimization_results_{timestamp}.pkl"
        filepath = self.results_dir / filename
        
        joblib.dump(results, filepath)
        
    def get_optimization_history(self) -> pd.DataFrame:
        """Recupera lo storico delle ottimizzazioni"""
        results = []
        
        for filepath in self.results_dir.glob('optimization_results_*.pkl'):
            try:
                result = joblib.load(filepath)
                results.append({
                    'timestamp': result.timestamp,
                    'performance': result.performance,
                    'parameters': result.parameters,
                    'method': result.metadata['method']
                })
            except Exception as e:
                self.logger.error(f"Error loading results from {filepath}: {str(e)}")
                
        return pd.DataFrame(results)
