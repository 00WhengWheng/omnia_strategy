import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.interpolate import interp1d, splrep, splev

def classify_problem_and_select_metrics(target_column: str, prediction_type: str, is_multiclass: bool = False) -> Tuple[str, List[str]]:
    """
    Classifica il tipo di problema e seleziona le metriche appropriate.
    
    Args:
        target_column (str): Nome della colonna target.
        prediction_type (str): Tipo di predizione - 'regression' o 'classification'.
        is_multiclass (bool, optional): Indica se il problema è di classificazione multiclasse. Default è False.
        
    Returns:
        Tuple[str, List[str]]: Una tupla contenente il tipo di problema e una lista di metriche appropriate.
    """
    problem_type = ''
    metrics = []
    
    if prediction_type == 'regression':
        if 'return' in target_column.lower() or 'price' in target_column.lower():
            problem_type = 'returns_prediction'
            metrics = ['MSE', 'RMSE', 'MAE', 'R2']
        else:
            problem_type = 'general_regression'
            metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    
    elif prediction_type == 'classification':
        if is_multiclass:
            problem_type = 'multiclass_classification'
            metrics = ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro', 'AUC_macro']
        else:
            if 'direction' in target_column.lower():
                problem_type = 'direction_prediction'
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            else:
                problem_type = 'general_classification'
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    
    else:
        raise ValueError(f"Invalid prediction_type: {prediction_type}. Must be 'regression' or 'classification'.")
    
    return problem_type, metrics

def evaluate_model(model, X_test, y_test, problem_type: str, metrics: List[str]) -> Dict[str, float]:
    """
    Valuta le prestazioni del modello utilizzando le metriche appropriate.
    
    Args:
        model: Modello addestrato.
        X_test (array-like): Dati di test delle features.
        y_test (array-like): Dati di test della variabile target.
        problem_type (str): Tipo di problema.
        metrics (List[str]): Lista di metriche da calcolare.
        
    Returns:
        Dict[str, float]: Un dizionario contenente i valori delle metriche calcolate.
    """
    y_pred = model.predict(X_test)
    
    metric_values = {}
    for metric in metrics:
        if metric == 'MSE':
            metric_values[metric] = mean_squared_error(y_test, y_pred)
        elif metric == 'RMSE':
            metric_values[metric] = np.sqrt(mean_squared_error(y_test, y_pred))
        elif metric == 'MAE':
            metric_values[metric] = mean_absolute_error(y_test, y_pred)
        elif metric == 'R2':
            metric_values[metric] = r2_score(y_test, y_pred)
        elif metric == 'Accuracy':
            metric_values[metric] = accuracy_score(y_test, y_pred)
        elif metric.startswith('Precision'):
            metric_values[metric] = precision_score(y_test, y_pred, average=metric.split('_')[1])
        elif metric.startswith('Recall'):
            metric_values[metric] = recall_score(y_test, y_pred, average=metric.split('_')[1])
        elif metric.startswith('F1'):
            metric_values[metric] = f1_score(y_test, y_pred, average=metric.split('_')[1])
        elif metric.startswith('AUC'):
            metric_values[metric] = roc_auc_score(y_test, y_pred, average=metric.split('_')[1], multi_class='ovr' if problem_type == 'multiclass_classification' else None)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    return metric_values

class TradingPreprocessor:
    def __init__(self,
                 price_columns: List[str],
                 volume_columns: List[str] = None,
                 categorical_columns: List[str] = None,
                 technical_indicators: bool = True,
                 outlier_std_threshold: float = 3.0,
                 rolling_window: int = 20):
        self.price_columns = price_columns
        self.volume_columns = volume_columns or []
        self.categorical_columns = categorical_columns or []
        self.technical_indicators = technical_indicators
        self.outlier_std_threshold = outlier_std_threshold
        self.rolling_window = rolling_window
        
        self.price_scaler = RobustScaler()
        self.volume_scaler = RobustScaler()
        self.other_scaler = MinMaxScaler()
        
        self.stats = {}
        
    def classify_data_and_handle_missing(self, df: pd.DataFrame):
        def classify_data(series: pd.Series) -> str:
            if pd.api.types.is_numeric_dtype(series):
                if series.name in self.price_columns:
                    return 'price'
                elif series.name in self.volume_columns:
                    return 'volume'
                else:
                    return 'numeric'
            elif pd.api.types.is_datetime64_dtype(series):
                return 'datetime'
            else:
                return 'categorical'
        
        data_types = df.apply(classify_data)
        
        def handle_missing(series: pd.Series) -> pd.Series:
            if series.name in self.price_columns:
                return series.interpolate(method='time')
            elif series.name in self.volume_columns:
                return series.fillna(0)
            elif data_types[series.name] == 'numeric':
                return series.interpolate(method='linear')
            elif data_types[series.name] == 'datetime':
                return series.fillna(method='ffill')
            else:
                return series.fillna(series.mode()[0])
        
        df_filled = df.apply(handle_missing)
        return df_filled
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (mantieni l'implementazione originale) ...
        return df_copy
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (mantieni l'implementazione originale) ...
        return df_copy
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (mantieni l'implementazione originale) ...
        return df_copy
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        # ... (mantieni l'implementazione originale) ...
        return df_copy
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = self.classify_data_and_handle_missing(df)
        df_clean = self._handle_outliers(df_clean)
        df_clean = self._add_technical_indicators(df_clean)
        df_clean = self._add_temporal_features(df_clean)
        df_scaled = self._scale_features(df_clean, fit=True)
        return df_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = self.classify_data_and_handle_missing(df)
        df_clean = self._handle_outliers(df_clean)
        df_clean = self._add_technical_indicators(df_clean)
        df_clean = self._add_temporal_features(df_clean)
        df_scaled = self._scale_features(df_clean, fit=False)
        return df_scaled
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        return self.transform(df.head(1)).columns.tolist()
    
'''
workflow per il backtesting:

# Classificazione del problema e selezione delle metriche
problem_type, metrics = classify_problem_and_select_metrics(target_column='daily_returns', prediction_type='regression')

# Valutazione del modello
metric_values = evaluate_model(model, X_test, y_test, problem_type, metrics)

# Stampa dei risultati
for metric, value in metric_values.items():
    print(f"{metric}: {value}")
'''