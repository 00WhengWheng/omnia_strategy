import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
import ta

class TradingPreprocessor:
    def __init__(self,
                 price_columns: List[str],
                 volume_columns: List[str] = None,
                 categorical_columns: List[str] = None,
                 technical_indicators: bool = True,
                 outlier_threshold: float = 3.0,
                 missing_data_method: str = 'iterative',
                 feature_selection_method: str = 'k_best',
                 n_features: int = 20,
                 pca_n_components: Union[int, float] = 0.95):
        self.price_columns = price_columns
        self.volume_columns = volume_columns or []
        self.categorical_columns = categorical_columns or []
        self.technical_indicators = technical_indicators
        self.outlier_threshold = outlier_threshold
        self.missing_data_method = missing_data_method
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.pca_n_components = pca_n_components
        
        self.price_scaler = RobustScaler()
        self.volume_scaler = RobustScaler()
        self.other_scaler = MinMaxScaler()
        
        self.imputers = {
            'knn': KNNImputer(n_neighbors=5),
            'iterative': IterativeImputer(random_state=42)
        }
        
        self.feature_selectors = {
            'k_best_regression': SelectKBest(f_regression, k=self.n_features),
            'k_best_classification': SelectKBest(f_classif, k=self.n_features)
        }
        
        self.pca = PCA(n_components=self.pca_n_components)
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementa la gestione dei dati mancanti utilizzando il metodo specificato
        if self.missing_data_method in self.imputers:
            imputer = self.imputers[self.missing_data_method]
            df_imputed = imputer.fit_transform(df)
            df_imputed = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
            return df_imputed
        else:
            raise ValueError(f"Unsupported missing data method: {self.missing_data_method}")
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementa la gestione degli outlier utilizzando la soglia specificata
        df_outlier_handled = df.copy()
        for col in df_outlier_handled.columns:
            if col in self.price_columns or col in self.volume_columns:
                mean = df_outlier_handled[col].mean()
                std = df_outlier_handled[col].std()
                upper_bound = mean + self.outlier_threshold * std
                lower_bound = mean - self.outlier_threshold * std
                df_outlier_handled[col] = df_outlier_handled[col].clip(lower=lower_bound, upper=upper_bound)
        return df_outlier_handled
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementa l'aggiunta di indicatori tecnici utilizzando la libreria TA-Lib o una libreria simile
        if not self.technical_indicators:
            return df
        
        df_with_indicators = df.copy()
        
        if len(self.price_columns) >= 4:
            high = df_with_indicators[self.price_columns[1]]
            low = df_with_indicators[self.price_columns[2]]
            close = df_with_indicators[self.price_columns[3]]
            
            df_with_indicators['RSI'] = ta.momentum.RSIIndicator(close).rsi()
            df_with_indicators['MACD'] = ta.trend.MACD(close).macd()
            df_with_indicators['BB_upper'], df_with_indicators['BB_middle'], df_with_indicators['BB_lower'] = ta.volatility.BollingerBands(close).bollinger_hband(), ta.volatility.BollingerBands(close).bollinger_mavg(), ta.volatility.BollingerBands(close).bollinger_lband()
            # Aggiungi altri indicatori tecnici as needed
        
        return df_with_indicators
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementa la codifica delle variabili categoriche utilizzando la codifica one-hot o altri metodi
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col)
        
        return df_encoded
    
    def select_features(self, df: pd.DataFrame, target_column: str, prediction_type: str) -> pd.DataFrame:
        # Implementa la selezione delle features utilizzando il metodo specificato
        if self.feature_selection_method.startswith('k_best'):
            if prediction_type == 'regression':
                selector = self.feature_selectors['k_best_regression']
            else:
                selector = self.feature_selectors['k_best_classification']
            
            selector.fit(df.drop(columns=[target_column]), df[target_column])
            selected_features = df.columns[selector.get_support()].tolist()
            return df[selected_features + [target_column]]
        else:
            raise ValueError(f"Unsupported feature selection method: {self.feature_selection_method}")
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementa la riduzione della dimensionalità utilizzando PCA
        pca_transformed = self.pca.fit_transform(df)
        columns = [f'PC{i+1}' for i in range(pca_transformed.shape[1])]
        df_pca = pd.DataFrame(pca_transformed, columns=columns, index=df.index)
        return df_pca
    
    def check_stationarity(self, df: pd.DataFrame) -> Dict[str, bool]:
        # Controlla la stazionarietà delle serie temporali utilizzando il test di Dickey-Fuller aumentato
        stationarity = {}
        for col in df.columns:
            result = adfuller(df[col])
            stationarity[col] = result[1] < 0.05  # p-value < 0.05 indica stazionarietà
        return stationarity
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str, prediction_type: str) -> pd.DataFrame:
        # Pipeline completa di preprocessing dei dati
        df_preprocessed = self.handle_missing_data(df)
        df_preprocessed = self.handle_outliers(df_preprocessed)
        df_preprocessed = self.add_technical_indicators(df_preprocessed)
        df_preprocessed = self.encode_categorical_features(df_preprocessed)
        df_preprocessed = self.select_features(df_preprocessed, target_column, prediction_type)
        df_preprocessed = self.reduce_dimensionality(df_preprocessed.drop(columns=[target_column]))
        df_preprocessed[target_column] = df[target_column]
        stationarity = self.check_stationarity(df_preprocessed.drop(columns=[target_column]))
        return df_preprocessed, stationarity