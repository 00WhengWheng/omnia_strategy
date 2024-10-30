import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import ta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TradingPreprocessor:
    def __init__(self,
                 price_columns: List[str],
                 volume_columns: List[str] = None,
                 categorical_columns: List[str] = None,
                 technical_indicators: bool = True,
                 outlier_std_threshold: float = 3.0,
                 rolling_window: int = 20):
        """
        Args:
            price_columns: Colonne contenenti prezzi
            volume_columns: Colonne contenenti volumi
            categorical_columns: Colonne categoriche
            technical_indicators: Se calcolare indicatori tecnici
            outlier_std_threshold: Soglia per detection outliers
            rolling_window: Finestra per calcoli rolling
        """
        self.price_columns = price_columns
        self.volume_columns = volume_columns or []
        self.categorical_columns = categorical_columns or []
        self.technical_indicators = technical_indicators
        self.outlier_std_threshold = outlier_std_threshold
        self.rolling_window = rolling_window
        
        # Inizializza scalers
        self.price_scaler = RobustScaler()
        self.volume_scaler = RobustScaler()
        self.other_scaler = MinMaxScaler()
        
        # Imputer per missing values
        self.imputer = KNNImputer(n_neighbors=5)
        
        # Salva statistiche
        self.stats = {}
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gestisce i missing values con diverse strategie"""
        df_copy = df.copy()
        
        # Forward fill per prezzi (assume ultimo prezzo conosciuto)
        for col in self.price_columns:
            df_copy[col].fillna(method='ffill', inplace=True)
            df_copy[col].fillna(method='bfill', inplace=True)
        
        # Media mobile per volumi
        for col in self.volume_columns:
            df_copy[col] = df_copy[col].fillna(
                df_copy[col].rolling(window=self.rolling_window, min_periods=1).mean()
            )
        
        # KNN imputer per altre features numeriche
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        other_cols = [col for col in numeric_cols 
                     if col not in self.price_columns + self.volume_columns]
        
        if other_cols:
            df_copy[other_cols] = pd.DataFrame(
                self.imputer.fit_transform(df_copy[other_cols]),
                columns=other_cols,
                index=df_copy.index
            )
        
        # Moda per categoriche
        for col in self.categorical_columns:
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        
        return df_copy
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gestisce gli outliers con diverse strategie"""
        df_copy = df.copy()
        
        def replace_outliers(series):
            """Sostituisce outliers con valori ai limiti"""
            mean = series.mean()
            std = series.std()
            threshold = self.outlier_std_threshold * std
            
            upper_limit = mean + threshold
            lower_limit = mean - threshold
            
            series[series > upper_limit] = upper_limit
            series[series < lower_limit] = lower_limit
            
            return series
        
        # Gestione outliers per prezzi (più conservativa)
        for col in self.price_columns:
            rolling_median = df_copy[col].rolling(window=self.rolling_window).median()
            rolling_std = df_copy[col].rolling(window=self.rolling_window).std()
            
            upper_bound = rolling_median + (rolling_std * self.outlier_std_threshold)
            lower_bound = rolling_median - (rolling_std * self.outlier_std_threshold)
            
            df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Gestione outliers per volumi
        for col in self.volume_columns:
            df_copy[col] = replace_outliers(df_copy[col])
        
        # Salva statistiche outliers
        self.stats['outliers'] = {
            col: sum((df[col] != df_copy[col]).astype(int))
            for col in self.price_columns + self.volume_columns
        }
        
        return df_copy
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge indicatori tecnici usando la libreria ta"""
        if not self.technical_indicators:
            return df
        
        df_copy = df.copy()
        
        # Assumiamo che il primo prezzo sia il close price
        close = df_copy[self.price_columns[0]]
        
        if len(self.price_columns) >= 4:
            high = df_copy[self.price_columns[1]]
            low = df_copy[self.price_columns[2]]
            open = df_copy[self.price_columns[3]]
            
            # Momentum indicators
            df_copy['rsi'] = ta.momentum.RSIIndicator(close).rsi()
            df_copy['stoch'] = ta.momentum.StochasticOscillator(
                high, low, close).stoch()
            
            # Trend indicators
            df_copy['macd'] = ta.trend.MACD(close).macd()
            df_copy['adx'] = ta.trend.ADXIndicator(high, low, close).adx()
            
            # Volatility indicators
            df_copy['bb_high'] = ta.volatility.BollingerBands(
                close).bollinger_hband()
            df_copy['bb_low'] = ta.volatility.BollingerBands(
                close).bollinger_lband()
            
            # Volume indicators
            if self.volume_columns:
                volume = df_copy[self.volume_columns[0]]
                df_copy['obv'] = ta.volume.OnBalanceVolumeIndicator(
                    close, volume).on_balance_volume()
        
        # Moving averages
        df_copy['sma_20'] = ta.trend.SMAIndicator(
            close, window=20).sma_indicator()
        df_copy['ema_20'] = ta.trend.EMAIndicator(
            close, window=20).ema_indicator()
        
        # Rimuovi le righe con NaN create dagli indicatori
        df_copy.dropna(inplace=True)
        
        return df_copy
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge features temporali"""
        df_copy = df.copy()
        
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy['hour'] = df_copy.index.hour
            df_copy['day_of_week'] = df_copy.index.dayofweek
            df_copy['day_of_month'] = df_copy.index.day
            df_copy['month'] = df_copy.index.month
            df_copy['year'] = df_copy.index.year
            
            # One-hot encoding per features cicliche
            df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour']/24)
            df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour']/24)
            df_copy['day_sin'] = np.sin(2 * np.pi * df_copy['day_of_week']/7)
            df_copy['day_cos'] = np.cos(2 * np.pi * df_copy['day_of_week']/7)
            
        return df_copy
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scala le features usando diversi scalers"""
        df_copy = df.copy()
        
        if fit:
            # Scala prezzi
            df_copy[self.price_columns] = self.price_scaler.fit_transform(
                df_copy[self.price_columns])
            
            # Scala volumi
            if self.volume_columns:
                df_copy[self.volume_columns] = self.volume_scaler.fit_transform(
                    df_copy[self.volume_columns])
            
            # Scala altre features numeriche
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            other_cols = [col for col in numeric_cols 
                         if col not in self.price_columns + self.volume_columns]
            
            if other_cols:
                df_copy[other_cols] = self.other_scaler.fit_transform(
                    df_copy[other_cols])
        else:
            # Trasforma usando scaler già fittati
            df_copy[self.price_columns] = self.price_scaler.transform(
                df_copy[self.price_columns])
            
            if self.volume_columns:
                df_copy[self.volume_columns] = self.volume_scaler.transform(
                    df_copy[self.volume_columns])
            
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            other_cols = [col for col in numeric_cols 
                         if col not in self.price_columns + self.volume_columns]
            
            if other_cols:
                df_copy[other_cols] = self.other_scaler.transform(
                    df_copy[other_cols])
        
        return df_copy
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica tutto il preprocessing e fitta i trasformatori"""
        # Gestione missing values
        df_clean = self._handle_missing_values(df)
        
        # Gestione outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Aggiunta features tecniche
        df_clean = self._add_technical_indicators(df_clean)
        
        # Aggiunta features temporali
        df_clean = self._add_temporal_features(df_clean)
        
        # Scaling
        df_scaled = self._scale_features(df_clean, fit=True)
        
        return df_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica il preprocessing usando i parametri già fittati"""
        df_clean = self._handle_missing_values(df)
        df_clean = self._handle_outliers(df_clean)
        df_clean = self._add_technical_indicators(df_clean)
        df_clean = self._add_temporal_features(df_clean)
        df_scaled = self._scale_features(df_clean, fit=False)
        return df_scaled
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Ritorna i nomi delle features dopo il preprocessing"""
        return self.transform(df.head(1)).columns.tolist()

# Esempio di utilizzo
if __name__ == "__main__":
    """
    # Esempio dati
    df = pd.DataFrame({
        'close': [100, 101, 102, np.nan, 103],
        'high': [102, 103, 104, 105, 106],
        'low': [98, 99, 100, 101, 102],
        'open': [99, 100, 101, 102, 103],
        'volume': [1000, np.nan, 1200, 1300, 1400]
    }, index=pd.date_range('2023-01-01', periods=5))
    
    # Inizializzazione preprocessor
    preprocessor = TradingPreprocessor(
        price_columns=['close', 'high', 'low', 'open'],
        volume_columns=['volume'],
        technical_indicators=True
    )
    
    # Preprocessing
    df_processed = preprocessor.fit_transform(df)
    
    # Statistiche outliers
    print("Outliers trovati:", preprocessor.stats['outliers'])
    
    # Features generate
    print("Features disponibili:", preprocessor.get_feature_names(df))
    """