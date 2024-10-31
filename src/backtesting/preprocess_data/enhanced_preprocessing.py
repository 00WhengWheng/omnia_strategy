import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from scipy.interpolate import interp1d, splrep, splev
from scipy.fft import fft, ifft

def preprocess_data(data: pd.DataFrame, 
                    time_col: str,
                    numeric_cols: List[str], 
                    categorical_cols: List[str] = None,
                    freq: str = 'D',
                    interpolation_methods: List[str] = ['linear', 'polynomial', 'spline', 'fourier'],
                    poly_degree: int = 3,
                    spline_degree: int = 3,
                    knn_neighbors: int = 5,
                    model_imputer: str = 'knn',
                    model_iterations: int = 10) -> pd.DataFrame:
    """
    Preprocess data using various interpolation and imputation techniques.
    
    Args:
        data (pd.DataFrame): Input data
        time_col (str): Name of the time column
        numeric_cols (List[str]): List of numeric column names
        categorical_cols (List[str], optional): List of categorical column names. Defaults to None.
        freq (str, optional): Time series frequency. Defaults to 'D' (daily).
        interpolation_methods (List[str], optional): Interpolation methods to apply. 
            Defaults to ['linear', 'polynomial', 'spline', 'fourier'].
        poly_degree (int, optional): Degree of polynomial for polynomial interpolation. Defaults to 3.
        spline_degree (int, optional): Degree of spline for spline interpolation. Defaults to 3.
        knn_neighbors (int, optional): Number of neighbors for KNN imputation. Defaults to 5.
        model_imputer (str, optional): Model-based imputer to use ('knn' or 'iterative'). Defaults to 'knn'.
        model_iterations (int, optional): Number of iterations for iterative imputer. Defaults to 10.
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    df = data.copy()
    
    # Convert time column to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    
    # Resample to specified frequency
    df = df.resample(freq).last()
    
    # Separate numeric and categorical columns
    if categorical_cols is None:
        categorical_cols = [col for col in df.columns if col not in numeric_cols]
    
    numeric_df = df[numeric_cols]
    categorical_df = df[categorical_cols]
    
    # Define interpolation functions
    def linear_interpolation(series):
        return series.interpolate(method='linear')
    
    def polynomial_interpolation(series):
        indices = series.index.astype(float).values
        values = series.values
        valid_mask = ~np.isnan(values)
        
        if sum(valid_mask) < 2:
            return series
        
        f = interp1d(indices[valid_mask], values[valid_mask], kind=poly_degree, fill_value='extrapolate')
        interpolated_values = f(indices)
        
        return pd.Series(interpolated_values, index=series.index)
    
    def spline_interpolation(series):
        indices = series.index.astype(float).values
        values = series.values
        valid_mask = ~np.isnan(values)
        
        if sum(valid_mask) < 4:
            return series
        
        tck = splrep(indices[valid_mask], values[valid_mask], k=spline_degree)
        interpolated_values = splev(indices, tck)
        
        return pd.Series(interpolated_values, index=series.index)
    
    def fourier_interpolation(series):
        values = series.values
        valid_mask = ~np.isnan(values)
        
        if sum(valid_mask) < 2:
            return series
        
        fft_values = fft(values[valid_mask])
        fft_freq = np.fft.fftfreq(len(values[valid_mask]))
        
        interpolated_fft = np.zeros_like(values, dtype=np.complex128)
        interpolated_fft[valid_mask] = fft_values
        interpolated_values = ifft(interpolated_fft).real
        
        return pd.Series(interpolated_values, index=series.index)
    
    # Apply interpolation methods
    interpolated_dfs = []
    
    for method in interpolation_methods:
        if method == 'linear':
            interpolated_df = numeric_df.apply(linear_interpolation)
        elif method == 'polynomial':
            interpolated_df = numeric_df.apply(polynomial_interpolation)
        elif method == 'spline':
            interpolated_df = numeric_df.apply(spline_interpolation)
        elif method == 'fourier':
            interpolated_df = numeric_df.apply(fourier_interpolation)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
            
        interpolated_dfs.append(interpolated_df)
        
    # Combine interpolated results
    numeric_df_interpolated = pd.concat(interpolated_dfs).groupby(level=0).mean()
    
    # Impute remaining missing values using model-based imputers
    if model_imputer == 'knn':
        imputer = KNNImputer(n_neighbors=knn_neighbors)
    elif model_imputer == 'iterative':
        imputer = IterativeImputer(max_iter=model_iterations, random_state=42)
    else:
        raise ValueError(f"Unsupported model imputer: {model_imputer}")
        
    numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df_interpolated), 
                                      columns=numeric_df.columns,
                                      index=numeric_df.index)
    
    # Impute categorical columns
    categorical_df_imputed = categorical_df.apply(lambda x: x.fillna(x.mode()[0]))
    
    # Combine numeric and categorical dataframes
    df_preprocessed = pd.concat([numeric_df_imputed, categorical_df_imputed], axis=1)
    
    return df_preprocessed
