import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from .base import BaseAnalyzer, AnalysisResult

class CorrelationAnalyzer(BaseAnalyzer):
    def _initialize_analyzer(self) -> None:
        """Inizializza l'analizzatore delle correlazioni"""
        self.lookback_window = self.config.get('correlation.lookback_window', 252)
        self.min_periods = self.config.get('correlation.min_periods', 30)
        self.correlation_threshold = self.config.get('correlation.threshold', 0.7)
        self.max_lag = self.config.get('correlation.max_lag', 10)
        
        self.scaler = StandardScaler()
        self.correlation_types = ['pearson', 'spearman']
        self.correlation_history = {}

    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        """Analizza le correlazioni tra i diversi asset/indicatori"""
        if not self.validate_data(data):
            raise ValueError("Invalid data for correlation analysis")

        # Calcola le diverse metriche di correlazione
        basic_corr = self._calculate_basic_correlations(data)
        rolling_corr = self._calculate_rolling_correlations(data)
        regime_corr = self._analyze_regime_correlations(data)
        lead_lag = self._analyze_lead_lag_relationships(data)

        # Calcola la stabilità delle correlazioni
        stability = self._calculate_correlation_stability(rolling_corr)

        # Identifica cluster di correlazione
        clusters = self._identify_correlation_clusters(basic_corr['pearson'])

        # Calcola la confidenza complessiva
        confidence = self._calculate_correlation_confidence(
            basic_corr, stability, len(data))

        result = AnalysisResult(
            timestamp=datetime.now(),
            value=self._calculate_composite_score(basic_corr, stability),
            confidence=confidence,
            components={
                'pearson_mean': basic_corr['pearson'].mean().mean(),
                'spearman_mean': basic_corr['spearman'].mean().mean(),
                'stability': stability,
                'clusters': len(clusters)
            },
            metadata={
                'correlation_matrices': basic_corr,
                'rolling_correlations': rolling_corr,
                'regime_correlations': regime_corr,
                'lead_lag_relationships': lead_lag,
                'correlation_clusters': clusters
            }
        )

        self.update_history(result)
        return result

    def _calculate_basic_correlations(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calcola correlazioni base usando diversi metodi"""
        results = {}
        
        # Pearson correlation
        results['pearson'] = data.corr(method='pearson')
        
        # Spearman correlation
        results['spearman'] = data.corr(method='spearman')
        
        return results

    def _calculate_rolling_correlations(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calcola correlazioni rolling per analisi della stabilità"""
        rolling_corr = {}
        
        for method in self.correlation_types:
            rolling = data.rolling(window=self.lookback_window, min_periods=self.min_periods)
            rolling_corr[method] = rolling.corr(method=method)
            
        return rolling_corr

    def _analyze_regime_correlations(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analizza correlazioni condizionate al regime di mercato"""
        if 'regime' not in data.columns:
            return {}
            
        regime_correlations = {}
        
        for regime in data['regime'].unique():
            regime_data = data[data['regime'] == regime]
            if len(regime_data) >= self.min_periods:
                regime_correlations[regime] = {
                    'pearson': regime_data.corr(method='pearson'),
                    'spearman': regime_data.corr(method='spearman')
                }
                
        return regime_correlations

    def _analyze_lead_lag_relationships(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Analizza relazioni lead/lag tra le serie"""
        lead_lag = {}
        
        for col1 in data.columns:
            lead_lag[col1] = {}
            for col2 in data.columns:
                if col1 != col2:
                    lag_analysis = self._calculate_lag_correlation(
                        data[col1], data[col2])
                    if abs(lag_analysis['max_correlation']) > self.correlation_threshold:
                        lead_lag[col1][col2] = lag_analysis
                        
        return lead_lag

    def _calculate_lag_correlation(self, series1: pd.Series, 
                                 series2: pd.Series) -> Dict[str, float]:
        """Calcola correlazione con lag per identificare lead/lag"""
        correlations = {}
        
        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag < 0:
                s1 = series1.iloc[:lag]
                s2 = series2.iloc[-lag:]
            elif lag > 0:
                s1 = series1.iloc[lag:]
                s2 = series2.iloc[:-lag]
            else:
                s1, s2 = series1, series2
                
            corr, _ = pearsonr(s1, s2)
            correlations[lag] = corr
            
        max_corr_lag = max(correlations.items(), key=lambda x: abs(x[1]))
        
        return {
            'optimal_lag': max_corr_lag[0],
            'max_correlation': max_corr_lag[1],
            'all_correlations': correlations
        }

    def _calculate_correlation_stability(self, rolling_correlations: Dict) -> float:
        """Calcola la stabilità delle correlazioni nel tempo"""
        stabilities = []
        
        for method, corr_matrix in rolling_correlations.items():
            # Calcola la volatilità delle correlazioni
            correlation_volatility = corr_matrix.std()
            # Normalizza e converti in misura di stabilità
            stability = 1 - correlation_volatility.mean()
            stabilities.append(stability)
            
        return np.mean(stabilities)

    def _identify_correlation_clusters(self, correlation_matrix: pd.DataFrame) -> List:
        """Identifica cluster di asset/indicatori altamente correlati"""
        # Converti matrice correlazione in matrice distanza
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        clusters = []
        visited = set()
        
        for col in correlation_matrix.columns:
            if col in visited:
                continue
                
            # Trova tutti gli elementi altamente correlati
            cluster = set([col])
            for other_col in correlation_matrix.columns:
                if other_col != col and abs(correlation_matrix.loc[col, other_col]) > self.correlation_threshold:
                    cluster.add(other_col)
                    
            if len(cluster) > 1:
                clusters.append(list(cluster))
                visited.update(cluster)
                
        return clusters

    def _calculate_correlation_confidence(self, basic_correlations: Dict,
                                       stability: float, n_samples: int) -> float:
        """Calcola la confidenza nelle correlazioni identificate"""
        # Maggiore confidenza per:
        # 1. Correlazioni stabili nel tempo
        # 2. Concordanza tra diversi metodi di correlazione
        # 3. Numero sufficiente di campioni
        
        # Stabilità temporale (0-1)
        stability_score = stability
        
        # Concordanza tra metodi
        method_agreement = 1 - abs(
            basic_correlations['pearson'].mean().mean() - 
            basic_correlations['spearman'].mean().mean()
        )
        
        # Adeguatezza campione
        sample_score = min(1.0, n_samples / self.lookback_window)
        
        # Combina i fattori
        confidence = (
            0.4 * stability_score +
            0.4 * method_agreement +
            0.2 * sample_score
        )
        
        return np.clip(confidence, 0, 1)

    def _calculate_composite_score(self, basic_correlations: Dict,
                                stability: float) -> float:
        """Calcola uno score composito di correlazione"""
        # Media delle correlazioni assolute
        mean_correlation = np.mean([
            np.abs(basic_correlations['pearson']).mean().mean(),
            np.abs(basic_correlations['spearman']).mean().mean()
        ])
        
        # Aggiusta per stabilità
        composite = mean_correlation * (0.5 + 0.5 * stability)
        
        return np.clip(composite, -1, 1)

    def get_required_columns(self) -> list:
        """Specifica le colonne richieste per l'analisi"""
        return ['close', 'volume', 'high', 'low']  # Minimo richiesto

    def get_correlation_summary(self) -> Dict:
        """Fornisce un sommario delle correlazioni più significative"""
        if self.results_history.empty:
            return {}
            
        latest_result = self.results_history.iloc[-1]
        
        return {
            'timestamp': latest_result['timestamp'],
            'overall_correlation': latest_result['value'],
            'confidence': latest_result['confidence'],
            'stability': latest_result['stability'],
            'significant_pairs': self._get_significant_pairs(latest_result),
            'clusters': latest_result['metadata']['correlation_clusters']
        }
