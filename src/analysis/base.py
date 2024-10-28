from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    timestamp: datetime
    value: float
    confidence: float
    components: Dict[str, float]
    metadata: Dict[str, Any]

class BaseAnalyzer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_history: pd.DataFrame = pd.DataFrame()
        self._initialize_analyzer()
    
    @abstractmethod
    def _initialize_analyzer(self) -> None:
        """Inizializza l'analizzatore specifico"""
        pass
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        """Esegue l'analisi sui dati"""
        pass
    
    def update_history(self, result: AnalysisResult) -> None:
        """Aggiorna lo storico dei risultati"""
        new_row = {
            'timestamp': result.timestamp,
            'value': result.value,
            'confidence': result.confidence,
            **result.components,
            **result.metadata
        }
        self.results_history = pd.concat([
            self.results_history,
            pd.DataFrame([new_row])
        ])
    
    def get_history(self, lookback: Optional[int] = None) -> pd.DataFrame:
        """Recupera lo storico dei risultati"""
        if lookback:
            return self.results_history.tail(lookback)
        return self.results_history

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Valida i dati in input"""
        if data.empty:
            return False
        required_columns = self.get_required_columns()
        return all(col in data.columns for col in required_columns)

    @abstractmethod
    def get_required_columns(self) -> list:
        """Ritorna le colonne richieste per l'analisi"""
        pass
    
