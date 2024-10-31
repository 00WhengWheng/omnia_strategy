# File: base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class TradeSignal:
    timestamp: datetime
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    size: float
    confidence: float
    metadata: Dict

class BaseStrategy(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Optional[TradeSignal]:
        pass
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        return logger