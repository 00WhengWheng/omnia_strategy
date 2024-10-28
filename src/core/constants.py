from enum import Enum

class MarketRegime(Enum):
    CRISIS = "crisis"
    RISK_OFF = "risk_off"
    NORMAL = "normal"
    RISK_ON = "risk_on"
    EUPHORIA = "euphoria"

class TimeFrame(Enum):
    M1 = "1min"
    M5 = "5min"
    M15 = "15min"
    M30 = "30min"
    H1 = "1hour"
    H4 = "4hour"
    D1 = "1day"
    W1 = "1week"

class SignalSource(Enum):
    MACRO = "macro"
    ALGORITHMIC = "algorithmic"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"

class SignalStrength(Enum):
    STRONG_SELL = -1.0
    MODERATE_SELL = -0.5
    NEUTRAL = 0.0
    MODERATE_BUY = 0.5
    STRONG_BUY = 1.0

DEFAULT_LOOKBACK = 252
MIN_CONFIDENCE_THRESHOLD = 0.6
MAX_POSITION_SIZE = 0.1
