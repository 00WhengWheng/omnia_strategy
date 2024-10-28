import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from .base import BaseAnalyzer, AnalysisResult
import requests
import json
from textblob import TextBlob
import yfinance as yf

@dataclass
class SentimentSource:
    name: str
    weight: float
    update_frequency: str
    confidence: float
    last_update: datetime

class SentimentAnalyzer(BaseAnalyzer):
    def _initialize_analyzer(self) -> None:
        """Inizializza l'analizzatore del sentiment"""
        # Configurazione fonti sentiment
        self.sources = {
            'market': SentimentSource(
                name='market_data',
                weight=0.35,
                update_frequency='1h',
                confidence=0.8,
                last_update=datetime.now()
            ),
            'social': SentimentSource(
                name='social_media',
                weight=0.20,
                update_frequency='1h',
                confidence=0.6,
                last_update=datetime.now()
            ),
            'news': SentimentSource(
                name='news_sentiment',
                weight=0.25,
                update_frequency='1h',
                confidence=0.7,
                last_update=datetime.now()
            ),
            'positioning': SentimentSource(
                name='market_positioning',
                weight=0.20,
                update_frequency='1d',
                confidence=0.75,
                last_update=datetime.now()
            )
        }
        
        # Metriche di mercato
        self.market_metrics = {
            'vix_threshold': self.config.get('sentiment.vix_threshold', 20),
            'put_call_threshold': self.config.get('sentiment.put_call_threshold', 1.0),
            'fear_greed_neutral': self.config.get('sentiment.fear_greed_neutral', 50)
        }
        
        # Cache per i dati
        self.cache = {}
        self.cache_duration = timedelta(hours=1)

    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        """Esegue l'analisi completa del sentiment"""
        # Analisi delle diverse componenti
        market_sentiment = self._analyze_market_sentiment(data)
        social_sentiment = self._analyze_social_sentiment()
        news_sentiment = self._analyze_news_sentiment()
        positioning_sentiment = self._analyze_positioning()
        
        # Combina i risultati
        composite_sentiment = self._calculate_composite_sentiment({
            'market': market_sentiment,
            'social': social_sentiment,
            'news': news_sentiment,
            'positioning': positioning_sentiment
        })
        
        # Calcola la confidenza
        confidence = self._calculate_sentiment_confidence({
            'market': market_sentiment,
            'social': social_sentiment,
            'news': news_sentiment,
            'positioning': positioning_sentiment
        })

        result = AnalysisResult(
            timestamp=datetime.now(),
            value=composite_sentiment['signal'],
            confidence=confidence,
            components={
                'market': market_sentiment['signal'],
                'social': social_sentiment['signal'],
                'news': news_sentiment['signal'],
                'positioning': positioning_sentiment['signal']
            },
            metadata={
                'market_details': market_sentiment,
                'social_details': social_sentiment,
                'news_details': news_sentiment,
                'positioning_details': positioning_sentiment,
                'composite': composite_sentiment
            }
        )

        self.update_history(result)
        return result

    def _analyze_market_sentiment(self, data: pd.DataFrame) -> Dict:
        """Analizza il sentiment basato su dati di mercato"""
        # VIX analysis
        vix_sentiment = self._analyze_vix()
        
        # Put/Call Ratio
        put_call = self._analyze_put_call_ratio()
        
        # Market Breadth
        breadth = self._analyze_market_breadth(data)
        
        # Fear & Greed Index
        fear_greed = self._calculate_fear_greed_index(data)
        
        # Calcola segnale composito
        signal = self._calculate_market_sentiment_signal(
            vix_sentiment,
            put_call,
            breadth,
            fear_greed
        )
        
        return {
            'signal': signal,
            'vix': vix_sentiment,
            'put_call': put_call,
            'breadth': breadth,
            'fear_greed': fear_greed,
            'confidence': self.sources['market'].confidence
        }

    def _analyze_social_sentiment(self) -> Dict:
        """Analizza il sentiment dai social media"""
        # Twitter sentiment
        twitter = self._analyze_twitter_sentiment()
        
        # Reddit sentiment
        reddit = self._analyze_reddit_sentiment()
        
        # StockTwits sentiment
        stocktwits = self._analyze_stocktwits_sentiment()
        
        # Calcola segnale composito
        signal = self._calculate_social_sentiment_signal(
            twitter, reddit, stocktwits)
            
        return {
            'signal': signal,
            'twitter': twitter,
            'reddit': reddit,
            'stocktwits': stocktwits,
            'confidence': self.sources['social'].confidence
        }

    def _analyze_news_sentiment(self) -> Dict:
        """Analizza il sentiment dalle news finanziarie"""
        # Financial News Analysis
        news_sentiment = self._analyze_financial_news()
        
        # Blogs and Analysis
        blog_sentiment = self._analyze_financial_blogs()
        
        # Analyst Reports
        analyst_sentiment = self._analyze_analyst_reports()
        
        # Calcola segnale composito
        signal = self._calculate_news_sentiment_signal(
            news_sentiment, blog_sentiment, analyst_sentiment)
            
        return {
            'signal': signal,
            'news': news_sentiment,
            'blogs': blog_sentiment,
            'analysts': analyst_sentiment,
            'confidence': self.sources['news'].confidence
        }

    def _analyze_positioning(self) -> Dict:
        """Analizza il positioning di mercato"""
        # COT Data
        cot = self._analyze_cot_data()
        
        # Fund Flows
        flows = self._analyze_fund_flows()
        
        # Margin Debt
        margin = self._analyze_margin_debt()
        
        # Short Interest
        short_interest = self._analyze_short_interest()
        
        # Calcola segnale composito
        signal = self._calculate_positioning_signal(
            cot, flows, margin, short_interest)
            
        return {
            'signal': signal,
            'cot': cot,
            'flows': flows,
            'margin': margin,
            'short_interest': short_interest,
            'confidence': self.sources['positioning'].confidence
        }

    def _calculate_composite_sentiment(self, sentiments: Dict) -> Dict:
        """Calcola il sentiment composito pesato"""
        composite_signal = 0
        total_weight = 0
        
        for source, sentiment in sentiments.items():
            weight = self.sources[source].weight
            composite_signal += sentiment['signal'] * weight
            total_weight += weight
            
        if total_weight > 0:
            composite_signal /= total_weight
            
        return {
            'signal': np.clip(composite_signal, -1, 1),
            'extremes': self._detect_sentiment_extremes(sentiments),
            'divergences': self._detect_sentiment_divergences(sentiments),
            'trends': self._analyze_sentiment_trends(sentiments)
        }

    def _calculate_sentiment_confidence(self, sentiments: Dict) -> float:
        """Calcola la confidenza complessiva del sentiment"""
        confidences = []
        weights = []
        
        for source, sentiment in sentiments.items():
            # Base confidence from source
            base_conf = self.sources[source].confidence
            
            # Adjust for data freshness
            freshness = self._calculate_data_freshness(
                self.sources[source].last_update,
                self.sources[source].update_frequency
            )
            
            # Adjust for signal strength
            signal_strength = abs(sentiment['signal'])
            
            confidence = base_conf * freshness * (0.5 + 0.5 * signal_strength)
            confidences.append(confidence)
            weights.append(self.sources[source].weight)
            
        return np.average(confidences, weights=weights)

    def _detect_sentiment_extremes(self, sentiments: Dict) -> Dict:
        """Identifica estremi nel sentiment"""
        extremes = {}
        for source, sentiment in sentiments.items():
            if abs(sentiment['signal']) > 0.8:
                extremes[source] = {
                    'value': sentiment['signal'],
                    'type': 'bullish' if sentiment['signal'] > 0 else 'bearish'
                }
        return extremes

    def _detect_sentiment_divergences(self, sentiments: Dict) -> List:
        """Identifica divergenze tra fonti di sentiment"""
        divergences = []
        sources = list(sentiments.keys())
        
        for i in range(len(sources)):
            for j in range(i+1, len(sources)):
                source1, source2 = sources[i], sources[j]
                signal1 = sentiments[source1]['signal']
                signal2 = sentiments[source2]['signal']
                
                if np.sign(signal1) != np.sign(signal2) and \
                   abs(signal1 - signal2) > 0.5:
                    divergences.append({
                        'sources': (source1, source2),
                        'difference': abs(signal1 - signal2),
                        'signals': (signal1, signal2)
                    })
                    
        return divergences

    def _analyze_sentiment_trends(self, sentiments: Dict) -> Dict:
        """Analizza i trend del sentiment"""
        trends = {}
        
        for source, sentiment in sentiments.items():
            if not self.results_history.empty:
                historical_signals = self.results_history[f'components.{source}']
                
                trends[source] = {
                    'current': sentiment['signal'],
                    'sma_5': historical_signals.tail(5).mean(),
                    'sma_20': historical_signals.tail(20).mean(),
                    'trend': self._calculate_trend_strength(historical_signals)
                }
                
        return trends

    def get_required_columns(self) -> list:
        """Ritorna le colonne richieste per l'analisi"""
        return ['close', 'volume', 'high', 'low']

    def get_sentiment_summary(self) -> Dict:
        """Fornisce un sommario del sentiment corrente"""
        if self.results_history.empty:
            return {}
            
        latest = self.results_history.iloc[-1]
        
        return {
            'timestamp': latest['timestamp'],
            'overall_sentiment': latest['value'],
            'confidence': latest['confidence'],
            'extremes': latest['metadata']['composite']['extremes'],
            'divergences': latest['metadata']['composite']['divergences'],
            'trends': latest['metadata']['composite']['trends']
        }
