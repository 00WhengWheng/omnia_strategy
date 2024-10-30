from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import requests
from collections import defaultdict
from ..base import BaseAnalyzer

@dataclass
class NewsItem:
    timestamp: datetime
    source: str
    title: str
    content: str
    relevance: float
    sentiment_score: float
    entities: List[str]
    topics: List[str]

@dataclass
class SentimentState:
    timestamp: datetime
    news_sentiment: Dict[str, float]      # News-based sentiment metrics
    social_sentiment: Dict[str, float]    # Social media sentiment
    market_sentiment: Dict[str, float]    # Market-based sentiment indicators
    investor_sentiment: Dict[str, float]  # Investor sentiment surveys
    aggregate_sentiment: float            # Combined sentiment score
    sentiment_signals: Dict[str, float]   # Trading signals
    momentum: Dict[str, float]            # Sentiment momentum
    confidence: float
    metadata: Dict

class SentimentAnalyzer(BaseAnalyzer):
    """Market Sentiment Analysis Component"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentiment analyzers
        self.text_analyzer = TextBlob
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Configuration
        self.news_sources = self.config.get('news_sources', [
            'reuters', 'bloomberg', 'wsj'
        ])
        self.social_sources = self.config.get('social_sources', [
            'twitter', 'reddit', 'stocktwits'
        ])
        self.sentiment_window = self.config.get('sentiment_window', 24)  # hours
        self.relevance_threshold = self.config.get('relevance_threshold', 0.5)
        
        # Weights for different sentiment sources
        self.source_weights = self.config.get('source_weights', {
            'news': 0.3,
            'social': 0.2,
            'market': 0.3,
            'investor': 0.2
        })
        
        # Cache for sentiment data
        self.sentiment_cache = defaultdict(list)
        self.news_cache: List[NewsItem] = []

    def analyze(self, data: Dict) -> SentimentState:
        """
        Perform comprehensive sentiment analysis
        
        Parameters:
        - data: Dictionary containing:
            - news_data: Recent news articles
            - social_data: Social media data
            - market_data: Market indicators
            - survey_data: Investor surveys
        
        Returns:
        - SentimentState object containing analysis results
        """
        try:
            # Analyze news sentiment
            news_sentiment = self._analyze_news_sentiment(data.get('news_data', []))
            
            # Analyze social media sentiment
            social_sentiment = self._analyze_social_sentiment(
                data.get('social_data', []))
            
            # Analyze market-based sentiment
            market_sentiment = self._analyze_market_sentiment(
                data.get('market_data', pd.DataFrame()))
            
            # Analyze investor sentiment
            investor_sentiment = self._analyze_investor_sentiment(
                data.get('survey_data', {}))
            
            # Calculate aggregate sentiment
            aggregate_sentiment = self._calculate_aggregate_sentiment(
                news_sentiment,
                social_sentiment,
                market_sentiment,
                investor_sentiment
            )
            
            # Generate sentiment signals
            sentiment_signals = self._generate_sentiment_signals(
                news_sentiment,
                social_sentiment,
                market_sentiment,
                investor_sentiment,
                aggregate_sentiment
            )
            
            # Calculate sentiment momentum
            sentiment_momentum = self._calculate_sentiment_momentum()
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                news_sentiment,
                social_sentiment,
                market_sentiment,
                investor_sentiment
            )
            
            # Generate metadata
            metadata = self._generate_metadata(data)
            
            state = SentimentState(
                timestamp=datetime.now(),
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                market_sentiment=market_sentiment,
                investor_sentiment=investor_sentiment,
                aggregate_sentiment=aggregate_sentiment,
                sentiment_signals=sentiment_signals,
                momentum=sentiment_momentum,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update cache
            self._update_sentiment_cache(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            raise

    def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict[str, float]:
        """Analyze sentiment from news articles"""
        article_sentiments = []
        entity_sentiments = defaultdict(list)
        topic_sentiments = defaultdict(list)
        
        for article in news_data:
            # Calculate article relevance
            relevance = self._calculate_relevance(article)
            
            if relevance >= self.relevance_threshold:
                # Analyze text sentiment
                text_sentiment = self._analyze_text_sentiment(
                    f"{article['title']} {article['content']}")
                
                # Extract entities and topics
                entities = self._extract_entities(article['content'])
                topics = self._extract_topics(article['content'])
                
                # Store article sentiment
                article_sentiments.append({
                    'sentiment': text_sentiment,
                    'relevance': relevance,
                    'timestamp': article['timestamp']
                })
                
                # Update entity and topic sentiments
                for entity in entities:
                    entity_sentiments[entity].append(text_sentiment)
                for topic in topics:
                    topic_sentiments[topic].append(text_sentiment)
        
        # Calculate aggregated sentiments
        return {
            'overall': self._calculate_weighted_sentiment(article_sentiments),
            'entities': {k: np.mean(v) for k, v in entity_sentiments.items()},
            'topics': {k: np.mean(v) for k, v in topic_sentiments.items()},
            'trend': self._calculate_sentiment_trend(article_sentiments),
            'volume': len(article_sentiments)
        }

    def _analyze_social_sentiment(self, social_data: List[Dict]) -> Dict[str, float]:
        """Analyze sentiment from social media data"""
        sentiments = defaultdict(list)
        
        for post in social_data:
            source = post['source']
            # Calculate sentiment score
            sentiment_score = self._analyze_text_sentiment(post['content'])
            
            # Calculate influence score
            influence = self._calculate_influence_score(post)
            
            sentiments[source].append({
                'sentiment': sentiment_score,
                'influence': influence,
                'timestamp': post['timestamp']
            })
        
        return {
            'twitter': self._aggregate_social_sentiment(sentiments['twitter']),
            'reddit': self._aggregate_social_sentiment(sentiments['reddit']),
            'stocktwits': self._aggregate_social_sentiment(sentiments['stocktwits']),
            'overall': self._combine_social_sentiments(sentiments),
            'momentum': self._calculate_social_momentum(sentiments)
        }

    def _analyze_market_sentiment(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze sentiment from market indicators"""
        # Fear & Greed index components
        fear_greed = self._calculate_fear_greed_index(market_data)
        
        # Put/Call ratio analysis
        put_call = self._analyze_put_call_ratio(market_data)
        
        # VIX analysis
        vix_sentiment = self._analyze_vix(market_data)
        
        # Market breadth
        breadth = self._analyze_market_breadth(market_data)
        
        return {
            'fear_greed': fear_greed,
            'put_call': put_call,
            'vix_sentiment': vix_sentiment,
            'market_breadth': breadth,
            'overall': self._combine_market_indicators(
                fear_greed, put_call, vix_sentiment, breadth)
        }

    def _analyze_investor_sentiment(self, survey_data: Dict) -> Dict[str, float]:
        """Analyze sentiment from investor surveys"""
        # AAII sentiment
        aaii = self._analyze_aaii_sentiment(survey_data.get('aaii', {}))
        
        # Investors Intelligence
        ii = self._analyze_ii_sentiment(survey_data.get('ii', {}))
        
        # Professional surveys
        prof = self._analyze_professional_sentiment(
            survey_data.get('professional', {}))
        
        return {
            'aaii': aaii,
            'investors_intelligence': ii,
            'professional': prof,
            'overall': self._combine_survey_sentiments(aaii, ii, prof),
            'divergence': self._calculate_survey_divergence(aaii, ii, prof)
        }

    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using multiple methods"""
        # TextBlob sentiment
        blob_sentiment = TextBlob(text).sentiment.polarity
        
        # VADER sentiment
        vader_sentiment = self.vader_analyzer.polarity_scores(text)
        
        # Combine sentiments
        combined = (blob_sentiment + vader_sentiment['compound']) / 2
        
        return np.clip(combined, -1, 1)

    def _calculate_aggregate_sentiment(self, news: Dict, social: Dict,
                                    market: Dict, investor: Dict) -> float:
        """Calculate aggregate sentiment from all sources"""
        components = {
            'news': news['overall'],
            'social': social['overall'],
            'market': market['overall'],
            'investor': investor['overall']
        }
        
        # Apply source weights
        weighted_sentiment = sum(
            components[source] * weight
            for source, weight in self.source_weights.items()
        )
        
        return np.clip(weighted_sentiment, -1, 1)

    def _generate_sentiment_signals(self, news: Dict, social: Dict,
                                  market: Dict, investor: Dict,
                                  aggregate: float) -> Dict[str, float]:
        """Generate trading signals from sentiment analysis"""
        signals = {}
        
        # Extreme sentiment signal
        signals['extreme'] = self._detect_extreme_sentiment(aggregate)
        
        # Sentiment divergence signal
        signals['divergence'] = self._detect_sentiment_divergence(
            news, social, market, investor)
        
        # Sentiment momentum signal
        signals['momentum'] = self._calculate_sentiment_momentum()
        
        # Sentiment reversal signal
        signals['reversal'] = self._detect_sentiment_reversal()
        
        # Combine signals
        signals['composite'] = self._combine_sentiment_signals(signals)
        
        return signals

    def _calculate_confidence(self, news: Dict, social: Dict,
                            market: Dict, investor: Dict) -> float:
        """Calculate confidence in sentiment analysis"""
        # Calculate individual confidences
        confidences = {
            'news': self._calculate_news_confidence(news),
            'social': self._calculate_social_confidence(social),
            'market': self._calculate_market_confidence(market),
            'investor': self._calculate_investor_confidence(investor)
        }
        
        # Weight confidences
        weighted_confidence = sum(
            confidences[source] * weight
            for source, weight in self.source_weights.items()
        )
        
        return np.clip(weighted_confidence, 0, 1)

    @property
    def required_columns(self) -> Dict[str, List[str]]:
        """Required columns for each data type"""
        return {
            'news_data': ['timestamp', 'source', 'title', 'content'],
            'social_data': ['timestamp', 'source', 'content', 'user_stats'],
            'market_data': ['close', 'volume', 'vix', 'advance_decline', 'put_call'],
            'survey_data': ['timestamp', 'bullish', 'bearish', 'neutral']
        }

    def get_sentiment_summary(self) -> Dict:
        """Get summary of current sentiment state"""
        if not self.sentiment_cache['aggregate']:
            return {}
            
        latest = self.sentiment_cache['aggregate'][-1]
        return {
            'timestamp': latest.timestamp,
            'aggregate_sentiment': latest.aggregate_sentiment,
            'news_sentiment': latest.news_sentiment['overall'],
            'social_sentiment': latest.social_sentiment['overall'],
            'market_sentiment': latest.market_sentiment['overall'],
            'signals': latest.sentiment_signals['composite'],
            'confidence': latest.confidence
        }
