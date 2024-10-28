import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import yfinance as yf
import requests
import sqlite3
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from ..core.constants import TimeFrame

class DataManager:
    def __init__(self, config: Dict):
        """Inizializza Data Manager"""
        self.config = config
        self.db_path = Path(config.get('db_path', 'data/market_data.db'))
        self.cache_dir = Path(config.get('cache_dir', 'data/cache'))
        self.sources = config.get('data_sources', {
            'primary': 'yfinance',
            'alternative': ['alpha_vantage', 'polygon']
        })
        
        # Setup directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Cache settings
        self.use_cache = config.get('use_cache', True)
        self.cache_expiry = config.get('cache_expiry', 24)  # hours
        
        # Data quality settings
        self.min_data_points = config.get('min_data_points', 252)
        self.max_missing_pct = config.get('max_missing_pct', 0.1)
        
    def _init_database(self):
        """Inizializza il database SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Market Data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    timestamp DATETIME,
                    timeframe TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    source TEXT,
                    PRIMARY KEY (symbol, timestamp, timeframe)
                )
            ''')
            
            # Fundamental Data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    symbol TEXT,
                    timestamp DATETIME,
                    data_type TEXT,
                    value REAL,
                    source TEXT,
                    PRIMARY KEY (symbol, timestamp, data_type)
                )
            ''')
            
            # Alternative Data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alternative_data (
                    symbol TEXT,
                    timestamp DATETIME,
                    data_type TEXT,
                    value TEXT,
                    source TEXT,
                    PRIMARY KEY (symbol, timestamp, data_type)
                )
            ''')
            
            conn.commit()
            
    def get_market_data(self,
                       symbols: Union[str, List[str]],
                       start_date: datetime,
                       end_date: datetime,
                       timeframe: TimeFrame = TimeFrame.D1,
                       include_fundamentals: bool = False,
                       force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """Recupera dati di mercato per i simboli specificati"""
        
        if isinstance(symbols, str):
            symbols = [symbols]
            
        data = {}
        missing_data = []
        
        for symbol in symbols:
            # Check cache first
            if self.use_cache and not force_download:
                cached_data = self._get_from_cache(symbol, start_date, end_date, timeframe)
                if cached_data is not None:
                    data[symbol] = cached_data
                    continue
                    
            # Try to get from database
            db_data = self._get_from_database(symbol, start_date, end_date, timeframe)
            if db_data is not None:
                data[symbol] = db_data
                continue
                
            missing_data.append(symbol)
            
        # Download missing data
        if missing_data:
            downloaded_data = self._download_market_data(
                missing_data, start_date, end_date, timeframe)
            data.update(downloaded_data)
            
        # Add fundamentals if requested
        if include_fundamentals:
            fundamental_data = self._get_fundamental_data(symbols, start_date, end_date)
            for symbol in symbols:
                if symbol in data and symbol in fundamental_data:
                    data[symbol] = pd.merge(
                        data[symbol],
                        fundamental_data[symbol],
                        left_index=True,
                        right_index=True,
                        how='left'
                    )
                    
        return data
        
    def _download_market_data(self,
                            symbols: List[str],
                            start_date: datetime,
                            end_date: datetime,
                            timeframe: TimeFrame) -> Dict[str, pd.DataFrame]:
        """Scarica dati di mercato da fonti multiple"""
        data = {}
        
        # Try primary source first
        primary_source = self.sources['primary']
        primary_data = self._download_from_source(
            primary_source, symbols, start_date, end_date, timeframe)
        
        # Fill missing data from alternative sources
        missing_symbols = [s for s in symbols if s not in primary_data]
        if missing_symbols:
            for source in self.sources['alternative']:
                if not missing_symbols:
                    break
                alt_data = self._download_from_source(
                    source, missing_symbols, start_date, end_date, timeframe)
                primary_data.update(alt_data)
                missing_symbols = [s for s in missing_symbols if s not in alt_data]
                
        # Process and validate downloaded data
        for symbol, df in primary_data.items():
            processed_df = self._process_market_data(df)
            if self._validate_data(processed_df):
                data[symbol] = processed_df
                # Save to database
                self._save_to_database(symbol, processed_df, timeframe)
                # Update cache
                if self.use_cache:
                    self._save_to_cache(symbol, processed_df, timeframe)
                    
        return data
        
    def _process_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa i dati di mercato"""
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Add derived columns
        df = self._add_derived_columns(df)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by index
        df.sort_index(inplace=True)
        
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gestisce i valori mancanti"""
        # Forward fill prices
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
        
        # Fill volume with 0
        df['volume'] = df['volume'].fillna(0)
        
        # Drop rows still containing NaN
        df.dropna(inplace=True)
        
        return df
        
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge colonne derivate"""
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Trading range
        df['range'] = df['high'] - df['low']
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        return df
        
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Valida la qualità dei dati"""
        if len(df) < self.min_data_points:
            return False
            
        missing_pct = df.isnull().mean().max()
        if missing_pct > self.max_missing_pct:
            return False
            
        return True
        
    def update_market_data(self, symbols: List[str],
                          timeframes: List[TimeFrame]) -> None:
        """Aggiorna i dati di mercato al presente"""
        last_dates = self._get_last_dates(symbols, timeframes)
        
        for symbol in symbols:
            for timeframe in timeframes:
                last_date = last_dates.get((symbol, timeframe))
                if last_date:
                    start_date = last_date + timedelta(days=1)
                    end_date = datetime.now()
                    
                    if start_date < end_date:
                        self.get_market_data(
                            symbol,
                            start_date,
                            end_date,
                            timeframe,
                            force_download=True
                        )
                        
    def _get_last_dates(self, symbols: List[str],
                       timeframes: List[TimeFrame]) -> Dict[Tuple[str, TimeFrame], datetime]:
        """Recupera le ultime date disponibili per ogni simbolo e timeframe"""
        last_dates = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for symbol in symbols:
                for timeframe in timeframes:
                    cursor.execute('''
                        SELECT MAX(timestamp)
                        FROM market_data
                        WHERE symbol = ? AND timeframe = ?
                    ''', (symbol, timeframe.value))
                    
                    result = cursor.fetchone()
                    if result[0]:
                        last_dates[(symbol, timeframe)] = pd.to_datetime(result[0])
                        
        return last_dates
        
    def get_fundamental_data(self,
                           symbols: List[str],
                           fields: List[str],
                           as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """Recupera dati fondamentali"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT symbol, data_type, value
                FROM fundamental_data
                WHERE symbol IN ({})
                AND data_type IN ({})
            '''.format(
                ','.join(['?'] * len(symbols)),
                ','.join(['?'] * len(fields))
            )
            
            if as_of_date:
                query += ' AND timestamp <= ?'
                params = symbols + fields + [as_of_date]
            else:
                params = symbols + fields
                
            df = pd.read_sql_query(query, conn, params=params)
            
        # Pivot table
        if not df.empty:
            df = df.pivot(index='symbol', columns='data_type', values='value')
            
        return df
        
    def add_alternative_data(self,
                           data: pd.DataFrame,
                           data_type: str,
                           source: str) -> None:
        """Aggiunge dati alternativi al database"""
        with sqlite3.connect(self.db_path) as conn:
            data.to_sql(
                'alternative_data',
                conn,
                if_exists='append',
                index=True
            )
            
    def get_data_quality_report(self) -> pd.DataFrame:
        """Genera report sulla qualità dei dati"""
        with sqlite3.connect(self.db_path) as conn:
            # Get data coverage
            coverage = pd.read_sql_query('''
                SELECT symbol,
                       timeframe,
                       COUNT(*) as data_points,
                       MIN(timestamp) as start_date,
                       MAX(timestamp) as end_date,
                       COUNT(DISTINCT source) as sources
                FROM market_data
                GROUP BY symbol, timeframe
            ''', conn)
            
            # Get missing data stats
            missing_stats = pd.read_sql_query('''
                SELECT symbol,
                       timeframe,
                       SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) as missing_open,
                       SUM(CASE WHEN high IS NULL THEN 1 ELSE 0 END) as missing_high,
                       SUM(CASE WHEN low IS NULL THEN 1 ELSE 0 END) as missing_low,
                       SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as missing_close,
                       SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as missing_volume
                FROM market_data
                GROUP BY symbol, timeframe
            ''', conn)
            
        return pd.merge(coverage, missing_stats, on=['symbol', 'timeframe'])
