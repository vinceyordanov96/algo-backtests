"""
Polygon.io data connector.

This module provides data fetching from the Polygon.io API:
    - Intraday minute-level data
    - Daily OHLCV data
    - Dividend data

Requires POLYGON_API_KEY environment variable or explicit API key.
"""

import os
import time
import pytz
import logging
import requests
import pandas as pd
from datetime import datetime, date
from typing import Tuple

from .base import (
    BaseConnector,
    DataSource,
    DataInterval,
    FetchConfig,
)

logger = logging.getLogger(__name__)


class PolygonConnector(BaseConnector):
    """
    Polygon.io data connector.
    
    Fetches market data from the Polygon.io REST API.
    
    Example:
        connector = PolygonConnector(api_key='your_key')
        
        config = FetchConfig(
            ticker='NVDA',
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        
        df_intraday = connector.fetch_intraday(config)
        df_daily = connector.fetch_daily(config)
    
    Note:
        Polygon.io has rate limits. The connector automatically handles
        rate limiting by waiting when limits are reached.
    """
    
    SOURCE = DataSource.POLYGON
    SUPPORTED_INTERVALS = [
        DataInterval.MINUTE_1,
        DataInterval.MINUTE_5,
        DataInterval.MINUTE_15,
        DataInterval.MINUTE_30,
        DataInterval.HOUR_1,
        DataInterval.DAY_1,
        DataInterval.WEEK_1,
        DataInterval.MONTH_1,
    ]
    
    # API configuration
    BASE_URL = 'https://api.polygon.io'
    DEFAULT_LIMIT = '50000'
    DEFAULT_TIMEZONE = 'America/New_York'
    
    # Interval mapping to Polygon API format
    INTERVAL_MAP = {
        DataInterval.MINUTE_1: ('1', 'minute'),
        DataInterval.MINUTE_5: ('5', 'minute'),
        DataInterval.MINUTE_15: ('15', 'minute'),
        DataInterval.MINUTE_30: ('30', 'minute'),
        DataInterval.HOUR_1: ('1', 'hour'),
        DataInterval.DAY_1: ('1', 'day'),
        DataInterval.WEEK_1: ('1', 'week'),
        DataInterval.MONTH_1: ('1', 'month'),
    }
    
    def __init__(
        self,
        api_key: str = None,
        enforce_rate_limit: bool = False,
        requests_per_minute: int = 5
    ):
        """
        Initialize the PolygonConnector.
        
        Args:
            api_key: Polygon API key (default: from POLYGON_API_KEY env var)
            enforce_rate_limit: Whether to enforce API rate limits
            requests_per_minute: Max requests per minute (free tier = 5)
        """
        super().__init__()
        
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Polygon API key required. Set POLYGON_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.enforce_rate_limit = enforce_rate_limit
        self.requests_per_minute = requests_per_minute
        self.eastern = pytz.timezone(self.DEFAULT_TIMEZONE)
    
    def _fetch_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        multiplier: str,
        timespan: str
    ) -> pd.DataFrame:
        """
        Fetch data from Polygon API with pagination.
        
        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            multiplier: Interval multiplier (e.g., '1', '5')
            timespan: Timespan type ('minute', 'hour', 'day', etc.)
            
        Returns:
            DataFrame with raw OHLCV data
        """
        url = (
            f'{self.BASE_URL}'
            f'/v2/aggs/ticker/{ticker}'
            f'/range/{multiplier}'
            f'/{timespan}'
            f'/{start_date}'
            f'/{end_date}'
            f'?adjusted=true&sort=asc&limit={self.DEFAULT_LIMIT}&apiKey={self.api_key}'
        )
        
        data_list = []
        request_count = 0
        first_request_time = None
        is_intraday = timespan == 'minute'
        page_count = 0
        total_raw_records = 0
        fetch_start_time = time.time()
        
        logger.info(f"Starting {timespan} data fetch for {ticker} ({start_date} to {end_date})")
        
        while True:
            # Rate limiting
            if self.enforce_rate_limit and request_count >= self.requests_per_minute:
                elapsed_time = time.time() - first_request_time
                if elapsed_time < 60:
                    wait_time = 60 - elapsed_time
                    logger.info(f"Rate limit reached ({self.requests_per_minute} requests/min). Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                request_count = 0
                first_request_time = time.time()
            
            if first_request_time is None and self.enforce_rate_limit:
                first_request_time = time.time()
            
            # Make request
            response = requests.get(url)
            
            if response.status_code != 200:
                error_data = response.json()
                error_message = error_data.get('error', 'Unknown error')
                
                if 'exceeded the maximum requests' in str(error_message).lower():
                    logger.warning("Rate limit exceeded by API. Waiting 60s...")
                    time.sleep(60)
                    request_count = 0
                    continue
                else:
                    logger.error(f"API error: {error_message}")
                    break
            
            data = response.json()
            request_count += 1
            page_count += 1
            
            results = data.get('results', [])
            total_raw_records += len(results)
            
            # Log progress for each page
            logger.info(f"Page {page_count}: Fetched {len(results)} records (Total raw: {total_raw_records:,})")
            
            # Process results
            records_added = 0
            for entry in results:
                utc_time = datetime.fromtimestamp(entry['t'] / 1000, pytz.utc)
                eastern_time = utc_time.astimezone(self.eastern)
                
                data_entry = {
                    'volume': entry['v'],
                    'open': entry['o'],
                    'high': entry['h'],
                    'low': entry['l'],
                    'close': entry['c'],
                    'caldt': eastern_time.replace(tzinfo=None)
                }
                
                # Filter market hours for intraday
                if is_intraday:
                    if (eastern_time.time() >= datetime.strptime('09:30', '%H:%M').time() and
                        eastern_time.time() <= datetime.strptime('15:59', '%H:%M').time()):
                        data_list.append(data_entry)
                        records_added += 1
                else:
                    data_list.append(data_entry)
                    records_added += 1
            
            # Check for pagination
            if 'next_url' in data and data['next_url']:
                url = data['next_url'] + '&apiKey=' + self.api_key
            else:
                break
        
        elapsed = time.time() - fetch_start_time
        logger.info(
            f"Fetch complete for {ticker} {timespan}: "
            f"{len(data_list):,} records in {elapsed:.1f}s "
            f"({page_count} API calls)"
        )
        
        return pd.DataFrame(data_list)
    
    def fetch_intraday(self, config: FetchConfig) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data.
        
        Args:
            config: Fetch configuration
            
        Returns:
            DataFrame with standardized intraday schema
        """
        self.validate_interval(config.interval)
        
        multiplier, timespan = self.INTERVAL_MAP.get(
            config.interval, ('1', 'minute')
        )
        
        logger.info(f"Fetching intraday data for {config.ticker}...")
        df = self._fetch_data(
            config.ticker,
            config.start_date,
            config.end_date,
            multiplier,
            timespan
        )
        
        if df.empty:
            logger.warning(f"No intraday data found for {config.ticker}")
            return df
        
        # Process data
        df['day'] = pd.to_datetime(df['caldt']).dt.date
        df.set_index('caldt', inplace=True)
        
        # Calculate daily metrics
        df = df.groupby('day', group_keys=False).apply(self.calculate_daily_metrics)
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Fetch and merge dividends
        if config.include_dividends:
            dividends = self.fetch_dividends(config.ticker)
            df = self.merge_dividends(df, dividends)
        else:
            df['dividend'] = 0.0
        
        logger.info(f"Processed {len(df)} intraday records for {config.ticker}")
        
        return df
    
    def fetch_daily(self, config: FetchConfig) -> pd.DataFrame:
        """
        Fetch daily OHLCV data.
        
        Args:
            config: Fetch configuration
            
        Returns:
            DataFrame with standardized daily schema
        """
        logger.info(f"Fetching daily data for {config.ticker}...")
        
        df = self._fetch_data(
            config.ticker,
            config.start_date,
            config.end_date,
            '1',
            'day'
        )
        
        if df.empty:
            logger.warning(f"No daily data found for {config.ticker}")
            return df
        
        # Standardize column names
        df = df.rename(columns={'caldt': 'caldt'})
        
        logger.info(f"Fetched {len(df)} daily records for {config.ticker}")
        
        return df
    
    def fetch_dividends(self, ticker: str) -> pd.DataFrame:
        """
        Fetch dividend data.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            DataFrame with 'caldt' and 'dividend' columns
        """
        url = (
            f'{self.BASE_URL}/v3/reference/dividends'
            f'?ticker={ticker}&limit=1000&apiKey={self.api_key}'
        )
        
        dividends_list = []
        
        while True:
            response = requests.get(url)
            data = response.json()
            
            if 'results' in data:
                for entry in data['results']:
                    dividends_list.append({
                        'caldt': datetime.strptime(
                            entry['ex_dividend_date'], '%Y-%m-%d'
                        ),
                        'dividend': entry['cash_amount']
                    })
            
            if 'next_url' in data and data['next_url']:
                url = data['next_url'] + '&apiKey=' + self.api_key
            else:
                break
        
        return pd.DataFrame(dividends_list)
    
    def process_data(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convenience method to fetch and process all data.
        
        This is a drop-in replacement for the old DataFetcher.process_data().
        
        Args:
            ticker: Stock ticker
            start_date: Start date (default: 2024-01-01)
            end_date: End date (default: today)
            
        Returns:
            Tuple of (intraday_df, daily_df)
        """
        config = FetchConfig(
            ticker=ticker,
            start_date='2021-02-01', #start_date or '2024-01-01',
            end_date='2025-12-31', #end_date or str(date.today()),
            interval=DataInterval.MINUTE_1,
            include_dividends=True
        )
        
        df_intraday = self.fetch_intraday(config)
        df_daily = self.fetch_daily(config)
        
        return df_intraday, df_daily


# Backwards compatibility alias
DataFetcher = PolygonConnector
