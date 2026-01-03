"""
Yahoo Finance data connector via yfinance.

This module provides data fetching from Yahoo Finance:
    - Intraday minute-level data (limited to last 7-60 days depending on interval)
    - Daily OHLCV data (full history available)
    - Dividend data

No API key required - uses the free yfinance library.

Note:
    Yahoo Finance has limitations on intraday data:
    - 1m data: last 7 days only
    - 2m-60m data: last 60 days only
    - Daily and higher: full history
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Tuple, Optional, List

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None

from .base import (
    BaseConnector,
    DataSource,
    DataInterval,
    FetchConfig,
    filter_market_hours,
)

logger = logging.getLogger(__name__)


class YFinanceConnector(BaseConnector):
    """
    Yahoo Finance data connector using yfinance library.
    
    Fetches market data from Yahoo Finance via the yfinance package.
    No API key required.
    
    Example:
        connector = YFinanceConnector()
        
        config = FetchConfig(
            ticker='NVDA',
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        
        df_intraday = connector.fetch_intraday(config)
        df_daily = connector.fetch_daily(config)
    
    Limitations:
        - 1-minute data: Only available for last 7 days
        - 2-60 minute data: Only available for last 60 days
        - Daily data: Full history available
        - Pre/post market data available but disabled by default
    """
    
    SOURCE = DataSource.YFINANCE
    SUPPORTED_INTERVALS = [
        DataInterval.MINUTE_1,
        DataInterval.MINUTE_2,
        DataInterval.MINUTE_5,
        DataInterval.MINUTE_15,
        DataInterval.MINUTE_30,
        DataInterval.HOUR_1,
        DataInterval.DAY_1,
        DataInterval.WEEK_1,
        DataInterval.MONTH_1,
    ]
    
    # yfinance interval string mapping
    INTERVAL_MAP = {
        DataInterval.MINUTE_1: '1m',
        DataInterval.MINUTE_2: '2m',
        DataInterval.MINUTE_5: '5m',
        DataInterval.MINUTE_15: '15m',
        DataInterval.MINUTE_30: '30m',
        DataInterval.HOUR_1: '1h',
        DataInterval.DAY_1: '1d',
        DataInterval.WEEK_1: '1wk',
        DataInterval.MONTH_1: '1mo',
    }
    
    # Maximum lookback for intraday data
    INTRADAY_LIMITS = {
        DataInterval.MINUTE_1: 7,    # 7 days
        DataInterval.MINUTE_2: 60,   # 60 days
        DataInterval.MINUTE_5: 60,
        DataInterval.MINUTE_15: 60,
        DataInterval.MINUTE_30: 60,
        DataInterval.HOUR_1: 730,    # ~2 years
    }
    
    def __init__(self, progress: bool = False):
        """
        Initialize the YFinanceConnector.
        
        Args:
            progress: Show download progress bar
            
        Raises:
            ImportError: If yfinance is not installed
        """
        if not HAS_YFINANCE:
            raise ImportError(
                "yfinance package required. Install with: pip install yfinance"
            )
        
        super().__init__()
        self.progress = progress
    
    def _get_ticker(self, symbol: str) -> 'yf.Ticker':
        """Get yfinance Ticker object."""
        return yf.Ticker(symbol)
    
    def _adjust_date_range_for_interval(
        self,
        start_date: str,
        end_date: str,
        interval: DataInterval
    ) -> Tuple[str, str]:
        """
        Adjust date range based on yfinance limitations.
        
        Args:
            start_date: Requested start date
            end_date: Requested end date
            interval: Data interval
            
        Returns:
            Adjusted (start_date, end_date) tuple
        """
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Check if interval has a limit
        max_days = self.INTRADAY_LIMITS.get(interval)
        
        if max_days:
            earliest_allowed = end_dt - timedelta(days=max_days)
            if start_dt < earliest_allowed:
                logger.warning(
                    f"Adjusting start date from {start_date} to "
                    f"{earliest_allowed.strftime('%Y-%m-%d')} due to "
                    f"yfinance {interval.value} data limitation ({max_days} days max)"
                )
                start_dt = earliest_allowed
        
        return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')
    
    def fetch_intraday(self, config: FetchConfig) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data.
        
        Args:
            config: Fetch configuration
            
        Returns:
            DataFrame with standardized intraday schema
        """
        self.validate_interval(config.interval)
        
        # Adjust dates for yfinance limitations
        start_date, end_date = self._adjust_date_range_for_interval(
            config.start_date,
            config.end_date,
            config.interval
        )
        
        interval_str = self.INTERVAL_MAP[config.interval]
        
        logger.info(f"Fetching intraday data for {config.ticker} from yfinance...")
        
        ticker = self._get_ticker(config.ticker)
        
        try:
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval_str,
                prepost=config.include_prepost,
                actions=False,  # We'll fetch dividends separately
                auto_adjust=config.adjust_prices,
            )
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            return pd.DataFrame()
        
        if df.empty:
            logger.warning(f"No intraday data found for {config.ticker}")
            return df
        
        # Standardize column names (yfinance uses capitalized names)
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
        })
        
        # Filter to market hours if not including pre/post
        if not config.include_prepost:
            df = filter_market_hours(df)
        
        if df.empty:
            logger.warning(f"No market hours data for {config.ticker}")
            return df
        
        # Add day column
        df['day'] = df.index.date
        
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
        logger.info(f"Fetching daily data for {config.ticker} from yfinance...")
        
        ticker = self._get_ticker(config.ticker)
        
        try:
            df = ticker.history(
                start=config.start_date,
                end=config.end_date,
                interval='1d',
                actions=False,
                auto_adjust=config.adjust_prices,
            )
        except Exception as e:
            logger.error(f"Error fetching daily data from yfinance: {e}")
            return pd.DataFrame()
        
        if df.empty:
            logger.warning(f"No daily data found for {config.ticker}")
            return df
        
        # Standardize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
        })
        
        # Add caldt column (matching Polygon schema)
        df['caldt'] = df.index
        
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
        logger.info(f"Fetching dividends for {ticker} from yfinance...")
        
        yf_ticker = self._get_ticker(ticker)
        
        try:
            dividends = yf_ticker.dividends
        except Exception as e:
            logger.warning(f"Could not fetch dividends: {e}")
            return pd.DataFrame(columns=['caldt', 'dividend'])
        
        if dividends.empty:
            return pd.DataFrame(columns=['caldt', 'dividend'])
        
        # Convert to standard format
        df = pd.DataFrame({
            'caldt': dividends.index,
            'dividend': dividends.values
        })
        
        logger.info(f"Found {len(df)} dividend records for {ticker}")
        
        return df
    
    def fetch_splits(self, ticker: str) -> pd.DataFrame:
        """
        Fetch stock split data.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            DataFrame with split dates and ratios
        """
        logger.info(f"Fetching splits for {ticker} from yfinance...")
        
        yf_ticker = self._get_ticker(ticker)
        
        try:
            splits = yf_ticker.splits
        except Exception as e:
            logger.warning(f"Could not fetch splits: {e}")
            return pd.DataFrame(columns=['caldt', 'split_ratio'])
        
        if splits.empty:
            return pd.DataFrame(columns=['caldt', 'split_ratio'])
        
        df = pd.DataFrame({
            'caldt': splits.index,
            'split_ratio': splits.values
        })
        
        return df
    
    def fetch_info(self, ticker: str) -> dict:
        """
        Fetch ticker info/metadata.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dictionary with ticker information
        """
        yf_ticker = self._get_ticker(ticker)
        
        try:
            return yf_ticker.info
        except Exception as e:
            logger.warning(f"Could not fetch info: {e}")
            return {}
    
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
            start_date: Start date (default: 60 days ago for intraday)
            end_date: End date (default: today)
            
        Returns:
            Tuple of (intraday_df, daily_df)
        """
        today = date.today()
        
        # Default to 60 days for intraday (max for most intervals)
        if start_date is None:
            start_date = (today - timedelta(days=60)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = today.strftime('%Y-%m-%d')
        
        config = FetchConfig(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=DataInterval.MINUTE_1,
            include_dividends=True
        )
        
        # For daily data, we can go further back
        daily_config = FetchConfig(
            ticker=ticker,
            start_date='2020-01-01',  # Go back further for daily
            end_date=end_date,
            interval=DataInterval.DAY_1,
            include_dividends=False
        )
        
        df_intraday = self.fetch_intraday(config)
        df_daily = self.fetch_daily(daily_config)
        
        return df_intraday, df_daily
    
    @staticmethod
    def download_multiple(
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1d',
        threads: bool = True
    ) -> pd.DataFrame:
        """
        Download data for multiple tickers at once.
        
        Uses yfinance's optimized multi-ticker download.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            threads: Use multithreading
            
        Returns:
            DataFrame with MultiIndex columns (ticker, field)
        """
        if not HAS_YFINANCE:
            raise ImportError("yfinance package required")
        
        return yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by='ticker',
            threads=threads,
            progress=False
        )
