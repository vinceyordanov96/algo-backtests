"""
Data management module for saving and loading market data.

This module provides:
    - DataManager: Central class for data persistence and retrieval
    - Consistent naming conventions for saved data
    - Support for resampling and date-based loading

Directory Structure:
    saved-data/
    ├── intraday/
    │   └── {ticker}/
    │       └── {ticker}_intraday_YYYYMMDD_HHMMSS.csv
    ├── daily/
    │   └── {ticker}/
    │       └── {ticker}_daily_YYYYMMDD_HHMMSS.csv
    └── dividends/
        └── {ticker}/
            └── {ticker}_dividends_YYYYMMDD_HHMMSS.csv

Example:
    from connectors import DataManager, DataFetcher
    
    # Fetch and save data
    fetcher = DataFetcher('NVDA', source='yfinance')
    df_intraday, df_daily = fetcher.process_data()
    
    manager = DataManager()
    manager.save_intraday('NVDA', df_intraday)
    manager.save_daily('NVDA', df_daily)
    
    # Load latest data
    df_intraday = manager.load_intraday('NVDA')
    
    # Load with resampling
    df_5min = manager.load_intraday_resampled('NVDA', interval='5min')
    
    # Load from specific date
    df_recent = manager.load_intraday_from_date('NVDA', start_date='2024-06-01')
"""

import os
import glob
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataManager:
    """
    Central manager for market data persistence and retrieval.
    
    Provides consistent file naming, directory organization, and
    various loading options including resampling and date filtering.
    
    Attributes:
        base_dir: Root directory for saved data
        intraday_dir: Directory for intraday data
        daily_dir: Directory for daily data
        dividends_dir: Directory for dividend data
    
    Example:
        manager = DataManager(base_dir='saved-data')
        
        # Save data
        manager.save_intraday('AAPL', df_intraday)
        manager.save_daily('AAPL', df_daily)
        
        # Load latest
        df = manager.load_intraday('AAPL')
        
        # Load with 5-minute resampling
        df_5min = manager.load_intraday_resampled('AAPL', interval='5min')
        
        # Load from specific date
        df = manager.load_intraday_from_date('AAPL', start_date='2024-01-01')
    """
    
    # Standard column schemas
    INTRADAY_COLUMNS = [
        'open', 'high', 'low', 'close', 'volume', 'day', 'vwap',
        'move_open', 'ticker_dvol', 'min_from_open', 'minute_of_day',
        'move_open_rolling_mean', 'sigma_open', 'dividend'
    ]
    
    DAILY_COLUMNS = ['caldt', 'open', 'high', 'low', 'close', 'volume']
    
    DIVIDEND_COLUMNS = ['caldt', 'dividend']
    

    def __init__(self, base_dir: str = 'outputs/data'):
        """
        Initialize the DataManager.
        
        Args:
            base_dir: Root directory for all saved data
        """
        self.base_dir = Path(base_dir)
        self.intraday_dir = self.base_dir / 'intraday'
        self.daily_dir = self.base_dir / 'daily'
        self.dividends_dir = self.base_dir / 'dividends'
        
        # Create directories if they don't exist
        self._ensure_directories()
    

    def _ensure_directories(self) -> None:
        """Create base directories if they don't exist."""
        for directory in [self.intraday_dir, self.daily_dir, self.dividends_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    

    def _get_ticker_dir(self, data_type: str, ticker: str) -> Path:
        """
        Get the directory for a specific ticker and data type.
        
        Args:
            data_type: 'intraday', 'daily', or 'dividends'
            ticker: Stock ticker symbol
            
        Returns:
            Path to ticker directory
        """
        base = {
            'intraday': self.intraday_dir,
            'daily': self.daily_dir,
            'dividends': self.dividends_dir
        }[data_type]
        
        ticker_dir = base / ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        return ticker_dir
    

    def _generate_filename(self, ticker: str, data_type: str) -> str:
        """
        Generate a filename with timestamp.
        
        Args:
            ticker: Stock ticker symbol
            data_type: 'intraday', 'daily', or 'dividends'
            
        Returns:
            Filename string (e.g., 'NVDA_intraday_20240115_143022.csv')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{ticker.upper()}_{data_type}_{timestamp}.csv"
    

    def _parse_filename_timestamp(self, filename: str) -> Optional[datetime]:
        """
        Parse timestamp from filename.
        
        Args:
            filename: Filename to parse
            
        Returns:
            datetime or None if parsing fails
        """
        try:
            # Extract timestamp portion (YYYYMMDD_HHMMSS)
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 3:
                date_str = parts[-2]
                time_str = parts[-1]
                return datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
        except (ValueError, IndexError):
            pass
        return None
    

    def _get_latest_file(self, ticker: str, data_type: str) -> Optional[Path]:
        """
        Get the most recent file for a ticker and data type.
        
        Args:
            ticker: Stock ticker symbol
            data_type: 'intraday', 'daily', or 'dividends'
            
        Returns:
            Path to latest file or None if no files exist
        """
        ticker_dir = self._get_ticker_dir(data_type, ticker)
        pattern = f"{ticker.upper()}_{data_type}_*.csv"
        files = list(ticker_dir.glob(pattern))
        
        if not files:
            return None
        
        # Sort by timestamp in filename (newest first)
        files_with_times = []
        for f in files:
            ts = self._parse_filename_timestamp(f.name)
            if ts:
                files_with_times.append((f, ts))
        
        if not files_with_times:
            # Fallback to modification time
            return max(files, key=lambda x: x.stat().st_mtime)
        
        files_with_times.sort(key=lambda x: x[1], reverse=True)
        return files_with_times[0][0]
    

    def _get_all_files(self, ticker: str, data_type: str) -> List[Path]:
        """
        Get all files for a ticker and data type, sorted newest first.
        
        Args:
            ticker: Stock ticker symbol
            data_type: 'intraday', 'daily', or 'dividends'
            
        Returns:
            List of file paths sorted by timestamp (newest first)
        """
        ticker_dir = self._get_ticker_dir(data_type, ticker)
        pattern = f"{ticker.upper()}_{data_type}_*.csv"
        files = list(ticker_dir.glob(pattern))
        
        files_with_times = []
        for f in files:
            ts = self._parse_filename_timestamp(f.name)
            if ts:
                files_with_times.append((f, ts))
        
        files_with_times.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in files_with_times]
    

    def save_intraday(
        self,
        ticker: str,
        df: pd.DataFrame,
        validate: bool = True
    ) -> str:
        """
        Save intraday data to CSV.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with intraday data
            validate: Whether to validate columns before saving
            
        Returns:
            Path to saved file
        """
        if validate:
            self._validate_dataframe(df, 'intraday')
        
        ticker_dir = self._get_ticker_dir('intraday', ticker)
        filename = self._generate_filename(ticker, 'intraday')
        filepath = ticker_dir / filename
        
        df.to_csv(filepath)
        logger.info(f"Saved intraday data to: {filepath}")
        
        return str(filepath)
    

    def save_daily(
        self,
        ticker: str,
        df: pd.DataFrame,
        validate: bool = True
    ) -> str:
        """
        Save daily data to CSV.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with daily data
            validate: Whether to validate columns before saving
            
        Returns:
            Path to saved file
        """
        if validate:
            self._validate_dataframe(df, 'daily')
        
        ticker_dir = self._get_ticker_dir('daily', ticker)
        filename = self._generate_filename(ticker, 'daily')
        filepath = ticker_dir / filename
        
        df.to_csv(filepath)
        logger.info(f"Saved daily data to: {filepath}")
        
        return str(filepath)
    

    def save_dividends(
        self,
        ticker: str,
        df: pd.DataFrame,
        validate: bool = True
    ) -> str:
        """
        Save dividend data to CSV.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with dividend data
            validate: Whether to validate columns before saving
            
        Returns:
            Path to saved file
        """
        if validate and not df.empty:
            self._validate_dataframe(df, 'dividends')
        
        ticker_dir = self._get_ticker_dir('dividends', ticker)
        filename = self._generate_filename(ticker, 'dividends')
        filepath = ticker_dir / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"Saved dividend data to: {filepath}")
        
        return str(filepath)
    

    def save_all(
        self,
        ticker: str,
        df_intraday: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_dividends: Optional[pd.DataFrame] = None,
        validate: bool = True
    ) -> Dict[str, str]:
        """
        Save all data types for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            df_intraday: Intraday DataFrame
            df_daily: Daily DataFrame
            df_dividends: Optional dividend DataFrame
            validate: Whether to validate columns
            
        Returns:
            Dictionary mapping data type to saved file path
        """
        paths = {
            'intraday': self.save_intraday(ticker, df_intraday, validate),
            'daily': self.save_daily(ticker, df_daily, validate),
        }
        
        if df_dividends is not None and not df_dividends.empty:
            paths['dividends'] = self.save_dividends(ticker, df_dividends, validate)
        
        return paths
    

    def _validate_dataframe(self, df: pd.DataFrame, data_type: str) -> None:
        """
        Validate DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            data_type: 'intraday', 'daily', or 'dividends'
            
        Raises:
            ValueError: If required columns are missing
        """
        required = {
            'intraday': ['open', 'high', 'low', 'close', 'volume'],
            'daily': ['open', 'high', 'low', 'close', 'volume'],
            'dividends': ['dividend'],
        }
        
        required_cols = required.get(data_type, [])
        missing = set(required_cols) - set(df.columns)
        
        if missing:
            logger.warning(f"DataFrame missing columns for {data_type}: {missing}")
    
    
    def load_intraday(
        self,
        ticker: str,
        version: str = 'latest'
    ) -> Optional[pd.DataFrame]:
        """
        Load intraday data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            version: 'latest' for most recent, or specific filename
            
        Returns:
            DataFrame or None if not found
        """
        if version == 'latest':
            filepath = self._get_latest_file(ticker, 'intraday')
        else:
            ticker_dir = self._get_ticker_dir('intraday', ticker)
            filepath = ticker_dir / version
        
        if filepath is None or not filepath.exists():
            logger.warning(f"No intraday data found for {ticker}")
            return None
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} intraday records for {ticker}")
        return df
    

    def load_daily(
        self,
        ticker: str,
        version: str = 'latest'
    ) -> Optional[pd.DataFrame]:
        """
        Load daily data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            version: 'latest' for most recent, or specific filename
            
        Returns:
            DataFrame or None if not found
        """
        if version == 'latest':
            filepath = self._get_latest_file(ticker, 'daily')
        else:
            ticker_dir = self._get_ticker_dir('daily', ticker)
            filepath = ticker_dir / version
        
        if filepath is None or not filepath.exists():
            logger.warning(f"No daily data found for {ticker}")
            return None
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} daily records for {ticker}")
        return df
    

    def load_dividends(
        self,
        ticker: str,
        version: str = 'latest'
    ) -> Optional[pd.DataFrame]:
        """
        Load dividend data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            version: 'latest' for most recent, or specific filename
            
        Returns:
            DataFrame or None if not found
        """
        if version == 'latest':
            filepath = self._get_latest_file(ticker, 'dividends')
        else:
            ticker_dir = self._get_ticker_dir('dividends', ticker)
            filepath = ticker_dir / version
        
        if filepath is None or not filepath.exists():
            logger.warning(f"No dividend data found for {ticker}")
            return None
        
        df = pd.read_csv(filepath, parse_dates=['caldt'])
        logger.info(f"Loaded {len(df)} dividend records for {ticker}")
        return df
    

    def load_all(
        self,
        ticker: str,
        version: str = 'latest'
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load all data types for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            version: 'latest' for most recent files
            
        Returns:
            Tuple of (intraday_df, daily_df, dividends_df)
        """
        return (
            self.load_intraday(ticker, version),
            self.load_daily(ticker, version),
            self.load_dividends(ticker, version)
        )
    

    def load_intraday_resampled(
        self,
        ticker: str,
        interval: str = '5min',
        version: str = 'latest',
        agg_dict: Optional[Dict[str, str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load intraday data and resample to a different frequency.
        
        Useful for ML feature generation or strategy testing at
        different time intervals.
        
        Args:
            ticker: Stock ticker symbol
            interval: Resample interval (e.g., '5min', '15min', '30min', '1H')
            version: 'latest' or specific filename
            agg_dict: Custom aggregation dictionary. If None, uses default OHLCV aggregation.
            
        Returns:
            Resampled DataFrame or None if not found
            
        Example:
            # Load 5-minute bars
            df_5min = manager.load_intraday_resampled('NVDA', interval='5min')
            
            # Load 15-minute bars with custom aggregation
            df_15min = manager.load_intraday_resampled(
                'NVDA',
                interval='15min',
                agg_dict={'close': 'last', 'volume': 'sum', 'vwap': 'mean'}
            )
        """
        df = self.load_intraday(ticker, version)
        
        if df is None or df.empty:
            return None
        
        # Default OHLCV aggregation
        if agg_dict is None:
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }
            
            # Add optional columns if present
            if 'vwap' in df.columns:
                agg_dict['vwap'] = 'last'
            if 'dividend' in df.columns:
                agg_dict['dividend'] = 'sum'
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Resample
        df_resampled = df.resample(interval).agg(agg_dict)
        
        # Drop rows with no data (outside market hours)
        df_resampled = df_resampled.dropna(subset=['close'])
        
        # Recalculate day column if present
        if 'day' in df.columns or True:
            df_resampled['day'] = df_resampled.index.date
        
        # Recalculate min_from_open
        df_resampled['min_from_open'] = (
            (df_resampled.index - df_resampled.index.normalize()) 
            / pd.Timedelta(minutes=1)
        ) - (9 * 60 + 30) + 1
        
        logger.info(
            f"Resampled {ticker} from 1min to {interval}: "
            f"{len(df)} -> {len(df_resampled)} records"
        )
        
        return df_resampled
    

    def load_daily_resampled(
        self,
        ticker: str,
        interval: str = 'W',
        version: str = 'latest',
        agg_dict: Optional[Dict[str, str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load daily data and resample to a different frequency.
        
        Args:
            ticker: Stock ticker symbol
            interval: Resample interval (e.g., 'W' for weekly, 'M' for monthly)
            version: 'latest' or specific filename
            agg_dict: Custom aggregation dictionary
            
        Returns:
            Resampled DataFrame or None if not found
        """
        df = self.load_daily(ticker, version)
        
        if df is None or df.empty:
            return None
        
        if agg_dict is None:
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df_resampled = df.resample(interval).agg(agg_dict).dropna()
        
        logger.info(
            f"Resampled {ticker} daily to {interval}: "
            f"{len(df)} -> {len(df_resampled)} records"
        )
        
        return df_resampled

    
    def load_intraday_from_date(
        self,
        ticker: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        version: str = 'latest'
    ) -> Optional[pd.DataFrame]:
        """
        Load intraday data filtered by date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (inclusive). If None, no lower bound.
            end_date: End date (inclusive). If None, no upper bound.
            version: 'latest' or specific filename
            
        Returns:
            Filtered DataFrame or None if not found
            
        Example:
            # Load data from June 2024 onwards
            df = manager.load_intraday_from_date('NVDA', start_date='2024-06-01')
            
            # Load data for Q3 2024
            df = manager.load_intraday_from_date(
                'NVDA',
                start_date='2024-07-01',
                end_date='2024-09-30'
            )
        """
        df = self.load_intraday(ticker, version)
        
        if df is None or df.empty:
            return None
        
        return self._filter_by_date(df, start_date, end_date)
    

    def load_daily_from_date(
        self,
        ticker: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        version: str = 'latest'
    ) -> Optional[pd.DataFrame]:
        """
        Load daily data filtered by date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            version: 'latest' or specific filename
            
        Returns:
            Filtered DataFrame or None if not found
        """
        df = self.load_daily(ticker, version)
        
        if df is None or df.empty:
            return None
        
        return self._filter_by_date(df, start_date, end_date)
    

    def load_intraday_from_index(
        self,
        ticker: str,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        version: str = 'latest'
    ) -> Optional[pd.DataFrame]:
        """
        Load intraday data filtered by row index.
        
        Useful for train/test splits or loading specific portions of data.
        
        Args:
            ticker: Stock ticker symbol
            start_idx: Starting row index (0-based, inclusive)
            end_idx: Ending row index (exclusive)
            version: 'latest' or specific filename
            
        Returns:
            Filtered DataFrame or None if not found
            
        Example:
            # Load first 1000 rows
            df = manager.load_intraday_from_index('NVDA', start_idx=0, end_idx=1000)
            
            # Load last 500 rows
            df = manager.load_intraday_from_index('NVDA', start_idx=-500)
        """
        df = self.load_intraday(ticker, version)
        
        if df is None or df.empty:
            return None
        
        return df.iloc[start_idx:end_idx]
    

    def load_intraday_last_n_days(
        self,
        ticker: str,
        n_days: int = 30,
        version: str = 'latest'
    ) -> Optional[pd.DataFrame]:
        """
        Load intraday data for the last N trading days.
        
        Args:
            ticker: Stock ticker symbol
            n_days: Number of trading days to load
            version: 'latest' or specific filename
            
        Returns:
            Filtered DataFrame or None if not found
            
        Example:
            # Load last 30 trading days
            df = manager.load_intraday_last_n_days('NVDA', n_days=30)
        """
        df = self.load_intraday(ticker, version)
        
        if df is None or df.empty:
            return None
        
        # Get unique trading days
        if 'day' in df.columns:
            days = df['day'].unique()
        else:
            days = pd.to_datetime(df.index).date
            days = np.unique(days)
        
        if len(days) <= n_days:
            return df
        
        # Get last n days
        cutoff_day = sorted(days)[-n_days]
        
        if 'day' in df.columns:
            return df[df['day'] >= cutoff_day]
        else:
            return df[pd.to_datetime(df.index).date >= cutoff_day]
    

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: Optional[Union[str, date, datetime]],
        end_date: Optional[Union[str, date, datetime]]
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            df: DataFrame to filter
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Filtered DataFrame
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        # Parse dates
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            elif isinstance(start_date, date):
                start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            elif isinstance(end_date, date):
                end_date = pd.to_datetime(end_date)
            # Make end_date inclusive (include full day)
            end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df.index <= end_date]
        
        original_len = len(df)
        logger.info(f"Filtered to {len(df)} records (date range filter)")
        
        return df
    
    
    def list_tickers(self, data_type: str = 'intraday') -> List[str]:
        """
        List all tickers with saved data.
        
        Args:
            data_type: 'intraday', 'daily', or 'dividends'
            
        Returns:
            List of ticker symbols
        """
        base = {
            'intraday': self.intraday_dir,
            'daily': self.daily_dir,
            'dividends': self.dividends_dir
        }[data_type]
        
        if not base.exists():
            return []
        
        return [d.name for d in base.iterdir() if d.is_dir()]
    

    def list_versions(self, ticker: str, data_type: str = 'intraday') -> List[Dict[str, Any]]:
        """
        List all saved versions for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            data_type: 'intraday', 'daily', or 'dividends'
            
        Returns:
            List of dicts with filename and timestamp info
        """
        files = self._get_all_files(ticker, data_type)
        
        versions = []
        for f in files:
            ts = self._parse_filename_timestamp(f.name)
            versions.append({
                'filename': f.name,
                'path': str(f),
                'timestamp': ts,
                'size_mb': f.stat().st_size / (1024 * 1024)
            })
        
        return versions
    

    def get_data_info(self, ticker: str, data_type: str = 'intraday') -> Optional[Dict[str, Any]]:
        """
        Get information about the latest saved data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            data_type: 'intraday', 'daily', or 'dividends'
            
        Returns:
            Dictionary with data info or None if not found
        """
        filepath = self._get_latest_file(ticker, data_type)
        
        if filepath is None:
            return None
        
        # Quick peek at the data
        df = pd.read_csv(filepath, index_col=0, parse_dates=True, nrows=5)
        df_tail = pd.read_csv(filepath, index_col=0, parse_dates=True).tail(5)
        
        # Get full row count
        with open(filepath, 'r') as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header
        
        return {
            'ticker': ticker,
            'data_type': data_type,
            'filepath': str(filepath),
            'filename': filepath.name,
            'timestamp': self._parse_filename_timestamp(filepath.name),
            'row_count': row_count,
            'columns': list(df.columns),
            'date_range': {
                'start': str(df.index[0]) if len(df) > 0 else None,
                'end': str(df_tail.index[-1]) if len(df_tail) > 0 else None,
            },
            'size_mb': filepath.stat().st_size / (1024 * 1024)
        }
    

    def cleanup_old_versions(
        self,
        ticker: str,
        data_type: str = 'intraday',
        keep_n: int = 3
    ) -> int:
        """
        Remove old versions, keeping only the N most recent.
        
        Args:
            ticker: Stock ticker symbol
            data_type: 'intraday', 'daily', or 'dividends'
            keep_n: Number of versions to keep
            
        Returns:
            Number of files deleted
        """
        files = self._get_all_files(ticker, data_type)
        
        if len(files) <= keep_n:
            return 0
        
        files_to_delete = files[keep_n:]
        deleted = 0
        
        for f in files_to_delete:
            try:
                f.unlink()
                deleted += 1
                logger.info(f"Deleted old version: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {f.name}: {e}")
        
        return deleted
