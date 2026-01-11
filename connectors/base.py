"""
Base connector interface and standard data schemas.

This module defines:
    - Standard DataFrame schemas for intraday and daily data
    - Abstract base class for all data connectors
    - Common utilities for data processing
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DataInterval(Enum):
    """
    Supported OHLCV data intervals.
    """
    MINUTE_1 = "1m"
    MINUTE_2 = "2m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1wk"
    MONTH_1 = "1mo"


class DataSource(Enum):
    """
    Supported data sources.
    """
    POLYGON = "polygon"
    YFINANCE = "yfinance"
    # ALPACA = "alpaca"
    # TIINGO = "tiingo"


@dataclass
class StandardSchema:
    """
    Standard column schema for market data DataFrames.
    
    All connectors must produce DataFrames conforming to this schema.
    """
    
    # Required columns for intraday data
    INTRADAY_COLUMNS: List[str] = field(default_factory=lambda: [
        'open',           # Opening price
        'high',           # High price
        'low',            # Low price
        'close',          # Closing price
        'volume',         # Volume
        'day',            # Trading day (date)
        'vwap',           # Volume-weighted average price
        'move_open',      # Absolute move from open
        'ticker_dvol',    # Daily volatility (rolling)
        'min_from_open',  # Minutes from market open
        'minute_of_day',  # Minute of the trading day
        'move_open_rolling_mean',  # Rolling mean of move from open
        'sigma_open',     # Rolling std of move from open
        'dividend',       # Dividend amount (0 if none)
    ])
    
    # Required columns for daily data
    DAILY_COLUMNS: List[str] = field(default_factory=lambda: [
        'caldt',          # Calendar date
        'open',           # Opening price
        'high',           # High price
        'low',            # Low price
        'close',          # Closing price
        'volume',         # Volume
    ])
    
    # Optional columns that may be present
    OPTIONAL_COLUMNS: List[str] = field(default_factory=lambda: [
        'adj_close',      # Adjusted close price
        'splits',         # Stock split ratio
        'ret',            # Daily return
    ])


@dataclass
class FetchConfig:
    """Configuration for data fetching."""
    ticker: str
    start_date: str = '2019-01-01' # YYYY-MM-DD
    end_date: str = '2025-12-31' # YYYY-MM-DD
    interval: DataInterval = DataInterval.MINUTE_1
    include_prepost: bool = False
    include_dividends: bool = True
    adjust_prices: bool = True


class BaseConnector(ABC):
    """
    Abstract base class for all data connectors.
    
    All data source connectors must inherit from this class and implement
    the required methods to ensure consistent data output.
    
    Example:
        class MyConnector(BaseConnector):
            def fetch_intraday(self, config: FetchConfig) -> pd.DataFrame:
                # Implementation
                pass
            
            def fetch_daily(self, config: FetchConfig) -> pd.DataFrame:
                # Implementation
                pass
    """
    
    # Class-level attributes
    SOURCE: DataSource = None  # Must be set by subclass
    SUPPORTED_INTERVALS: List[DataInterval] = []  # Must be set by subclass
    
    def __init__(self):
        """Initialize the connector."""
        self.schema = StandardSchema()
        self._validate_implementation()
    
    def _validate_implementation(self) -> None:
        """Validate that subclass has set required attributes."""
        if self.SOURCE is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define SOURCE attribute"
            )
        if not self.SUPPORTED_INTERVALS:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define SUPPORTED_INTERVALS"
            )
    
    @abstractmethod
    def fetch_intraday(self, config: FetchConfig) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data.
        
        Args:
            config: Fetch configuration
            
        Returns:
            DataFrame with standardized intraday schema
        """
        pass
    
    @abstractmethod
    def fetch_daily(self, config: FetchConfig) -> pd.DataFrame:
        """
        Fetch daily OHLCV data.
        
        Args:
            config: Fetch configuration
            
        Returns:
            DataFrame with standardized daily schema
        """
        pass
    
    @abstractmethod
    def fetch_dividends(self, ticker: str) -> pd.DataFrame:
        """
        Fetch dividend data.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with 'caldt' and 'dividend' columns
        """
        pass
    
    def validate_interval(self, interval: DataInterval) -> None:
        """
        Validate that the interval is supported.
        
        Args:
            interval: Data interval to validate
            
        Raises:
            ValueError: If interval is not supported
        """
        if interval not in self.SUPPORTED_INTERVALS:
            supported = [i.value for i in self.SUPPORTED_INTERVALS]
            raise ValueError(
                f"Interval {interval.value} not supported by {self.SOURCE.value}. "
                f"Supported intervals: {supported}"
            )
    
    def calculate_vwap(self, group: pd.DataFrame) -> pd.Series:
        """
        Calculate cumulative VWAP for a trading day.
        
        Args:
            group: DataFrame group for a single day
            
        Returns:
            Series of VWAP values
        """
        hlc = (group['high'] + group['low'] + group['close']) / 3
        cum_vol_hlc = (group['volume'] * hlc).cumsum()
        cum_volume = group['volume'].cumsum()
        return cum_vol_hlc / cum_volume.replace(0, np.nan)
    
    def calculate_move_from_open(self, group: pd.DataFrame) -> pd.Series:
        """
        Calculate absolute move from open for a trading day.
        
        Args:
            group: DataFrame group for a single day
            
        Returns:
            Series of absolute move values
        """
        open_price = group['open'].iloc[0]
        if open_price == 0:
            return pd.Series(0, index=group.index)
        return (group['close'] / open_price - 1).abs()
    
    def calculate_daily_metrics(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all daily metrics for a group.
        
        Args:
            group: DataFrame group for a single day
            
        Returns:
            DataFrame with calculated metrics
        """
        group = group.copy()
        group['vwap'] = self.calculate_vwap(group)
        group['move_open'] = self.calculate_move_from_open(group)
        return group
    
    def add_derived_features(
        self,
        df: pd.DataFrame,
        volatility_window: int = 15,
        rolling_window: int = 14
    ) -> pd.DataFrame:
        """
        Add derived features to intraday DataFrame.
        
        This adds:
            - ticker_dvol: Rolling daily volatility
            - min_from_open: Minutes from market open
            - minute_of_day: Minute of day
            - move_open_rolling_mean: Rolling mean of move from open
            - sigma_open: Rolling std of move from open
        
        Args:
            df: Intraday DataFrame with basic OHLCV columns
            volatility_window: Window for volatility calculation
            rolling_window: Window for rolling statistics
            
        Returns:
            DataFrame with derived features added
        """
        df = df.copy()
        
        # Calculate daily returns for volatility
        daily_closes = df.groupby('day')['close'].last()
        daily_returns = daily_closes.pct_change()
        
        # Rolling volatility
        rolling_vol = daily_returns.rolling(
            window=volatility_window, min_periods=volatility_window
        ).std()
        df['ticker_dvol'] = df['day'].map(rolling_vol)
        
        # Minutes from open (assuming 9:30 AM open)
        if df.index.dtype == 'datetime64[ns]' or hasattr(df.index, 'hour'):
            df['min_from_open'] = (
                (df.index - df.index.normalize()) / pd.Timedelta(minutes=1)
            ) - (9 * 60 + 30) + 1
        else:
            # Fallback: assume sequential minutes
            df['min_from_open'] = df.groupby('day').cumcount() + 1
        
        df['minute_of_day'] = df['min_from_open'].round().astype(int)
        
        # Rolling statistics by minute of day
        df['move_open_rolling_mean'] = df.groupby('minute_of_day')['move_open'].transform(
            lambda x: x.shift(1).rolling(window=rolling_window, min_periods=rolling_window-1).mean()
        )
        df['sigma_open'] = df.groupby('minute_of_day')['move_open'].transform(
            lambda x: x.shift(1).rolling(window=rolling_window, min_periods=rolling_window-1).std()
        )
        
        return df
    
    def merge_dividends(
        self,
        df: pd.DataFrame,
        dividends: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge dividend data into intraday DataFrame.
        
        Args:
            df: Intraday DataFrame
            dividends: DataFrame with 'caldt' and 'dividend' columns
            
        Returns:
            DataFrame with dividend column added
        """
        df = df.copy()
        
        if dividends.empty:
            df['dividend'] = 0.0
            return df
        
        # Ensure consistent date format
        dividends = dividends.copy()
        dividends['day'] = pd.to_datetime(dividends['caldt']).dt.date
        
        # Reset index if needed for merge
        if 'day' not in df.columns:
            df['day'] = pd.to_datetime(df.index).date
        
        df_with_day = df.reset_index()
        df_with_day = df_with_day.merge(
            dividends[['day', 'dividend']], on='day', how='left'
        )
        df_with_day['dividend'] = df_with_day['dividend'].fillna(0)
        
        # Restore index
        if 'caldt' in df_with_day.columns:
            df_with_day = df_with_day.set_index('caldt')
        elif 'index' in df_with_day.columns:
            df_with_day = df_with_day.set_index('index')
        
        return df_with_day
    
    def validate_output(
        self,
        df: pd.DataFrame,
        data_type: str = 'intraday'
    ) -> bool:
        """
        Validate that output DataFrame conforms to schema.
        
        Args:
            df: DataFrame to validate
            data_type: 'intraday' or 'daily'
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If DataFrame doesn't conform to schema
        """
        if data_type == 'intraday':
            required = self.schema.INTRADAY_COLUMNS
        else:
            required = self.schema.DAILY_COLUMNS
        
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}"
            )
        
        return True


def filter_market_hours(
    df: pd.DataFrame,
    market_open: str = '09:30',
    market_close: str = '15:59'
) -> pd.DataFrame:
    """
    Filter DataFrame to market hours only.
    
    Args:
        df: DataFrame with datetime index
        market_open: Market open time (HH:MM)
        market_close: Market close time (HH:MM)
        
    Returns:
        Filtered DataFrame
    """
    open_time = datetime.strptime(market_open, '%H:%M').time()
    close_time = datetime.strptime(market_close, '%H:%M').time()
    
    if hasattr(df.index, 'time'):
        mask = (df.index.time >= open_time) & (df.index.time <= close_time)
        return df[mask]
    
    return df
