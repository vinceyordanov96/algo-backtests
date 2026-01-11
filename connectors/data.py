"""
Unified data fetching interface.

This module provides a single entry point for fetching market data from
any supported data source. It automatically routes requests to the 
appropriate connector and ensures consistent output schema.

Example:
    from connectors import DataFetcher
    
    # Using default source (Polygon)
    fetcher = DataFetcher('NVDA')
    df_intraday, df_daily = fetcher.process_data()
    
    # Using specific source
    fetcher = DataFetcher('NVDA', source='yfinance')
    df_intraday, df_daily = fetcher.process_data()
    
    # With custom date range
    fetcher = DataFetcher(
        ticker='AAPL',
        source='polygon',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    df_intraday, df_daily = fetcher.process_data()
"""

import os
import logging
import pandas as pd
from dotenv import load_dotenv
from datetime import date, datetime
from typing import Tuple, Optional, Union, Dict, Any

from .base import (
    BaseConnector,
    DataSource,
    DataInterval,
    FetchConfig,
    StandardSchema,
)
from .polygon import PolygonConnector
from .yfinance import YFinanceConnector

logger = logging.getLogger(__name__)

load_dotenv()

# Registry of available connectors
CONNECTORS: Dict[DataSource, type] = {
    DataSource.POLYGON: PolygonConnector,
    DataSource.YFINANCE: YFinanceConnector,
}


def get_connector(
    source: Union[str, DataSource],
    **kwargs
) -> BaseConnector:
    """
    Get a connector instance for the specified data source.
    
    Args:
        source: Data source ('polygon', 'yfinance', or DataSource enum)
        **kwargs: Additional arguments passed to connector constructor
        
    Returns:
        Connector instance
        
    Raises:
        ValueError: If source is not supported
    """
    if isinstance(source, str):
        source = DataSource(source.lower())
    
    connector_class = CONNECTORS.get(source)
    
    if connector_class is None:
        available = [s.value for s in CONNECTORS.keys()]
        raise ValueError(
            f"Unknown data source: {source}. Available sources: {available}"
        )
    
    return connector_class(**kwargs)


class DataFetcher:
    """
    Unified data fetcher that routes to appropriate connector.
    
    This is the main entry point for fetching market data. It provides
    a consistent interface regardless of the underlying data source.
    
    Example:
        # Default (Polygon)
        fetcher = DataFetcher('NVDA')
        df_intra, df_daily = fetcher.process_data()
        
        # Yahoo Finance (no API key needed)
        fetcher = DataFetcher('NVDA', source='yfinance')
        df_intra, df_daily = fetcher.process_data()
        
        # With custom configuration
        fetcher = DataFetcher(
            ticker='AAPL',
            source='polygon',
            start_date='2024-01-01',
            end_date='2024-06-30',
            api_key='your_polygon_key'
        )
    
    Attributes:
        ticker: Stock ticker symbol
        source: Data source being used
        connector: Underlying connector instance
    """
    
    # Default source selection logic
    DEFAULT_SOURCE = DataSource.POLYGON
    
    def __init__(
        self,
        ticker: str,
        source: Union[str, DataSource] = None,
        start_date: str = None,
        end_date: str = None,
        interval: Union[str, DataInterval] = DataInterval.MINUTE_1,
        **connector_kwargs
    ):
        """
        Initialize the DataFetcher.
        
        Args:
            ticker: Stock ticker symbol
            source: Data source ('polygon', 'yfinance', or auto-detect)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (default: 1 minute)
            **connector_kwargs: Additional arguments for the connector
                - For Polygon: api_key, enforce_rate_limit
                - For yfinance: progress
        """
        self.ticker = ticker.upper()
        self.start_date = '2021-02-01' #start_date or '2019-01-01'
        self.end_date = '2025-12-31' #end_date or str(date.today())
        
        # Handle interval
        if isinstance(interval, str):
            interval = DataInterval(interval)
        self.interval = interval
        
        # Determine source
        self.source = self._determine_source(source, connector_kwargs)
        
        # Create connector
        self.connector = self._create_connector(connector_kwargs)
        
        logger.info(
            f"DataFetcher initialized for {self.ticker} "
            f"using {self.source.value} connector"
        )
    
    def _determine_source(
        self,
        source: Union[str, DataSource, None],
        kwargs: dict
    ) -> DataSource:
        """
        Determine which data source to use.
        
        Priority:
        1. Explicitly specified source
        2. If POLYGON_API_KEY is set, use Polygon
        3. Fall back to yfinance (free, no key needed)
        
        Args:
            source: Explicitly specified source or None
            kwargs: Connector kwargs (may contain api_key)
            
        Returns:
            DataSource to use
        """
        if source is not None:
            if isinstance(source, str):
                return DataSource(source.lower())
            return source
        
        # Auto-detect based on available credentials
        polygon_key = kwargs.get('api_key') or os.getenv('POLYGON_API_KEY')
        
        if polygon_key:
            return DataSource.POLYGON
        else:
            logger.info(
                "No POLYGON_API_KEY found, using yfinance (free, no key required)"
            )
            return DataSource.YFINANCE
    
    def _create_connector(self, kwargs: dict) -> BaseConnector:
        """
        Create the appropriate connector instance.
        
        Args:
            kwargs: Arguments to pass to connector
            
        Returns:
            Connector instance
        """
        # Filter kwargs to only those relevant for the connector
        if self.source == DataSource.POLYGON:
            valid_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ['api_key', 'enforce_rate_limit', 'requests_per_minute']
            }
        elif self.source == DataSource.YFINANCE:
            valid_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ['progress']
            }
        else:
            valid_kwargs = kwargs
        
        return get_connector(self.source, **valid_kwargs)
    
    def fetch_intraday(self) -> pd.DataFrame:
        """
        Fetch intraday data.
        
        Returns:
            DataFrame with standardized intraday schema
        """
        config = FetchConfig(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.interval,
            include_dividends=True
        )
        
        return self.connector.fetch_intraday(config)
    
    def fetch_daily(self) -> pd.DataFrame:
        """
        Fetch daily data.
        
        Returns:
            DataFrame with standardized daily schema
        """
        config = FetchConfig(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=DataInterval.DAY_1,
            include_dividends=False
        )
        
        return self.connector.fetch_daily(config)
    
    def fetch_dividends(self) -> pd.DataFrame:
        """
        Fetch dividend data.
        
        Returns:
            DataFrame with dividend data
        """
        return self.connector.fetch_dividends(self.ticker)
    
    def process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch and process all data.
        
        This is the main method that returns both intraday and daily data
        in the standardized format expected by the backtesting engine.
        
        Returns:
            Tuple of (intraday_df, daily_df)
        """
        logger.info(f"Processing data for {self.ticker}...")
        
        df_intraday = self.fetch_intraday()
        df_daily = self.fetch_daily()
        
        return df_intraday, df_daily
    
    def save_data(
        self,
        df_intraday: pd.DataFrame,
        df_daily: pd.DataFrame,
        output_dir: str = None
    ) -> Tuple[str, str]:
        """
        Save processed data to CSV files.
        
        Args:
            df_intraday: Intraday DataFrame
            df_daily: Daily DataFrame
            output_dir: Output directory (default: data/{ticker}/)
            
        Returns:
            Tuple of (intraday_path, daily_path)
        """
        if output_dir is None:
            output_dir = f'data/{self.ticker}'
        
        os.makedirs(output_dir, exist_ok=True)
        
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        
        intraday_path = os.path.join(
            output_dir, f'{self.ticker}_intra_feed_{version}.csv'
        )
        daily_path = os.path.join(
            output_dir, f'{self.ticker}_daily_feed_{version}.csv'
        )
        
        df_intraday.to_csv(intraday_path)
        df_daily.to_csv(daily_path)
        
        logger.info(f"Saved intraday data to: {intraday_path}")
        logger.info(f"Saved daily data to: {daily_path}")
        
        return intraday_path, daily_path
    
    @classmethod
    def load_data(
        cls,
        ticker: str,
        data_dir: str = 'data'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load existing data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            data_dir: Base data directory
            
        Returns:
            Tuple of (intraday_df, daily_df)
            
        Raises:
            FileNotFoundError: If no data files found
        """
        ticker_dir = os.path.join(data_dir, ticker)
        
        if not os.path.exists(ticker_dir):
            raise FileNotFoundError(f"No data directory found for {ticker}")
        
        # Find latest files
        intra_files = [
            f for f in os.listdir(ticker_dir)
            if f.startswith(f'{ticker}_intra_feed') and f.endswith('.csv')
        ]
        daily_files = [
            f for f in os.listdir(ticker_dir)
            if f.startswith(f'{ticker}_daily_feed') and f.endswith('.csv')
        ]
        
        if not intra_files or not daily_files:
            raise FileNotFoundError(f"No data files found for {ticker}")
        
        # Sort by modification time (newest first)
        intra_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(ticker_dir, x)),
            reverse=True
        )
        daily_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(ticker_dir, x)),
            reverse=True
        )
        
        # Load data
        df_intraday = pd.read_csv(
            os.path.join(ticker_dir, intra_files[0]),
            index_col=0,
            parse_dates=True
        )
        df_daily = pd.read_csv(
            os.path.join(ticker_dir, daily_files[0]),
            index_col=0,
            parse_dates=True
        )
        
        logger.info(f"Loaded {len(df_intraday)} intraday records for {ticker}")
        
        return df_intraday, df_daily
    
    @classmethod
    def load_or_fetch(
        cls,
        ticker: str,
        data_dir: str = 'data',
        source: str = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load existing data or fetch new data if not available.
        
        Args:
            ticker: Stock ticker symbol
            data_dir: Base data directory
            source: Data source to use if fetching
            **kwargs: Additional arguments for DataFetcher
            
        Returns:
            Tuple of (intraday_df, daily_df)
        """
        try:
            return cls.load_data(ticker, data_dir)
        except FileNotFoundError:
            logger.info(f"No existing data for {ticker}, fetching...")
            fetcher = cls(ticker, source=source, **kwargs)
            df_intraday, df_daily = fetcher.process_data()
            
            if not df_intraday.empty:
                fetcher.save_data(
                    df_intraday, df_daily,
                    os.path.join(data_dir, ticker)
                )
            
            return df_intraday, df_daily


def fetch_risk_free_rate(
    output_path: str = 'data/dtb3.csv'
) -> pd.DataFrame:
    """
    Fetch 3-month Treasury bill rate.
    
    Note: This is a placeholder. For actual implementation,
    consider using FRED API or downloading manually.
    
    Args:
        output_path: Path to save the CSV file
        
    Returns:
        DataFrame with date and rate columns
    """
    logger.warning(
        "Risk-free rate fetch not fully implemented. "
        "Please download DTB3 data from FRED manually or use an API."
    )
    
    # Try yfinance as a fallback for Treasury data
    try:
        import yfinance as yf
        # ^IRX is the 13-week Treasury Bill
        irx = yf.Ticker('^IRX')
        hist = irx.history(period='max')
        
        if not hist.empty:
            df = pd.DataFrame({
                'date': hist.index,
                'rate': hist['Close'].values  # Already in percentage form
            })
            df.to_csv(output_path, index=False)
            logger.info(f"Saved risk-free rate to {output_path}")
            return df
    except Exception as e:
        logger.warning(f"Could not fetch Treasury data: {e}")
    
    return pd.DataFrame(columns=['date', 'rate'])


def load_risk_free_rate(path: str = 'data/dtb3.csv') -> Optional[pd.Series]:
    """
    Load risk-free rate series from CSV.
    
    Args:
        path: Path to CSV file with 'date' and 'rate' columns
        
    Returns:
        Series indexed by date with rates in decimal form, or None if not found
    """
    try:
        rf = pd.read_csv(path)
        rf['date'] = pd.to_datetime(rf['date'])
        rf = rf.set_index('date')
        rf.index = rf.index.normalize()
        
        # Convert to decimal if in percentage form
        if rf['rate'].mean() > 1:
            rf['rate'] = rf['rate'] / 100.0
        
        return rf['rate']
    except Exception as e:
        logger.warning(f"Could not load risk-free rate from {path}: {e}")
        return None


# Convenience function for quick data fetching
def fetch_data(
    ticker: str,
    source: str = None,
    start_date: str = None,
    end_date: str = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick function to fetch data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        source: Data source ('polygon', 'yfinance', or auto)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        **kwargs: Additional connector arguments
        
    Returns:
        Tuple of (intraday_df, daily_df)
        
    Example:
        df_intra, df_daily = fetch_data('NVDA')
        df_intra, df_daily = fetch_data('AAPL', source='yfinance')
    """
    fetcher = DataFetcher(
        ticker=ticker,
        source=source,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
    return fetcher.process_data()
