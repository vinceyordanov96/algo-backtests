"""
Data connectors module.

This module provides unified access to market data from multiple sources:
    - Polygon.io (requires API key)
    - Yahoo Finance via yfinance (free, no key required)

Main Entry Points:
    - DataFetcher: Unified data fetcher that auto-selects or uses specified source
    - fetch_data(): Quick function for fetching data
    
Connectors:
    - PolygonConnector: Direct access to Polygon.io API
    - YFinanceConnector: Direct access to Yahoo Finance via yfinance

Example:
    # Auto-detect source (uses Polygon if API key available, else yfinance)
    from connectors import DataFetcher
    
    fetcher = DataFetcher('NVDA')
    df_intraday, df_daily = fetcher.process_data()
    
    # Explicitly use yfinance (free)
    fetcher = DataFetcher('NVDA', source='yfinance')
    df_intraday, df_daily = fetcher.process_data()
    
    # Quick fetch
    from connectors import fetch_data
    df_intra, df_daily = fetch_data('AAPL', source='yfinance')

Schema:
    All connectors produce DataFrames with a standardized schema.
    See connectors.base.StandardSchema for column definitions.
"""

# Base classes and types
from .base import (
    BaseConnector,
    DataSource,
    DataInterval,
    FetchConfig,
    StandardSchema,
    filter_market_hours,
)

# Connectors
from .polygon import PolygonConnector
from .yfinance import YFinanceConnector

# Unified interface
from .data import (
    DataFetcher,
    get_connector,
    fetch_data,
    fetch_risk_free_rate,
    load_risk_free_rate,
    CONNECTORS,
)

from .calendar import MarketCalendar, get_market_calendar
from .manager import DataManager

__all__ = [
    # Base
    'BaseConnector',
    'DataSource',
    'DataInterval',
    'FetchConfig',
    'StandardSchema',
    'filter_market_hours',
    
    # Connectors
    'PolygonConnector',
    'YFinanceConnector',
    'CONNECTORS',
    'get_connector',
    
    # Unified interface
    'DataFetcher',
    'fetch_data',
    'fetch_risk_free_rate',
    'load_risk_free_rate',
    
    # Calendar
    'MarketCalendar',
    'get_market_calendar',
    'DataManager',
]
