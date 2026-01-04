# Data Connectors Module

Market data fetching, persistence, and management from multiple sources.

## Overview

This module provides a unified interface for accessing market data from multiple sources, with consistent output schemas regardless of the data provider. It also includes data persistence, resampling, and market calendar utilities.

## Components

### 1) DataFetcher

Unified data fetcher that auto-selects or uses a specified source.

**Key Features:**
- Auto-detection of data source based on available API keys
- Consistent output schema across all sources
- Built-in data saving and loading utilities

**Source Selection Logic:**
1. If `POLYGON_API_KEY` environment variable is set → uses Polygon
2. Otherwise → falls back to yfinance (free, no key required)

---

### 2) Connectors

#### i) PolygonConnector

Direct access to Polygon.io REST API.

**Features:**
- Full historical intraday data (1-minute to daily)
- Automatic pagination for large datasets
- Built-in rate limiting (5 requests/minute for free tier)
- Dividend data fetching

**Requirements:**
- `POLYGON_API_KEY` environment variable or explicit `api_key` parameter

**Limitations:**
- Rate limited (free tier: 5 requests/minute)

#### ii) YFinanceConnector

Direct access to Yahoo Finance via the yfinance library.

**Features:**
- No API key required (free)
- Daily data with full history
- Dividend and stock split data
- Multi-ticker batch downloads

**Limitations:**
- 1-minute data: Last 7 days only
- 2-60 minute data: Last 60 days only
- Daily and higher intervals: Full history available

---

### DataManager

Central manager for data persistence and retrieval.

**Features:**
- Consistent file naming and directory organization
- Version management (multiple saves per ticker)
- Resampling support (e.g., 1-min to 5-min bars)
- Date-based filtering on load
- Automatic cleanup of old versions

**Directory Structure:**
```
outputs/data/
├── intraday/
│   └── {TICKER}/
│       └── {TICKER}_intraday_YYYYMMDD_HHMMSS.csv
├── daily/
│   └── {TICKER}/
│       └── {TICKER}_daily_YYYYMMDD_HHMMSS.csv
└── dividends/
    └── {TICKER}/
        └── {TICKER}_dividends_YYYYMMDD_HHMMSS.csv
```

---

### MarketCalendar

US equity market calendar management.

**Features:**
- Market holidays by year (2023-2026)
- Early close dates and times
- Trading day validation
- Trading minutes calculation

---

## Data Schema

All connectors produce DataFrames with standardized schemas.

### Intraday Schema

| Column | Type | Description |
|--------|------|-------------|
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing price |
| `volume` | int | Volume |
| `day` | date | Trading day |
| `vwap` | float | Volume-weighted average price |
| `move_open` | float | Absolute move from open |
| `ticker_dvol` | float | Rolling daily volatility |
| `min_from_open` | float | Minutes from market open |
| `minute_of_day` | int | Minute of the trading day |
| `move_open_rolling_mean` | float | Rolling mean of move from open |
| `sigma_open` | float | Rolling std of move from open |
| `dividend` | float | Dividend amount (0 if none) |

---

### Daily Schema

| Column | Type | Description |
|--------|------|-------------|
| `caldt` | datetime | Calendar date |
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing price |
| `volume` | int | Volume |

---

## Usage

### Basic Data Fetching

```python
from connectors import DataFetcher

# Auto-detect source (Polygon if API key available, else yfinance)
fetcher = DataFetcher('NVDA')
df_intraday, df_daily = fetcher.process_data()

# Explicitly use yfinance (free, no key required)
fetcher = DataFetcher('NVDA', source='yfinance')
df_intraday, df_daily = fetcher.process_data()

# Use Polygon with explicit API key
fetcher = DataFetcher('NVDA', source='polygon', api_key='your_key')
df_intraday, df_daily = fetcher.process_data()

# Custom date range
fetcher = DataFetcher(
    ticker='AAPL',
    source='polygon',
    start_date='2024-01-01',
    end_date='2024-06-30'
)
df_intraday, df_daily = fetcher.process_data()
```
---

### Quick Fetch Function

```python
from connectors import fetch_data

# One-liner data fetch
df_intraday, df_daily = fetch_data('AAPL', source='yfinance')

# With date range
df_intraday, df_daily = fetch_data(
    'TSLA',
    source='polygon',
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

---

### Data Persistence with DataManager

```python
from connectors import DataFetcher, DataManager

# Fetch data
fetcher = DataFetcher('NVDA', source='yfinance')
df_intraday, df_daily = fetcher.process_data()

# Save data
manager = DataManager(base_dir='outputs/data')
paths = manager.save_all('NVDA', df_intraday, df_daily)
# Returns: {'intraday': 'path/to/file.csv', 'daily': 'path/to/file.csv'}

# Load latest data
df_intraday = manager.load_intraday('NVDA')
df_daily = manager.load_daily('NVDA')

# Load with resampling (e.g., for ML features)
df_5min = manager.load_intraday_resampled('NVDA', interval='5min')
df_15min = manager.load_intraday_resampled('NVDA', interval='15min')

# Load with date filtering
df_recent = manager.load_intraday_from_date('NVDA', start_date='2024-06-01')
df_q3 = manager.load_intraday_from_date(
    'NVDA',
    start_date='2024-07-01',
    end_date='2024-09-30'
)

# Load last N trading days
df_last_30 = manager.load_intraday_last_n_days('NVDA', n_days=30)

# List available tickers
tickers = manager.list_tickers(data_type='intraday')

# Get data info
info = manager.get_data_info('NVDA', 'intraday')
# Returns: {'row_count': 50000, 'date_range': {...}, 'size_mb': 12.5, ...}

# Cleanup old versions (keep only 3 most recent)
deleted = manager.cleanup_old_versions('NVDA', 'intraday', keep_n=3)
```

---

### Risk-Free Rate

```python
from connectors import fetch_risk_free_rate, load_risk_free_rate

# Fetch and save risk-free rate (3-month T-Bill)
rf_df = fetch_risk_free_rate(output_path='data/dtb3.csv')

# Load existing risk-free rate series
rf_series = load_risk_free_rate('data/dtb3.csv')
# Returns pandas Series indexed by date with rates in decimal form
```

---

### Market Calendar

```python
from connectors import MarketCalendar, get_market_calendar
from datetime import date

calendar = MarketCalendar()

# Check if date is a holiday
if calendar.is_holiday(date(2024, 12, 25)):
    print("Market closed")

# Check if date is an early close
if calendar.is_early_close(date(2024, 12, 24)):
    close_time = calendar.get_early_close_time(date(2024, 12, 24))
    print(f"Market closes at {close_time}")  # 13:00

# Get all holidays for a year
holidays_2024 = calendar.get_holidays(2024)

# Check if trading day (not weekend, not holiday)
if calendar.is_trading_day(date(2024, 7, 4)):
    print("Markets open")
else:
    print("Markets closed")

# Get trading minutes for a date
minutes = calendar.get_trading_minutes(date(2024, 12, 24))  # 210 (early close)

# Get calendar as dictionary (for BacktestEngine)
calendar_dict = calendar.to_dict()
# or use convenience function:
calendar_dict = get_market_calendar()
```

---

### Direct Connector Access

```python
from connectors import PolygonConnector, YFinanceConnector, FetchConfig, DataInterval

# Polygon direct access
polygon = PolygonConnector(api_key='your_key')
config = FetchConfig(
    ticker='NVDA',
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval=DataInterval.MINUTE_5
)
df = polygon.fetch_intraday(config)

# yfinance direct access
yf_connector = YFinanceConnector()
df = yf_connector.fetch_intraday(config)

# Fetch dividends
dividends = yf_connector.fetch_dividends('AAPL')

# Multi-ticker download (yfinance only)
df_multi = YFinanceConnector.download_multiple(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1d'
)
```

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `POLYGON_API_KEY` | Polygon.io API key (enables Polygon as default source) |

---

### FetchConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `start_date` | str | '2024-01-01' | Start date (YYYY-MM-DD) |
| `end_date` | str | today | End date (YYYY-MM-DD) |
| `interval` | DataInterval | MINUTE_1 | Data interval |
| `include_prepost` | bool | False | Include pre/post market data |
| `include_dividends` | bool | True | Fetch and merge dividend data |
| `adjust_prices` | bool | True | Use split/dividend adjusted prices |

---

### Supported Intervals

| Interval | Polygon | yfinance | Notes |
|----------|---------|----------|-------|
| 1 minute | ✓ | ✓ | yfinance: 7 days max |
| 2 minutes | ✓ | ✓ | yfinance: 60 days max |
| 5 minutes | ✓ | ✓ | yfinance: 60 days max |
| 15 minutes | ✓ | ✓ | yfinance: 60 days max |
| 30 minutes | ✓ | ✓ | yfinance: 60 days max |
| 1 hour | ✓ | ✓ | |
| 1 day | ✓ | ✓ | Full history |
| 1 week | ✓ | ✓ | Full history |
| 1 month | ✓ | ✓ | Full history |

---

## Integration with Backtesting

```python
from connectors import DataManager, MarketCalendar, load_risk_free_rate
from backtesting import BacktestEngine

# Load data
manager = DataManager()
df_intraday = manager.load_intraday('TSLA')
df_daily = manager.load_daily('TSLA')

# Get calendar and risk-free rate
calendar = MarketCalendar()
rf_series = load_risk_free_rate('data/dtb3.csv')

# Prepare for backtest
all_days = df_intraday['day'].unique()

engine = BacktestEngine(ticker='TSLA')
results = engine.run(
    df=df_intraday,
    ticker_daily_data=df_daily,
    all_days=all_days,
    config=config,
    market_calendar=calendar.to_dict(),
    risk_free_rate_series=rf_series
)
```
