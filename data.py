import time
import pytz
import logging
import requests
import pandas as pd
from typing import Tuple
from datetime import datetime
from constants import Constants


logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Data fetcher that can handle multiple tickers dynamically.
    """
    def __init__(self, ticker: str):
        """
        Initialize the DataFetcher for a specific ticker.
        
        Args:
            ticker: The stock ticker symbol to fetch data for
        """
        self.constants = Constants()
        self.ticker = ticker
        self.base_url = self.constants.base_url
        self.api_key = self.constants.api_key
        self.multiplier = self.constants.multiplier
        self.limit = self.constants.limit
        self.eastern = pytz.timezone(self.constants.timezone)
        self.enforce_rate_limit = self.constants.enforce_rate_limit
        self.start_date = self.constants.start_date
        self.end_date = self.constants.end_date

    
    def fetch_polygon_data(self, start_date: str, end_date: str, period: str) -> pd.DataFrame:
        """
        Fetch data from the Polygon API for the ticker.
        """
        url = (
            f'{self.base_url}' 
            + f'/v2/aggs/ticker/{self.ticker}' 
            + f'/range/{self.multiplier}' 
            + f'/{period}' 
            + f'/{start_date}' 
            + f'/{end_date}' 
            + f'?adjusted=true&sort=asc&limit={self.limit}&apiKey={self.api_key}'
        )
        
        data_list = []
        request_count = 0
        first_request_time = None

        while True:
            if self.enforce_rate_limit and request_count >= 5:
                elapsed_time = time.time() - first_request_time
                if elapsed_time < 60:
                    wait_time = 60 - elapsed_time
                    logger.info(f"API rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                request_count = 0
                first_request_time = time.time()

            if first_request_time is None and self.enforce_rate_limit:
                first_request_time = time.time()

            response = requests.get(url)
            if response.status_code != 200:
                error_message = response.json().get('error', 'No specific error message provided')
                if 'exceeded the maximum requests per minute' in error_message:
                    logger.warning("API rate limit reached. Waiting 60 seconds...")
                    time.sleep(60)
                    request_count = 0
                    continue
                else:
                    logger.error(f"Error fetching data: {error_message}")
                    break

            data = response.json()
            request_count += 1

            results_count = len(data.get('results', []))
            logger.debug(f"Fetched {results_count} entries for {self.ticker}")

            if 'results' in data:
                for entry in data['results']:
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

                    if period == 'minute':
                        if (eastern_time.time() >= datetime.strptime('09:30', '%H:%M').time() and
                            eastern_time.time() <= datetime.strptime('15:59', '%H:%M').time()):
                            data_list.append(data_entry)
                    else:
                        data_list.append(data_entry)

            if 'next_url' in data and data['next_url']:
                url = data['next_url'] + '&apiKey=' + self.api_key
            else:
                break

        df = pd.DataFrame(data_list)
        return df

    
    def fetch_dividends(self) -> pd.DataFrame:
        """
        Fetch dividend data for the ticker.
        """
        url = f'{self.base_url}/v3/reference/dividends?ticker={self.ticker}&limit=1000&apiKey={self.api_key}'
        dividends_list = []
        
        while True:
            response = requests.get(url)
            data = response.json()
            if 'results' in data:
                for entry in data['results']:
                    dividends_list.append({
                        'caldt': datetime.strptime(entry['ex_dividend_date'], '%Y-%m-%d'),
                        'dividend': entry['cash_amount']
                    })

            if 'next_url' in data and data['next_url']:
                url = data['next_url'] + '&apiKey=' + self.api_key
            else:
                break

        return pd.DataFrame(dividends_list)

    
    def calculate_daily_vwap(self, group: pd.DataFrame) -> pd.Series:
        """Calculate the daily VWAP for a given group of data."""
        hlc = (group['high'] + group['low'] + group['close']) / 3
        cum_vol_hlc = (group['volume'] * hlc).cumsum()
        cum_volume = group['volume'].cumsum()
        return cum_vol_hlc / cum_volume

    
    def calculate_daily_move_open(self, group: pd.DataFrame) -> pd.Series:
        """Calculate the daily move open for a given group of data."""
        open_price = group['open'].iloc[0]
        return (group['close'] / open_price - 1).abs()

    
    def calculate_daily_metrics(self, group: pd.DataFrame) -> pd.DataFrame:
        """Apply all daily calculations to a group."""
        group = group.copy()
        group['vwap'] = self.calculate_daily_vwap(group)
        group['move_open'] = self.calculate_daily_move_open(group)
        return group

    
    def process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the raw data and calculate various metrics.
        
        Returns:
            Tuple of (intraday_df, daily_df)
        """
        start_date_str = str(self.start_date)
        end_date_str = str(self.end_date)
        
        logger.info(f"Fetching intraday data for {self.ticker}...")
        ticker_intra_data = self.fetch_polygon_data(start_date_str, end_date_str, 'minute')
        
        logger.info(f"Fetching daily data for {self.ticker}...")
        ticker_daily_data = self.fetch_polygon_data(start_date_str, end_date_str, 'day')
        
        logger.info(f"Fetching dividends for {self.ticker}...")
        dividends = self.fetch_dividends()

        if ticker_intra_data.empty:
            logger.error(f"No intraday data fetched for {self.ticker}")
            return pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(ticker_intra_data)
        df['day'] = pd.to_datetime(df['caldt']).dt.date
        df.set_index('caldt', inplace=True)

        # Calculate daily metrics
        df = df.groupby('day', group_keys=False).apply(self.calculate_daily_metrics)

        # Handle daily returns
        daily_closes = df.groupby('day')['close'].last()
        daily_returns = daily_closes.pct_change()
        
        # Rolling volatility
        rolling_vol = daily_returns.rolling(window=15, min_periods=15).std()
        df['ticker_dvol'] = df['day'].map(rolling_vol)

        # Minutes from open
        df['min_from_open'] = ((df.index - df.index.normalize()) / pd.Timedelta(minutes=1)) - (9 * 60 + 30) + 1
        df['minute_of_day'] = df['min_from_open'].round().astype(int)

        # Rolling mean and sigma
        df['move_open_rolling_mean'] = df.groupby('minute_of_day')['move_open'].transform(
            lambda x: x.shift(1).rolling(window=14, min_periods=13).mean()
        )
        df['sigma_open'] = df.groupby('minute_of_day')['move_open'].transform(
            lambda x: x.shift(1).rolling(window=14, min_periods=13).std()
        )

        # Handle dividends
        if not dividends.empty:
            dividends['day'] = pd.to_datetime(dividends['caldt']).dt.date
            df_with_day = df.reset_index()
            df_with_day = df_with_day.merge(dividends[['day', 'dividend']], on='day', how='left')
            df_with_day['dividend'] = df_with_day['dividend'].fillna(0)
            df = df_with_day.set_index('caldt')
        else:
            df['dividend'] = 0

        return df, ticker_daily_data
