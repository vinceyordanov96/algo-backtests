# algo-backtests/utils/constants.py
import os
import dotenv
import pandas as pd
from datetime import time, date

# Load environment variables
dotenv.load_dotenv()


class Constants:
    """
    Constants for the trading strategies.
    """
    def __init__(self):
        """
        Initialize the Constants class.
        """

        # Data fetching constants
        self.api_key = os.getenv("POLYGON_API_KEY")
        self.base_url = 'https://api.polygon.io'
        self.enforce_rate_limit = True
        self.limit = '50000'
        self.multiplier = '1'
        self.period = 'minute'
        self.benchmark_ticker = 'SPY'
        self.timezone = 'America/New_York'
        self.start_date = str(date(2024, 1, 1))
        self.end_date = str(date(2025, 12, 18))
        
        # ML Strategy constants
        self.ml_model_dir = 'ml/models'
        self.ml_window_sizes = [30]
        self.ml_model_types = ['gradient_boosting']

        # Base AUM constant (accessible directly as constants.AUM_0)
        self.AUM_0 = 100000.0

        # Market holidays and early closes
        self.holidays = {
            '2023': {
                'holidays': [pd.Timestamp(d).date() for d in [
                    '2023-01-02',
                    '2023-01-16',
                    '2023-02-20',
                    '2023-04-07',
                    '2023-05-29',
                    '2023-06-19',
                    '2023-07-04',
                    '2023-09-04',
                    '2023-11-23',
                    '2023-12-25'
                ]],
                'early_closes': {
                    pd.Timestamp(d).date(): t for d, t in {
                        '2023-07-03': time(13, 0),
                        '2023-11-24': time(13, 0)
                    }.items()
                }
            },
            '2024': {
                'holidays': [pd.Timestamp(d).date() for d in [
                    '2024-01-01',
                    '2024-01-15',
                    '2024-02-19',
                    '2024-03-29',
                    '2024-05-27',
                    '2024-06-19',
                    '2024-07-04',
                    '2024-09-02',
                    '2024-11-28',
                    '2024-12-25'
                ]],
                'early_closes': {
                    pd.Timestamp(d).date(): t for d, t in {
                        '2024-07-03': time(13, 0),
                        '2024-11-29': time(13, 0),
                        '2024-12-24': time(13, 0)
                    }.items()
                }
            },
            '2025': {
                'holidays': [pd.Timestamp(d).date() for d in [
                    '2025-01-01',
                    '2025-01-20',
                    '2025-02-17',
                    '2025-04-18',
                    '2025-05-26',
                    '2025-06-19',
                    '2025-07-04',
                    '2025-09-01',
                ]],
                'early_closes': {
                    pd.Timestamp(d).date(): t for d, t in {
                        '2025-07-03': time(13, 0),
                        '2025-11-27': time(13, 0),
                        '2025-12-24': time(13, 0)
                    }.items()
                }
            }
        }
