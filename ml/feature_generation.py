# algo-backtests/ml/feature_generation.py

import logging
import time
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Set

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureConfig:
    """
    Configuration for feature generation.
    
    Centralizes all feature generation parameters to ensure consistency
    between training and inference.
    """
    
    # Rolling window sizes for various features
    VOLATILITY_WINDOWS = [5, 10, 20, 30, 50, 60]
    MA_WINDOWS = [5, 10, 20, 30, 50, 60]
    EMA_WINDOWS = [5, 10, 20, 60]
    RSI_PERIODS = [5, 10, 14, 26]
    ZSCORE_WINDOWS = [20, 30]
    MOMENTUM_LAGS = [1, 5, 10]
    
    # MACD configurations: (fast, slow, signal)
    MACD_CONFIGS = {
        'macd_normal': (12, 26, 9),
        'macd_short': (10, 50, 5),
    }
    
    # Risk feature window
    RISK_WINDOW = 20
    
    # Minimum data length required for feature calculation
    MIN_LOOKBACK = 60


class FeatureGeneration:
    """
    This module provides unified feature generation that ensures consistency
    between training and inference. All features are generated using the same
    code to prevent train/inference mismatch.
    
    Features generated:
        - Lagged OHLCV values
        - Rolling volatility at multiple windows
        - Simple and exponential moving averages
        - Log returns
        - RSI at multiple periods
        - MACD variants
        - Z-scores
        - Price momentum
        - Volume features
        - Rolling risk metrics (Sharpe, Sortino, drawdown)
    """
    
    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        min_lookback: Optional[int] = None
    ):
        """
        Initialize the feature generator.
        
        Args:
            config: Feature configuration. If None, uses default FeatureConfig.
            min_lookback: Override for minimum lookback period. If None, uses config default.
        """
        self.config = config or FeatureConfig()
        self.min_lookback = min_lookback or self.config.MIN_LOOKBACK
        self.feature_names: List[str] = []
        self._generated_features: Set[str] = set()
    
    
    def generate_features(
        self,
        df: pd.DataFrame,
        drop_lookback_rows: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Generate all technical features from OHLCV data.
        
        This is the PRIMARY feature generation method used for both training
        and inference. It ensures feature consistency across all use cases.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                Optional columns: ['vwap', 'day', 'caldt']
            drop_lookback_rows: If True, drops initial rows that have NaN from
                               lookback calculations. Set to False for inference
                               where you need all rows.
            verbose: Whether to log progress information
            
        Returns:
            DataFrame with all generated features
        """
        start_time = time.time()
        data = df.copy()
        
        # Ensure proper column names (lowercase)
        data.columns = data.columns.str.lower()
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Store original columns to identify new features later
        original_cols = set(data.columns)
        initial_rows = len(data)
        
        if verbose:
            logger.info(f"Starting feature generation on {initial_rows:,} rows...")
        
        # Generate all features using unified methods with progress tracking
        feature_groups = [
            ('lagged', self._add_lagged_features),
            ('volatility', self._add_volatility_features),
            ('moving_average', self._add_moving_average_features),
            ('returns', self._add_return_features),
            ('rsi', self._add_rsi_features),
            ('macd', self._add_macd_features),
            ('zscore', self._add_zscore_features),
            ('momentum', self._add_momentum_features),
            ('volume', self._add_volume_features),
            ('risk', self._add_risk_features),
            ('price_position', self._add_price_position_features),
            ('time', self._add_time_features),
        ]
        
        for group_name, group_func in feature_groups:
            pre_cols = len(data.columns)
            data = group_func(data)
            new_cols = len(data.columns) - pre_cols
            if verbose and new_cols > 0:
                logger.info(f"  + {group_name}: {new_cols} features")
        
        # Clean up NaN and inf values
        data = self._clean_data(data)
        
        # Store feature names (excluding original columns)
        self.feature_names = [c for c in data.columns if c not in original_cols]
        self._generated_features = set(self.feature_names)
        
        # Optionally drop initial rows with NaN from lookback
        if drop_lookback_rows and len(data) > self.min_lookback:
            data = data.iloc[self.min_lookback:].reset_index(drop=True)
        
        elapsed = time.time() - start_time
        if verbose:
            logger.info(f"Feature generation complete: {len(self.feature_names)} features, "
                       f"{len(data):,} rows | {elapsed:.1f}s")
        
        return data
    
    
    def generate_features_for_inference(
        self,
        df: pd.DataFrame,
        required_features: List[str],
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Generate features for inference/backtesting.
        
        This method generates features and validates that all required features
        are present. Unlike the training path, it does NOT fill missing features
        with zeros (which would corrupt predictions).
        
        Args:
            df: DataFrame with OHLCV data
            required_features: List of feature names the model expects
            verbose: Whether to log progress information
            
        Returns:
            DataFrame with features in the order expected by the model
            
        Raises:
            ValueError: If any required features cannot be generated
        """
        # Generate all features without dropping lookback rows
        data = self.generate_features(df, drop_lookback_rows=False, verbose=verbose)
        
        # Validate all required features are present
        available = set(data.columns)
        required = set(required_features)
        missing = required - available
        
        if missing:
            raise ValueError(
                f"Cannot generate required features: {missing}. "
                f"Available features: {sorted(available & required)[:10]}... "
                f"This indicates a train/inference mismatch."
            )
        
        # Return only required features in the correct order
        return data[required_features].copy()
    
    
    def _add_lagged_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged OHLCV values.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added lagged features
        """
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                data[f'prev_{col}'] = data[col].shift(1)
        
        return data
    
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling volatility features at multiple windows.
        
        Args:
            data: DataFrame with close prices
            
        Returns:
            DataFrame with added volatility features
        """
        for window in self.config.VOLATILITY_WINDOWS:
            if len(data) >= window:
                data[f'vol_{window}'] = (
                    data['close'].rolling(window=window).std().abs()
                )
        
        return data
    
    
    def _add_moving_average_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add simple and exponential moving averages.
        
        Args:
            data: DataFrame with close prices
            
        Returns:
            DataFrame with added MA features
        """
        # Simple moving averages
        for window in self.config.MA_WINDOWS:
            if len(data) >= window:
                data[f'ma_{window}'] = data['close'].rolling(window=window).mean()
        
        # Exponential moving averages
        for window in self.config.EMA_WINDOWS:
            if len(data) >= window:
                data[f'ema_{window}'] = data['close'].ewm(span=window, adjust=False).mean()
        
        return data
    
    
    def _add_return_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add log return features.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added return features
        """
        # Log returns for price columns
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                # Handle potential zeros/negatives
                safe_col = data[col].replace(0, np.nan)
                data[f'lr_{col}'] = np.log(safe_col).diff().fillna(0)
        
        # Volume difference (not log, as volume can be zero)
        if 'volume' in data.columns:
            data['r_volume'] = data['volume'].diff().fillna(0)
        
        return data
    
    
    def _add_rsi_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add RSI at multiple periods.
        
        Args:
            data: DataFrame with close prices
            
        Returns:
            DataFrame with added RSI features
        """
        for period in self.config.RSI_PERIODS:
            if len(data) >= period:
                data[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
        
        return data
    
    
    def _add_macd_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add MACD variants.
        
        Args:
            data: DataFrame with close prices
            
        Returns:
            DataFrame with added MACD features
        """
        for name, (fast, slow, signal) in self.config.MACD_CONFIGS.items():
            min_required = max(fast, slow, signal)
            if len(data) >= min_required:
                data[name] = self._calculate_macd(data['close'], fast, slow, signal)
        
        return data
    
    
    def _add_zscore_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add z-score features.
        
        Args:
            data: DataFrame with close prices
            
        Returns:
            DataFrame with added z-score features
        """
        for window in self.config.ZSCORE_WINDOWS:
            if len(data) >= window:
                ma = data['close'].rolling(window=window).mean()
                std = data['close'].rolling(window=window).std()
                # Avoid division by zero
                std_safe = std.replace(0, np.nan)
                data[f'zscore_{window}'] = (data['close'] - ma) / std_safe
        
        return data
    
    
    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add price momentum features.
        
        Args:
            data: DataFrame with close prices
            
        Returns:
            DataFrame with added momentum features
        """
        for lag in self.config.MOMENTUM_LAGS:
            if len(data) > lag:
                data[f'momentum_{lag}'] = data['close'].pct_change(lag)
        
        return data
    
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Args:
            data: DataFrame with volume column
            
        Returns:
            DataFrame with added volume features
        """
        if 'volume' not in data.columns:
            return data
        
        window = self.config.RISK_WINDOW
        if len(data) >= window:
            data['volume_ma_20'] = data['volume'].rolling(window=window).mean()
            # Avoid division by zero
            vol_ma_safe = data['volume_ma_20'].replace(0, np.nan)
            data['volume_ratio'] = data['volume'] / vol_ma_safe
        
        # VWAP deviation if available
        if 'vwap' in data.columns:
            vwap_safe = data['vwap'].replace(0, np.nan)
            data['vwap_deviation'] = (data['close'] - data['vwap']) / vwap_safe
        
        return data
    
    
    def _add_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling risk metrics.
        
        Args:
            data: DataFrame with close prices
            
        Returns:
            DataFrame with added risk features
        """
        window = self.config.RISK_WINDOW
        if len(data) < window:
            return data
        
        # Calculate returns for risk metrics
        returns = data['close'].pct_change()
        
        # Rolling Sharpe (simplified - no risk-free rate subtraction)
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_std_safe = rolling_std.replace(0, np.nan)
        data['rolling_sharpe'] = rolling_mean / rolling_std_safe
        
        # Rolling Sortino
        neg_returns = returns.copy()
        neg_returns[neg_returns > 0] = 0
        downside_std = neg_returns.rolling(window).std()
        downside_std_safe = downside_std.replace(0, np.nan)
        data['rolling_sortino'] = rolling_mean / downside_std_safe
        
        # Drawdown series
        cummax = data['close'].cummax()
        cummax_safe = cummax.replace(0, np.nan)
        data['drawdown'] = (data['close'] - cummax) / cummax_safe
        
        return data
    
    
    def _add_price_position_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add features describing price position relative to range.
        
        Args:
            data: DataFrame with OHLC columns
            
        Returns:
            DataFrame with added price position features
        """
        # Price position within high-low range
        range_size = data['high'] - data['low']
        range_safe = range_size.replace(0, np.nan)
        data['price_position'] = (data['close'] - data['low']) / range_safe
        
        # Distance from high and low as percentage
        high_safe = data['high'].replace(0, np.nan)
        close_safe = data['close'].replace(0, np.nan)
        data['dist_from_high'] = (data['high'] - data['close']) / high_safe
        data['dist_from_low'] = (data['close'] - data['low']) / close_safe
        
        return data
    
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        These features capture intraday patterns and time-of-day effects.
        The method tries to extract time information from various possible
        columns (caldt, timestamp, index) to ensure compatibility with
        different data formats.
        
        Args:
            data: DataFrame, optionally with datetime column or index
            
        Returns:
            DataFrame with added time features
        """
        # Try to get datetime from various sources
        datetime_col = None
        is_index = False
        
        # Check for common datetime column names
        for col_name in ['caldt', 'timestamp', 'datetime', 'date', 'time']:
            if col_name in data.columns:
                try:
                    datetime_col = pd.to_datetime(data[col_name])
                    break
                except Exception:
                    continue
        
        # Try the index if no column found
        if datetime_col is None and isinstance(data.index, pd.DatetimeIndex):
            datetime_col = data.index
            is_index = True
        
        # If we have datetime info, generate time features
        if datetime_col is not None:
            # Handle both Series and DatetimeIndex
            if is_index:
                hour = datetime_col.hour
                minute = datetime_col.minute
                dayofweek = datetime_col.dayofweek
            else:
                hour = datetime_col.dt.hour
                minute = datetime_col.dt.minute
                dayofweek = datetime_col.dt.dayofweek
            
            # Minute of day (0-389 for regular trading hours 9:30-16:00)
            data['minute_of_day'] = hour * 60 + minute
            
            # Minutes from market open (9:30 AM = 570 minutes from midnight)
            market_open_minutes = 9 * 60 + 30  # 9:30 AM
            data['min_from_open'] = data['minute_of_day'] - market_open_minutes
            
            # Normalize to 0-1 range for model compatibility
            # Regular trading day is ~390 minutes (9:30 AM to 4:00 PM)
            data['time_of_day_pct'] = data['min_from_open'] / 390.0
            
            # Hour of day (for capturing hourly patterns)
            data['hour_of_day'] = hour
            
            # Day of week (0=Monday, 4=Friday)
            data['day_of_week'] = dayofweek
            
            # Is it near market open (first 30 minutes)?
            data['near_open'] = (data['min_from_open'] <= 30).astype(int)
            
            # Is it near market close (last 30 minutes)?
            data['near_close'] = (data['min_from_open'] >= 360).astype(int)
        else:
            # If no datetime available, create placeholder features
            # These will be filled based on row index within each day
            # This handles cases where inference gets raw arrays
            n = len(data)
            
            # Assume 390 minutes per day, calculate position within day
            # This is a fallback that works for intraday data
            if 'day' in data.columns:
                # Group by day and create sequential minute index
                data['min_from_open'] = data.groupby('day').cumcount()
                data['minute_of_day'] = data['min_from_open'] + 570  # 9:30 AM offset
            else:
                # Last resort: use row index modulo typical day length
                data['min_from_open'] = np.arange(n) % 390
                data['minute_of_day'] = data['min_from_open'] + 570
            
            data['time_of_day_pct'] = data['min_from_open'] / 390.0
            data['hour_of_day'] = (data['minute_of_day'] // 60).astype(int)
            data['day_of_week'] = 0  # Unknown, default to Monday
            data['near_open'] = (data['min_from_open'] <= 30).astype(int)
            data['near_close'] = (data['min_from_open'] >= 360).astype(int)
        
        return data
    
    
    @staticmethod
    def _calculate_rsi(price: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            price: Series of prices
            period: RSI period
            
        Returns:
            Series of RSI values (0-100)
        """
        delta = price.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        # Avoid division by zero
        avg_loss_safe = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss_safe
        
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    
    @staticmethod
    def _calculate_macd(
        price: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> pd.Series:
        """
        Calculate MACD signal line.
        
        Args:
            price: Series of prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal EMA period
            
        Returns:
            Series of MACD histogram values
        """
        fast_ema = price.ewm(span=fast, adjust=False).mean()
        slow_ema = price.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        return macd_line - signal_line
    
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up NaN and inf values.
        
        Args:
            data: DataFrame to clean
            
        Returns:
            DataFrame with cleaned data
        """
        # Replace inf with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then back fill
        data = data.ffill().bfill()
        
        # Fill any remaining NaN with 0
        data = data.fillna(0)
        
        return data
    
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of generated feature names.
        
        Returns:
            List of feature column names
        """
        return self.feature_names.copy()
    
    
    def get_min_required_rows(self) -> int:
        """
        Get minimum rows required for feature generation.
        
        Returns:
            Minimum number of rows needed
        """
        return self.min_lookback
