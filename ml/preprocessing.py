"""
Feature generation for ML strategies.
Uses 'ta' library only (pandas-ta has dependency conflicts).
"""

import numpy as np
import pandas as pd
import ta
import warnings
from typing import Dict, Any, List, Optional

warnings.filterwarnings('ignore')


class PreProcessing:
    """
    Generates features for ML/RL trading strategies.
    Compatible with your existing DataFetcher output format and BackTest precomputed data.
    """
    
    def __init__(self, min_lookback: int = 200):
        """
        Args:
            min_lookback: Minimum bars required for feature calculation.
                         First min_lookback rows will be dropped.
        """
        self.min_lookback = min_lookback
        self.feature_names = []
    
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical features from OHLCV data.
        
        Used during training on historical DataFrames.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'day']
            
        Returns:
            DataFrame with all features
        """
        data = df.copy()
        
        # Ensure proper column names
        data.columns = data.columns.str.lower()
        
        # Store original columns
        original_cols = list(data.columns)

        # 1. Custom features (works on DataFrame)
        data = self._add_custom_features(data)
        
        # 2. Rolling risk metrics
        data = self._add_risk_features(data)
        
        # 3. Optionally add ta library features
        # data = self._add_ta_features(data)
        
        # Clean up
        data = self._fix_dataset_inconsistencies(data)
        
        # Drop initial rows with NaN from lookback
        data = data.iloc[self.min_lookback:].reset_index(drop=True)
        
        # Store feature names (excluding original columns)
        self.feature_names = [c for c in data.columns if c not in original_cols]
        
        return data
        
    
    def generate_features_from_arrays(
        self,
        current_day_info: Dict[str, Any],
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate features from BackTest's precomputed array format.
        
        Used during inference/backtesting when data is in precomputed dict format.
        Produces features consistent with generate_features() for model compatibility.
        
        Args:
            current_day_info: Dictionary containing precomputed day data with keys:
                - close_prices: np.ndarray of close prices
                - volumes: np.ndarray of volumes
                - vwap: np.ndarray of VWAP values
                - sigma_open: np.ndarray of sigma values
                - open: float, opening price
                - ticker_dvol: float, daily volatility
            feature_names: Optional list of feature names to generate.
                          If None, generates all available features.
                          
        Returns:
            DataFrame with features matching the trained model's expected input
        """
        # Extract arrays from precomputed data
        close_prices = current_day_info['close_prices']
        volumes = current_day_info['volumes']
        vwap = current_day_info['vwap']
        sigma_open = current_day_info['sigma_open']
        open_price = current_day_info['open']
        ticker_dvol = current_day_info['ticker_dvol']
        
        n = len(close_prices)
        
        # Build base DataFrame from intraday arrays
        data = pd.DataFrame({
            'close': close_prices,
            'volume': volumes,
            'vwap': vwap,
            'sigma_open': sigma_open,
            'open': np.full(n, open_price),
            'high': np.maximum.accumulate(close_prices),
            'low': np.minimum.accumulate(close_prices),
        })
        
        # Generate custom features (subset that works with intraday data)
        data = self._add_custom_features_intraday(data, ticker_dvol)
        
        # Clean up
        data = self._fix_dataset_inconsistencies(data)
        
        # Filter to requested features if specified
        if feature_names is not None:
            available_cols = [c for c in feature_names if c in data.columns]
            missing_cols = [c for c in feature_names if c not in data.columns]
            
            if missing_cols:
                # Add missing columns as zeros (model expects them)
                for col in missing_cols:
                    data[col] = 0.0
            
            # Reorder to match expected feature order
            data = data[feature_names]
        
        return data
    
    
    def _add_custom_features_intraday(
        self, 
        data: pd.DataFrame, 
        ticker_dvol: float
    ) -> pd.DataFrame:
        """
        Add custom features optimized for intraday data.
        
        Generates a subset of features that are meaningful for single-day
        intraday bars (typically ~390 1-minute bars).
        
        Args:
            data: DataFrame with OHLCV columns
            ticker_dvol: Daily volatility for the ticker
            
        Returns:
            DataFrame with added features
        """
        n = len(data)
        
        # Lagged OHLCV
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                data[f'prev_{col}'] = data[col].shift(1)
        
        # Rolling volatility at multiple windows (subset for intraday)
        for window in [5, 10, 20, 30, 50, 60]:
            if n >= window:
                data[f'vol_{window}'] = data['close'].rolling(window=window).std().abs()
        
        # Simple moving averages
        for window in [5, 10, 20, 30, 50, 60]:
            if n >= window:
                data[f'ma_{window}'] = data['close'].rolling(window=window).mean()
        
        # Exponential moving averages
        for window in [5, 10, 20, 60]:
            if n >= window:
                data[f'ema_{window}'] = data['close'].ewm(span=window, adjust=False).mean()
        
        # Log returns
        data['lr_open'] = 0.0  # Intraday - open is constant
        data['lr_high'] = np.log(data['high']).diff().fillna(0)
        data['lr_low'] = np.log(data['low']).diff().fillna(0)
        data['lr_close'] = np.log(data['close']).diff().fillna(0)
        data['r_volume'] = data['volume'].diff().fillna(0)
        
        # RSI at multiple periods (subset for intraday)
        for period in [5, 10, 14, 26]:
            if n >= period:
                data[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
        
        # MACD variants (only if enough bars)
        if n >= 26:
            data['macd_normal'] = self._calculate_macd(data['close'], 12, 26, 9)
        if n >= 50:
            data['macd_short'] = self._calculate_macd(data['close'], 10, 50, 5)
        
        # Z-score
        for window in [20, 30]:
            if n >= window:
                ma = data['close'].rolling(window=window).mean()
                std = data['close'].rolling(window=window).std()
                data[f'zscore_{window}'] = (data['close'] - ma) / std
        
        # Price momentum
        for lag in [1, 5, 10]:
            if n > lag:
                data[f'momentum_{lag}'] = data['close'].pct_change(lag)
        
        # Volume features
        if n >= 20:
            data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma_20'].replace(0, np.nan)
        
        # VWAP deviation
        if 'vwap' in data.columns:
            data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap'].replace(0, np.nan)
        
        # Daily volatility (constant for intraday)
        data['ticker_dvol'] = ticker_dvol
        
        # Intraday-specific features
        data['intraday_return'] = (data['close'] - data['open'].iloc[0]) / data['open'].iloc[0]
        data['intraday_range'] = (data['high'] - data['low']) / data['close']
        data['close_to_high'] = (data['high'] - data['close']) / data['high'].replace(0, np.nan)
        data['close_to_low'] = (data['close'] - data['low']) / data['close'].replace(0, np.nan)
        
        # Rolling Sharpe (simplified, intraday)
        if n >= 20:
            returns = data['close'].pct_change()
            data['rolling_sharpe'] = (
                returns.rolling(20).mean() / 
                returns.rolling(20).std().replace(0, np.nan)
            )
        
        # Rolling Sortino (simplified, intraday)
        if n >= 20:
            returns = data['close'].pct_change()
            neg_returns = returns.copy()
            neg_returns[neg_returns > 0] = 0
            downside_std = neg_returns.rolling(20).std()
            data['rolling_sortino'] = (
                returns.rolling(20).mean() / 
                downside_std.replace(0, np.nan)
            )
        
        # Drawdown series
        cummax = data['close'].cummax()
        data['to_drawdown_series'] = (data['close'] - cummax) / cummax.replace(0, np.nan)
        
        return data
    
    
    def _add_ta_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add features from the 'ta' library.
        
        Args:
            data: DataFrame to add ta features to
            
        Returns:
            DataFrame with added ta features
        """
        try:
            data = ta.add_all_ta_features(
                data, 
                open='open', high='high', low='low', 
                close='close', volume='volume',
                fillna=True
            )
        except Exception as e:
            print(f"Warning: ta features failed: {e}")
        
        return data
    
    
    def _add_custom_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom features.
        
        Args:
            data: DataFrame to add custom features to
            
        Returns:
            DataFrame with added custom features
        """
        
        # Lagged OHLCV
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[f'prev_{col}'] = data[col].shift(1)
        
        # Rolling volatility at multiple windows
        for window in [5, 10, 20, 30, 50, 60, 100, 200]:
            data[f'vol_{window}'] = data['close'].rolling(window=window).std().abs()
        
        # Simple moving averages
        for window in [5, 10, 20, 30, 50, 60, 100, 200]:
            data[f'ma_{window}'] = data['close'].rolling(window=window).mean()
        
        # Exponential moving averages
        for window in [5, 10, 20, 60, 64, 120]:
            data[f'ema_{window}'] = data['close'].ewm(span=window, adjust=False).mean()
        
        # Log returns
        data['lr_open'] = np.log(data['open']).diff().fillna(0)
        data['lr_high'] = np.log(data['high']).diff().fillna(0)
        data['lr_low'] = np.log(data['low']).diff().fillna(0)
        data['lr_close'] = np.log(data['close']).diff().fillna(0)
        data['r_volume'] = data['volume'].diff().fillna(0)
        
        # RSI at multiple periods
        for period in [5, 6, 7, 10, 14, 26, 28, 100]:
            data[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
        
        # MACD variants
        data['macd_normal'] = self._calculate_macd(data['close'], 12, 26, 9)
        data['macd_short'] = self._calculate_macd(data['close'], 10, 50, 5)
        data['macd_long'] = self._calculate_macd(data['close'], 200, 100, 50)
        
        return data
    
    
    def _add_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling risk metrics.
        
        Args:
            data: DataFrame to add risk features to
            
        Returns:
            DataFrame with added risk features
        """
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Rolling Sharpe (simplified)
        window = 20
        data['rolling_sharpe'] = (
            returns.rolling(window).mean() / 
            returns.rolling(window).std()
        )
        
        # Rolling volatility
        data['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Drawdown series
        cummax = data['close'].cummax()
        data['to_drawdown_series'] = (data['close'] - cummax) / cummax
        
        # Rolling Sortino
        neg_returns = returns.copy()
        neg_returns[neg_returns > 0] = 0
        downside_std = neg_returns.rolling(window).std()
        data['rolling_sortino'] = (
            returns.rolling(window).mean() / downside_std
        )
        
        return data
    
    
    @staticmethod
    def _calculate_rsi(price: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI.
        
        Args:
            price: Series of prices
            period: RSI period
            
        Returns:
            Series of RSI values
        """
        r = price.diff()
        upside = r.clip(lower=0)
        downside = -r.clip(upper=0)
        rs = upside.ewm(alpha=1/period).mean() / downside.ewm(alpha=1/period).mean()
        return 100 * (1 - (1 + rs) ** -1)
    
    
    @staticmethod
    def _calculate_macd(price: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
        """
        Calculate MACD signal.
        
        Args:
            price: Series of prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal EMA period
            
        Returns:
            Series of MACD values
        """
        fm = price.ewm(span=fast, adjust=False).mean()
        sm = price.ewm(span=slow, adjust=False).mean()
        md = fm - sm
        return md - md.ewm(span=signal, adjust=False).mean()
    
    
    def _fix_dataset_inconsistencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up NaN and inf values.
        
        Args:
            data: DataFrame to clean
            
        Returns:
            DataFrame with cleaned data
        """
        # Replace inf with nan
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then back fill
        data = data.ffill().bfill()
        
        # Fill any remaining NaN with 0
        data = data.fillna(0)
        
        # Drop any remaining columns with all NaN
        data = data.dropna(axis=1, how='all')
        
        return data
