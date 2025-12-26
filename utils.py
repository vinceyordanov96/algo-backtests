import pandas as pd
import numpy as np
from numba import njit
from typing import List, Union
from numpy.lib.stride_tricks import sliding_window_view


# Optional Polars import
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


@njit(cache=True)
def calculate_var_numba(returns: np.ndarray, confidence_level: float) -> float:
    """
    JIT-compiled Value at Risk calculation using historical simulation.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95)
        
    Returns:
        VaR value
    """
    if len(returns) < 2:
        return np.nan
    
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Calculate the index for the percentile
    percentile_idx = int((1 - confidence_level) * len(sorted_returns))
    
    return -sorted_returns[percentile_idx]


@njit(cache=True)
def calculate_rolling_sharpe_numba(
    returns: np.ndarray,
    lookback: int
) -> float:
    """
    JIT-compiled rolling Sharpe ratio calculation (not annualized).
    
    Args:
        returns: Array of returns
        lookback: Lookback period
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < lookback:
        return np.nan
    
    recent = returns[-lookback:]
    mean_ret = np.mean(recent)
    std_ret = np.std(recent)
    
    if std_ret <= 0:
        return 0.0
    
    return mean_ret / std_ret


@njit(cache=True)
def calculate_position_size_numba(
    returns: np.ndarray,
    base_size: int,
    lookback: int
) -> int:
    """
    JIT-compiled position size calculation based on rolling Sharpe ratio.
    
    Args:
        returns: Array of previous returns
        base_size: Base position size
        lookback: Lookback period
        
    Returns:
        Adjusted position size
    """
    if len(returns) < lookback:
        return base_size
    
    sharpe = calculate_rolling_sharpe_numba(returns, lookback)
    
    if np.isnan(sharpe):
        return base_size
    
    # Scale factor based on Sharpe
    scale = 1.0 + 0.2 * sharpe
    scale = max(0.5, min(1.5, scale))  # Clip to [0.5, 1.5]
    
    return int(base_size * scale)


@njit(cache=True)
def calculate_drawdown_series_numba(aum: np.ndarray) -> np.ndarray:
    """
    JIT-compiled drawdown calculation for an AUM series.
    
    Args:
        aum: Array of AUM values
        
    Returns:
        Array of drawdown values (as negative decimals)
    """
    n = len(aum)
    drawdowns = np.zeros(n, dtype=np.float64)
    peak = aum[0]
    
    for i in range(n):
        if aum[i] > peak:
            peak = aum[i]
        drawdowns[i] = (aum[i] - peak) / peak if peak > 0 else 0.0
    
    return drawdowns


@njit(cache=True)
def calculate_sortino_components_numba(
    excess_returns: np.ndarray
) -> tuple:
    """
    JIT-compiled Sortino ratio component calculation.
    
    Args:
        excess_returns: Array of excess returns
        
    Returns:
        Tuple of (mean_excess, downside_deviation)
    """
    mean_excess = np.mean(excess_returns)
    
    # Calculate downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        return mean_excess, np.nan
    
    downside_var = np.mean(negative_returns ** 2)
    downside_dev = np.sqrt(downside_var)
    
    return mean_excess, downside_dev


class Utils:
    """
    Optimized utility functions for the trading strategies.
    Supports both pandas and Polars DataFrames where applicable.
    """
    def __init__(self):
        """
        Initialize the Utils class.
        """
        self.aum_0 = 100000.0
        self.symbol = 'TQQQ'


    def calculate_returns(
        self, 
        prices: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """
        Calculate the returns for a given prices series.
        
        Args:
            prices: Series or array of prices
            
        Returns:
            Series or array of returns
        """
        if isinstance(prices, pd.Series):
            return prices.pct_change()
        else:
            # NumPy array
            returns = np.zeros_like(prices)
            returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
            returns[0] = np.nan
            return returns

    
    def calculate_volatility(
        self, 
        returns: Union[pd.Series, np.ndarray], 
        rolling: bool = False, 
        lookback_period: int = 20,
        annualized: bool = False
    ) -> Union[float, pd.Series, np.ndarray]:
        """
        Calculate the volatility for a given returns series and lookback period.
        
        Args:
            returns: Series or array of returns
            rolling: If True, calculate rolling volatility
            lookback_period: Window for rolling calculation
            annualized: If True, annualize the volatility
            
        Returns:
            Volatility value, Series, or array
        """
        factor = np.sqrt(252) if annualized else 1
        
        if isinstance(returns, pd.Series):
            if rolling:
                return returns.rolling(lookback_period).std() * factor
            return returns.std() * factor
        else:
            # NumPy array
            if rolling:
                # Use stride tricks for efficient rolling std
                if len(returns) >= lookback_period:
                    windows = sliding_window_view(returns, lookback_period)
                    rolling_std = np.std(windows, axis=1) * factor
                    # Pad with NaN for alignment
                    result = np.full(len(returns), np.nan)
                    result[lookback_period-1:] = rolling_std
                    return result
                return np.full(len(returns), np.nan)
            return np.nanstd(returns) * factor

    
    def calculate_VaR(
        self, 
        returns: Union[pd.Series, List[float], np.ndarray], 
        confidence_level: float
    ) -> float:
        """
        Calculate Value at Risk using historical simulation.
        
        Args:
            returns: Returns data
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            VaR value
        """
        # Convert to numpy array if needed
        if isinstance(returns, pd.Series):
            returns_arr = returns.dropna().values.astype(np.float64)
        elif isinstance(returns, list):
            returns_arr = np.array(returns, dtype=np.float64)
        else:
            returns_arr = returns.astype(np.float64)
        
        return calculate_var_numba(returns_arr, confidence_level)


    def calculate_drawdown(self, previous_aum: float, peak_aum: float) -> float:
        """
        Calculate the drawdown for a given previous AUM and peak AUM.
        
        Args:
            previous_aum: Previous AUM value
            peak_aum: Peak AUM value
            
        Returns:
            Drawdown as a decimal
        """
        if peak_aum == 0:
            return 0.0
        return (previous_aum - peak_aum) / peak_aum


    def calculate_drawdown_series(
        self, 
        aum: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """
        Calculate drawdown series using Numba-optimized function.
        
        Args:
            aum: Series or array of AUM values
            
        Returns:
            Series or array of drawdown values
        """
        if isinstance(aum, pd.Series):
            aum_arr = aum.values.astype(np.float64)
            drawdowns = calculate_drawdown_series_numba(aum_arr)
            return pd.Series(drawdowns, index=aum.index)
        else:
            return calculate_drawdown_series_numba(aum.astype(np.float64))
    

    def get_daily_risk_free_rate(
        self, 
        risk_free_rate_series: pd.Series, 
        date: pd.Timestamp
    ) -> float:
        """
        Get the daily risk-free rate for a given date.
        
        Args:
            risk_free_rate_series: Series with date index and annualized rates in decimal form
            date: The date to get the rate for
            
        Returns:
            Daily risk-free rate (decimal)
        """
        if risk_free_rate_series is None or risk_free_rate_series.empty:
            return 0.0
        
        date_normalized = pd.Timestamp(date).normalize()
        
        # Try exact match first
        if date_normalized in risk_free_rate_series.index:
            annual_rate = risk_free_rate_series.loc[date_normalized]
        else:
            # Find most recent available rate
            available_dates = risk_free_rate_series.index[risk_free_rate_series.index <= date_normalized]
            if len(available_dates) == 0:
                return 0.0
            annual_rate = risk_free_rate_series.loc[available_dates[-1]]
        
        if pd.isna(annual_rate):
            return 0.0
        
        return annual_rate / 252


    def subtract_risk_free_rate(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = None,
        risk_free_rate_series: pd.Series = None
    ) -> pd.Series:
        """
        Subtract the daily risk-free rate from the returns.
        
        Args:
            returns: Series of daily returns
            risk_free_rate: Single annualized risk-free rate (decimal)
            risk_free_rate_series: Series of annualized rates indexed by date (decimal)
            
        Returns:
            Series of excess returns
        """
        if len(returns) < 2:
            return returns
        
        if risk_free_rate_series is not None and not risk_free_rate_series.empty:
            # Vectorized approach using reindex
            rf_aligned = risk_free_rate_series.reindex(returns.index, method='ffill') / 252
            rf_aligned = rf_aligned.fillna(0)
            return returns - rf_aligned
        elif risk_free_rate is not None:
            daily_rf = risk_free_rate / 252
            return returns - daily_rf
        else:
            return returns

    
    def calculate_position_size(
        self, 
        previous_returns: Union[pd.Series, List[float], np.ndarray], 
        base_size: int = 1000, 
        lookback_period: int = 14
    ) -> int:
        """
        Calculate position size based on rolling Sharpe ratio.
        
        Args:
            previous_returns: Previous returns data
            base_size: Base position size
            lookback_period: Lookback window
            
        Returns:
            Adjusted position size
        """
        # Convert to numpy array
        if isinstance(previous_returns, pd.Series):
            returns_arr = previous_returns.dropna().values.astype(np.float64)
        elif isinstance(previous_returns, list):
            returns_arr = np.array(previous_returns, dtype=np.float64)
        else:
            returns_arr = previous_returns.astype(np.float64)
        
        return calculate_position_size_numba(returns_arr, base_size, lookback_period)


    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        lookback_period: int = 20,
        risk_free_rate: float = None,
        risk_free_rate_series: pd.Series = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate the Sharpe ratio for a given returns series.
        
        Args:
            returns: Series of daily returns
            lookback_period: Minimum observations required
            risk_free_rate: Single annualized risk-free rate (decimal)
            risk_free_rate_series: Series of annualized rates indexed by date (decimal)
            annualize: Whether to annualize the ratio
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < lookback_period:
            return np.nan
        
        # Calculate excess returns
        excess_returns = self.subtract_risk_free_rate(
            returns, 
            risk_free_rate=risk_free_rate,
            risk_free_rate_series=risk_free_rate_series
        )
        
        excess_returns_clean = excess_returns.dropna()
        
        if len(excess_returns_clean) < lookback_period:
            return np.nan
        
        ret_std = excess_returns_clean.std()
        
        if ret_std <= 0:
            return 0.0
        
        sharpe = excess_returns_clean.mean() / ret_std
        
        if annualize:
            sharpe *= np.sqrt(252)
        
        return sharpe


    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        lookback_period: int = 20,
        risk_free_rate: float = None,
        risk_free_rate_series: pd.Series = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate the Sortino ratio using Numba-optimized components.
        
        Args:
            returns: Series of daily returns
            lookback_period: Minimum observations required
            risk_free_rate: Single annualized risk-free rate (decimal)
            risk_free_rate_series: Series of annualized rates indexed by date (decimal)
            annualize: Whether to annualize the ratio
            
        Returns:
            Sortino ratio
        """
        if len(returns) < lookback_period:
            return np.nan
        
        # Calculate excess returns
        excess_returns = self.subtract_risk_free_rate(
            returns,
            risk_free_rate=risk_free_rate,
            risk_free_rate_series=risk_free_rate_series
        )
        
        excess_returns_clean = excess_returns.dropna().values.astype(np.float64)
        
        if len(excess_returns_clean) < lookback_period:
            return np.nan
        
        mean_excess, downside_dev = calculate_sortino_components_numba(excess_returns_clean)
        
        if np.isnan(downside_dev) or downside_dev <= 0:
            return np.nan
        
        sortino = mean_excess / downside_dev
        
        if annualize:
            sortino *= np.sqrt(252)
        
        return sortino

    
    def estimate_slippage(
        self,
        price: float, 
        volume: float, 
        volatility: float,
        df_daily: pd.DataFrame, 
        slippage_factor: float = 0.1
    ) -> float:
        """
        Estimate slippage based on price, volume, and volatility.
        
        Args:
            price: Current price
            volume: Current volume
            volatility: Current volatility
            df_daily: Daily data DataFrame
            slippage_factor: Base slippage in basis points
            
        Returns:
            Slippage per share in dollars
        """
        price_val = float(price) if not isinstance(price, (pd.Series, np.ndarray)) else float(price)
        
        if isinstance(volume, (pd.Series, np.ndarray)):
            volume_val = float(np.mean(volume))
        else:
            volume_val = float(volume)
        
        # Calculate average volume
        if isinstance(df_daily, pd.DataFrame) and 'volume' in df_daily.columns:
            avg_volume = df_daily['volume'].mean()
        else:
            avg_volume = volume_val
        
        # Volatility factor
        if isinstance(df_daily, pd.DataFrame) and 'ret' in df_daily.columns:
            baseline_vol = df_daily['ret'].std()
            if baseline_vol > 0 and not np.isnan(volatility):
                vol_factor = 1 + (volatility / baseline_vol - 1) * 0.5
            else:
                vol_factor = 1.0
        else:
            vol_factor = 1.0
        
        # Volume factor
        if avg_volume > 0 and volume_val > 0:
            volume_factor = 1 + max(0, (avg_volume / volume_val - 1) * 0.3)
        else:
            volume_factor = 1.0
        
        # Cap factors
        vol_factor = min(vol_factor, 3.0)
        volume_factor = min(volume_factor, 3.0)
        
        # Calculate slippage per share
        slippage_per_share = price_val * (slippage_factor * vol_factor * volume_factor) / 10000
        
        return float(slippage_per_share)


    # Polars-specific utilities (only available if Polars is installed)
    if POLARS_AVAILABLE:
        @staticmethod
        def pandas_to_polars(df: pd.DataFrame) -> 'pl.DataFrame':
            """
            Convert pandas DataFrame to Polars DataFrame.
            
            Args:
                df: pandas DataFrame
                
            Returns:
                Polars DataFrame
            """
            return pl.from_pandas(df)
        
        
        @staticmethod
        def polars_to_pandas(df: 'pl.DataFrame') -> pd.DataFrame:
            """
            Convert Polars DataFrame to pandas DataFrame.
            
            Args:
                df: Polars DataFrame
                
            Returns:
                pandas DataFrame
            """
            return df.to_pandas()
        
        
        @staticmethod
        def calculate_returns_polars(prices: 'pl.Series') -> 'pl.Series':
            """
            Calculate returns using Polars.
            
            Args:
                prices: Polars Series of prices
                
            Returns:
                Polars Series of returns
            """
            return prices.pct_change()
        
        
        @staticmethod
        def calculate_rolling_volatility_polars(
            returns: 'pl.Series',
            window: int = 20,
            annualize: bool = True
        ) -> 'pl.Series':
            """
            Calculate rolling volatility using Polars.
            
            Args:
                returns: Polars Series of returns
                window: Rolling window size
                annualize: Whether to annualize
                
            Returns:
                Polars Series of rolling volatility
            """
            factor = np.sqrt(252) if annualize else 1
            return returns.rolling_std(window_size=window) * factor
