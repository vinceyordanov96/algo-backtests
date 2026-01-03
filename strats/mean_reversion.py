import numpy as np
from numba import njit
from typing import Tuple, Dict, Any, List


@njit(cache=True)
def calculate_rolling_mean_std(
    prices: np.ndarray,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled rolling mean and standard deviation calculation.
    
    Args:
        prices: Array of prices
        lookback: Lookback period
        
    Returns:
        Tuple of (rolling_mean, rolling_std)
    """
    n = len(prices)
    rolling_mean = np.full(n, np.nan, dtype=np.float64)
    rolling_std = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        window = prices[i - lookback:i]  # Exclude current bar
        rolling_mean[i] = np.mean(window)
        rolling_std[i] = np.std(window)
    
    return rolling_mean, rolling_std


@njit(cache=True)
def calculate_zscore(
    prices: np.ndarray,
    rolling_mean: np.ndarray,
    rolling_std: np.ndarray
) -> np.ndarray:
    """JIT-compiled z-score calculation.
    
    Args:
        prices: Array of prices
        rolling_mean: Array of rolling means
        rolling_std: Array of rolling standard deviations
        
    Returns:
        Array of z-scores
    """
    n = len(prices)
    zscore = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if not np.isnan(rolling_mean[i]) and not np.isnan(rolling_std[i]) and rolling_std[i] > 0:
            zscore[i] = (prices[i] - rolling_mean[i]) / rolling_std[i]
    
    return zscore


class MeanReversion:
    """
    Mean Reversion (z-score based) strategy.
    Uses static methods for stateless signal generation.
    """

    def __init__(
        self,
        lookback_period: List[int] = [80, 100, 120],
        n_std_upper: List[float] = [1.5, 2.0],
        n_std_lower: List[float] = [1.5, 2.0],
        exit_threshold: List[float] = [0.0]
    ):
        """
        Initialize the mean reversion strategy Class with default parameters.
        Args:
            lookback_period: List of lookback periods for moving average and std calculation
            n_std_upper: List of number of standard deviations for upper band (sell signal)
            n_std_lower: List of number of standard deviations for lower band (buy signal)
            exit_threshold: List of z-score thresholds for exit signals (default 0.0)
        """
        self.lookback_period = lookback_period
        self.n_std_upper = n_std_upper
        self.n_std_lower = n_std_lower
        self.exit_threshold = exit_threshold
    
    
    def get_parameter_grid(self) -> Dict[str, list]:
        """
        Get parameter grid for mean reversion strategy.
        
        Returns:
            Dictionary of parameter names to lists of values to test
        """
        return {
            'zscore_lookbacks': self.lookback_period,
            'n_std_uppers': self.n_std_upper,
            'n_std_lowers': self.n_std_lower,
            'exit_thresholds': self.exit_threshold
        }

    
    @staticmethod
    def generate_signals(
        close_prices: np.ndarray,
        lookback_period: int,
        n_std_upper: float,
        n_std_lower: float,
        exit_threshold: float = 0.0
    ) -> np.ndarray:
        """
        Vectorized signal generation based on z-score mean reversion strategy.
        
        Args:
            close_prices: Array of close prices
            lookback_period: Number of periods for moving average and std calculation
            n_std_upper: Number of standard deviations for upper band (sell signal)
            n_std_lower: Number of standard deviations for lower band (buy signal)
            exit_threshold: Z-score threshold for exit signals (default 0.0)
            
        Returns:
            Array of signals: 1 (long/buy), -1 (exit/sell), 0 (hold)
        """
        n = len(close_prices)
        
        # Calculate rolling mean and std using Numba
        rolling_mean, rolling_std = calculate_rolling_mean_std(
            close_prices.astype(np.float64),
            lookback_period
        )
        
        # Calculate z-score
        zscore = calculate_zscore(
            close_prices.astype(np.float64), 
            rolling_mean, 
            rolling_std
        )
        
        # Initialize signals
        signals = np.zeros(n, dtype=np.int32)
        
        # Mean reversion logic
        buy_condition = zscore < -n_std_lower
        exit_condition = (zscore > n_std_upper) | (
            (zscore >= exit_threshold) & (zscore <= n_std_upper)
        )
        
        signals[buy_condition] = 1
        signals[exit_condition] = -1
        
        return signals

    
    @staticmethod
    def generate_signals_with_zscore(
        close_prices: np.ndarray,
        lookback_period: int,
        n_std_upper: float,
        n_std_lower: float,
        exit_threshold: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate signals with z-score data for analysis.
        
        Args:
            close_prices: Array of close prices
            lookback_period: Number of periods for moving average and std calculation
            n_std_upper: Number of standard deviations for upper band (sell signal)
            n_std_lower: Number of standard deviations for lower band (buy signal)
            exit_threshold: Z-score threshold for exit signals (default 0.0)
            
        Returns:
            Tuple of (signals, zscore, rolling_mean, rolling_std)
        """
        n = len(close_prices)
        
        rolling_mean, rolling_std = calculate_rolling_mean_std(
            close_prices.astype(np.float64),
            lookback_period
        )
        
        zscore = calculate_zscore(
            close_prices.astype(np.float64), 
            rolling_mean, 
            rolling_std
        )
        
        signals = np.zeros(n, dtype=np.int32)
        
        buy_condition = zscore < -n_std_lower
        exit_condition = (zscore > n_std_upper) | (
            (zscore >= exit_threshold) & (zscore <= n_std_upper)
        )
        
        signals[buy_condition] = 1
        signals[exit_condition] = -1
        
        return signals, zscore, rolling_mean, rolling_std
