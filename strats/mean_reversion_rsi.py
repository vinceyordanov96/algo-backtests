# strats/mean_reversion_rsi.py

import numpy as np
from numba import njit
from typing import Tuple, Dict, Any, List


@njit(cache=True)
def calculate_rsi_numba(
    prices: np.ndarray,
    period: int
) -> np.ndarray:
    """
    JIT-compiled RSI (Relative Strength Index) calculation.
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over the lookback period
    
    Uses Wilder's smoothing method (exponential moving average).
    
    Args:
        prices: Array of close prices
        period: Lookback period for RSI calculation
        
    Returns:
        Array of RSI values (0-100 scale)
    """
    n = len(prices)
    rsi = np.full(n, np.nan, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    deltas = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        deltas[i] = prices[i] - prices[i - 1]
    
    # Separate gains and losses
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if deltas[i] > 0:
            gains[i] = deltas[i]
        elif deltas[i] < 0:
            losses[i] = -deltas[i]
    
    # Calculate initial average gain and loss (simple average for first period)
    first_avg_gain = 0.0
    first_avg_loss = 0.0
    
    for i in range(1, period + 1):
        first_avg_gain += gains[i]
        first_avg_loss += losses[i]
    
    first_avg_gain /= period
    first_avg_loss /= period
    
    # Calculate RSI for the first valid point
    if first_avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = first_avg_gain / first_avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Use Wilder's smoothing for subsequent values
    avg_gain = first_avg_gain
    avg_loss = first_avg_loss
    
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


@njit(cache=True)
def calculate_sma_numba(
    prices: np.ndarray,
    period: int
) -> np.ndarray:
    """
    JIT-compiled Simple Moving Average calculation.
    
    Args:
        prices: Array of close prices
        period: Lookback period for SMA calculation
        
    Returns:
        Array of SMA values
    """
    n = len(prices)
    sma = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return sma
    
    # Calculate SMA using rolling window (excluding current bar to avoid look-ahead)
    for i in range(period, n):
        window_sum = 0.0
        for j in range(i - period, i):
            window_sum += prices[j]
        sma[i] = window_sum / period
    
    return sma


@njit(cache=True)
def calculate_ema_numba(
    prices: np.ndarray,
    period: int
) -> np.ndarray:
    """
    JIT-compiled Exponential Moving Average calculation.
    
    Args:
        prices: Array of close prices
        period: Lookback period for EMA calculation
        
    Returns:
        Array of EMA values
    """
    n = len(prices)
    ema = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return ema
    
    # Calculate multiplier
    multiplier = 2.0 / (period + 1.0)
    
    # First EMA value is SMA
    sma_sum = 0.0
    for i in range(period):
        sma_sum += prices[i]
    ema[period - 1] = sma_sum / period
    
    # Calculate EMA for subsequent values
    for i in range(period, n):
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


class MeanReversionRSI:
    """
    RSI + SMA Mean Reversion Strategy.
    
    Combines momentum indicator (RSI) with trend filter (SMA) for
    mean reversion signals in trending markets.
    
    Strategy Logic:
    - Buy when RSI is oversold AND price is above SMA (buy dips in uptrend)
    - Exit when RSI is overbought OR price falls below SMA
    
    Uses static methods for stateless signal generation,
    compatible with vectorized backtesting.
    """

    def __init__(
        self,
        rsi_periods: List[int] = [5, 7, 10, 14, 21],
        rsi_oversold_levels: List[float] = [20, 25, 30, 35],
        rsi_overbought_levels: List[float] = [65, 70, 75, 80],
        sma_periods: List[int] = [50, 100, 150, 200]
    ):
        """
        Initialize the RSI + SMA mean reversion strategy.
        
        Args:
            rsi_periods: List of lookback periods for RSI calculation
            rsi_oversold_levels: List of oversold levels for RSI calculation
            rsi_overbought_levels: List of overbought levels for RSI calculation
            sma_periods: List of lookback periods for SMA calculation
        """
        self.rsi_periods = rsi_periods
        self.rsi_oversold_levels = rsi_oversold_levels
        self.rsi_overbought_levels = rsi_overbought_levels
        self.sma_periods = sma_periods

    
    def get_parameter_grid(self) -> Dict[str, list]:
        """
        Get parameter grid for RSI + SMA mean reversion strategy.
        
        Returns:
            Dictionary of parameter names to lists of values to test
        """
        return {
            'rsi_periods': self.rsi_periods,
            'rsi_oversold_levels': self.rsi_oversold_levels,
            'rsi_overbought_levels': self.rsi_overbought_levels,
            'sma_periods': self.sma_periods
        }

    
    @staticmethod
    def generate_signals(
        close_prices: np.ndarray,
        rsi_period: int = 10,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        sma_period: int = 200,
        exit_on_sma_cross: bool = True
    ) -> np.ndarray:
        """
        Vectorized signal generation for RSI + SMA mean reversion strategy.
        
        Entry: RSI < rsi_oversold AND price > SMA
        Exit: RSI > rsi_overbought OR (exit_on_sma_cross AND price < SMA)
        
        Args:
            close_prices: Array of close prices
            rsi_period: Lookback period for RSI calculation (default: 10)
            rsi_oversold: RSI threshold for oversold/entry condition (default: 30)
            rsi_overbought: RSI threshold for overbought/exit condition (default: 70)
            sma_period: Lookback period for SMA trend filter (default: 200)
            exit_on_sma_cross: Whether to exit when price crosses below SMA (default: True)
            
        Returns:
            Array of signals: 1 (long/buy), -1 (exit/sell), 0 (hold)
        """
        n = len(close_prices)
        
        # Calculate indicators
        rsi = calculate_rsi_numba(
            close_prices.astype(np.float64),
            rsi_period
        )
        
        sma = calculate_sma_numba(
            close_prices.astype(np.float64),
            sma_period
        )
        
        # Initialize signals
        signals = np.zeros(n, dtype=np.int32)
        
        # Entry condition: RSI oversold AND price above SMA (buying dip in uptrend)
        # Use price > sma to confirm we're in an uptrend
        entry_condition = (rsi < rsi_oversold) & (close_prices > sma) & (~np.isnan(rsi)) & (~np.isnan(sma))
        
        # Exit condition: RSI overbought (mean reversion complete)
        # Optionally also exit if price falls below SMA (trend reversal)
        if exit_on_sma_cross:
            exit_condition = (
                ((rsi > rsi_overbought) & (~np.isnan(rsi))) |
                ((close_prices < sma) & (~np.isnan(sma)))
            )
        else:
            exit_condition = (rsi > rsi_overbought) & (~np.isnan(rsi))
        
        signals[entry_condition] = 1
        signals[exit_condition] = -1
        
        return signals


    @staticmethod
    def generate_signals_with_indicators(
        close_prices: np.ndarray,
        rsi_period: int = 10,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        sma_period: int = 200,
        exit_on_sma_cross: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate signals with indicator data for analysis and visualization.
        
        Args:
            close_prices: Array of close prices
            rsi_period: Lookback period for RSI calculation
            rsi_oversold: RSI threshold for oversold condition
            rsi_overbought: RSI threshold for overbought condition
            sma_period: Lookback period for SMA calculation
            exit_on_sma_cross: Whether to exit when price crosses below SMA
            
        Returns:
            Tuple of (signals, rsi, sma)
        """
        n = len(close_prices)
        
        # Calculate indicators
        rsi = calculate_rsi_numba(
            close_prices.astype(np.float64),
            rsi_period
        )
        
        sma = calculate_sma_numba(
            close_prices.astype(np.float64),
            sma_period
        )
        
        # Initialize signals
        signals = np.zeros(n, dtype=np.int32)
        
        # Entry condition
        entry_condition = (rsi < rsi_oversold) & (close_prices > sma) & (~np.isnan(rsi)) & (~np.isnan(sma))
        
        # Exit condition
        if exit_on_sma_cross:
            exit_condition = (
                ((rsi > rsi_overbought) & (~np.isnan(rsi))) |
                ((close_prices < sma) & (~np.isnan(sma)))
            )
        else:
            exit_condition = (rsi > rsi_overbought) & (~np.isnan(rsi))
        
        signals[entry_condition] = 1
        signals[exit_condition] = -1
        
        return signals, rsi, sma


    @staticmethod
    def generate_signals_dual_sma(
        close_prices: np.ndarray,
        rsi_period: int = 10,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        sma_short_period: int = 50,
        sma_long_period: int = 200,
        require_sma_alignment: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate signals using RSI with dual SMA trend confirmation.
        
        Entry: RSI < oversold AND price > SMA_long AND (optionally SMA_short > SMA_long)
        Exit: RSI > overbought OR price < SMA_long OR (optionally SMA_short < SMA_long)
        
        Args:
            close_prices: Array of close prices
            rsi_period: Lookback period for RSI calculation
            rsi_oversold: RSI threshold for oversold condition
            rsi_overbought: RSI threshold for overbought condition
            sma_short_period: Lookback period for short-term SMA
            sma_long_period: Lookback period for long-term SMA
            require_sma_alignment: Require short SMA > long SMA for entry
            
        Returns:
            Tuple of (signals, rsi, sma_short, sma_long)
        """
        n = len(close_prices)
        
        # Calculate indicators
        rsi = calculate_rsi_numba(
            close_prices.astype(np.float64),
            rsi_period
        )
        
        sma_short = calculate_sma_numba(
            close_prices.astype(np.float64),
            sma_short_period
        )
        
        sma_long = calculate_sma_numba(
            close_prices.astype(np.float64),
            sma_long_period
        )
        
        # Initialize signals
        signals = np.zeros(n, dtype=np.int32)
        
        # Valid data mask
        valid_mask = (~np.isnan(rsi)) & (~np.isnan(sma_short)) & (~np.isnan(sma_long))
        
        # Entry condition
        if require_sma_alignment:
            entry_condition = (
                (rsi < rsi_oversold) &
                (close_prices > sma_long) &
                (sma_short > sma_long) &
                valid_mask
            )
        else:
            entry_condition = (
                (rsi < rsi_oversold) &
                (close_prices > sma_long) &
                valid_mask
            )
        
        # Exit condition
        exit_condition = (
            ((rsi > rsi_overbought) | (close_prices < sma_long)) &
            valid_mask
        )
        
        signals[entry_condition] = 1
        signals[exit_condition] = -1
        
        return signals, rsi, sma_short, sma_long


    @staticmethod
    def generate_signals_contrarian(
        close_prices: np.ndarray,
        rsi_period: int = 10,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        sma_period: int = 200
    ) -> np.ndarray:
        """
        Contrarian version: Buy oversold in downtrend, sell overbought in uptrend.
        
        This is the opposite of the standard strategy - it bets on mean reversion
        regardless of trend direction.
        
        Entry: RSI < oversold (regardless of SMA)
        Exit: RSI > overbought OR RSI returns to neutral (50)
        
        Args:
            close_prices: Array of close prices
            rsi_period: Lookback period for RSI calculation
            rsi_oversold: RSI threshold for oversold condition
            rsi_overbought: RSI threshold for overbought condition
            sma_period: Lookback period for SMA (used for context, not filtering)
            
        Returns:
            Array of signals: 1 (long/buy), -1 (exit/sell), 0 (hold)
        """
        n = len(close_prices)
        
        rsi = calculate_rsi_numba(
            close_prices.astype(np.float64),
            rsi_period
        )
        
        signals = np.zeros(n, dtype=np.int32)
        
        # Pure RSI mean reversion
        entry_condition = (rsi < rsi_oversold) & (~np.isnan(rsi))
        exit_condition = (rsi > rsi_overbought) & (~np.isnan(rsi))
        
        signals[entry_condition] = 1
        signals[exit_condition] = -1
        
        return signals
