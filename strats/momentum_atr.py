# algo-backtests/strats/momentum/momentum_atr.py

import numpy as np
from numba import njit
from typing import Dict, List, Tuple


@njit(cache=True)
def calculate_true_range_numba(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray
) -> np.ndarray:
    """
    JIT-compiled True Range calculation.
    
    True Range = max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    )
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        
    Returns:
        Array of true range values
    """
    n = len(high_prices)
    tr = np.zeros(n, dtype=np.float64)
    
    # First bar: TR = high - low
    tr[0] = high_prices[0] - low_prices[0]
    
    for i in range(1, n):
        prev_close = close_prices[i - 1]
        high_low = high_prices[i] - low_prices[i]
        high_prev_close = abs(high_prices[i] - prev_close)
        low_prev_close = abs(low_prices[i] - prev_close)
        
        tr[i] = max(high_low, high_prev_close, low_prev_close)
    
    return tr


@njit(cache=True)
def calculate_atr_numba(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    period: int
) -> np.ndarray:
    """
    JIT-compiled Average True Range (ATR) calculation.
    
    Uses Wilder's smoothing method (exponential moving average).
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        period: ATR period (typically 14)
        
    Returns:
        Array of ATR values
    """
    n = len(high_prices)
    atr = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return atr
    
    # Calculate True Range
    tr = calculate_true_range_numba(high_prices, low_prices, close_prices)
    
    # First ATR = simple average of first 'period' TRs
    first_atr = 0.0
    for i in range(period):
        first_atr += tr[i]
    first_atr /= period
    atr[period - 1] = first_atr
    
    # Subsequent ATRs use Wilder's smoothing
    # ATR = ((period - 1) * prev_ATR + current_TR) / period
    multiplier = (period - 1.0) / period
    for i in range(period, n):
        atr[i] = multiplier * atr[i - 1] + tr[i] / period
    
    return atr


@njit(cache=True)
def calculate_atr_bands_numba(
    close_prices: np.ndarray,
    atr: np.ndarray,
    reference_price: float,
    atr_mult: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled ATR band calculation.
    
    Upper Band = reference_price + (atr_mult × ATR)
    Lower Band = reference_price - (atr_mult × ATR)
    
    Args:
        close_prices: Array of close prices
        atr: Array of ATR values
        reference_price: Reference price for band center
        atr_mult: ATR multiplier for band width
        
    Returns:
        Tuple of (upper_band, lower_band) arrays
    """
    n = len(close_prices)
    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if not np.isnan(atr[i]):
            upper_band[i] = reference_price + (atr_mult * atr[i])
            lower_band[i] = reference_price - (atr_mult * atr[i])
    
    return upper_band, lower_band


@njit(cache=True)
def calculate_trailing_atr_bands_numba(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    atr: np.ndarray,
    atr_mult: float,
    lookback: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled trailing ATR band calculation with rolling window.
    
    Uses PREVIOUS bar's rolling highest high and lowest low (excluding current bar):
        Upper Band = prev_rolling_high + (atr_mult × ATR × buffer_scale)
        Lower Band = prev_rolling_low - (atr_mult × ATR × buffer_scale)
    
    By excluding the current bar from the rolling calculation, the current
    bar's price CAN break above/below the band when making a new high/low.
    
    Args:
        close_prices: Array of close prices
        high_prices: Array of high prices
        low_prices: Array of low prices
        atr: Array of ATR values  
        atr_mult: ATR multiplier for band width
        lookback: Rolling window size for high/low calculation (default: 20)
        
    Returns:
        Tuple of (upper_band, lower_band) arrays
    """
    n = len(close_prices)
    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)
    
    # Small buffer scale - bands are just slightly beyond recent range
    buffer_scale = 0.1
    
    for i in range(1, n):  # Start from 1 since we need previous bars
        # Calculate rolling window bounds EXCLUDING current bar
        # Look at bars [i-lookback, i-1] (previous lookback bars)
        start_idx = max(0, i - lookback)
        end_idx = i  # Exclusive - does not include current bar
        
        if start_idx >= end_idx:
            continue
            
        # Find rolling high and low over the PREVIOUS window
        rolling_high = high_prices[start_idx]
        rolling_low = low_prices[start_idx]
        
        for j in range(start_idx + 1, end_idx):
            if high_prices[j] > rolling_high:
                rolling_high = high_prices[j]
            if low_prices[j] < rolling_low:
                rolling_low = low_prices[j]
        
        if not np.isnan(atr[i]):
            buffer = atr_mult * atr[i] * buffer_scale
            upper_band[i] = rolling_high + buffer
            lower_band[i] = rolling_low - buffer
    
    return upper_band, lower_band


class MomentumATR:
    """
    ATR-based Momentum (band breakout) strategy.
    
    This strategy uses Average True Range (ATR) as the volatility measure
    instead of sigma_open, providing a more adaptive volatility measure
    that accounts for gaps and intraday range.
    
    Strategy Logic:
        - Buy when price breaks above ATR-based upper band AND above VWAP
        - Exit when price falls below ATR-based lower band OR below VWAP
    
    Band Calculation:
        Upper Band = reference_high + (atr_mult × ATR)
        Lower Band = reference_low - (atr_mult × ATR)
    
    Key Differences from Standard Momentum:
        1. Uses ATR instead of sigma_open for volatility measurement
        2. ATR captures true price movement including gaps
        3. ATR adapts to changing market conditions via smoothing
        4. Band width is absolute (not percentage-based)
    
    Example:
        signals = MomentumATR.generate_signals(
            close_prices=close,
            high_prices=high,
            low_prices=low,
            vwap=vwap,
            reference_high=prev_high,
            reference_low=prev_low,
            atr_mult=2.0,
            atr_period=14
        )
    """

    def __init__(
        self,
        atr_mult: List[float] = [1.5, 2.0, 2.5, 3.0],
        atr_period: List[int] = [10, 14, 20]
    ):
        """
        Initialize the ATR momentum strategy class with default parameters.
        
        Args:
            atr_mult: List of ATR multipliers to test. Higher = wider bands,
                     fewer signals. Lower = tighter bands, more signals.
            atr_period: List of ATR periods to test. Shorter = more responsive
                       to recent volatility. Longer = smoother, less reactive.
        """
        self.atr_mult = atr_mult
        self.atr_period = atr_period


    def get_parameter_grid(self) -> Dict[str, list]:
        """
        Get parameter grid for ATR momentum strategy.
        
        Returns:
            Dictionary of parameter names to lists of values to test
        """
        return {
            'atr_multipliers': self.atr_mult,
            'atr_periods': self.atr_period
        }


    @staticmethod
    def generate_signals(
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        vwap: np.ndarray,
        reference_high: float,
        reference_low: float,
        atr_mult: float,
        atr_period: int = 14
    ) -> np.ndarray:
        """
        Vectorized signal generation based on ATR band breakouts.
        
        Args:
            close_prices: Array of close prices
            high_prices: Array of high prices
            low_prices: Array of low prices
            vwap: Array of VWAP values
            reference_high: Reference price for upper band anchor
            reference_low: Reference price for lower band anchor
            atr_mult: ATR multiplier for band width
            atr_period: Period for ATR calculation (default: 14)
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
        """
        n = len(close_prices)
        
        # Calculate ATR
        atr = calculate_atr_numba(
            high_prices.astype(np.float64),
            low_prices.astype(np.float64),
            close_prices.astype(np.float64),
            atr_period
        )
        
        # Calculate ATR-based bands
        upper_band, lower_band = calculate_atr_bands_numba(
            close_prices.astype(np.float64),
            atr,
            reference_high,
            atr_mult
        )
        
        # Lower band uses reference_low as anchor
        lower_band_adj = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if not np.isnan(atr[i]):
                lower_band_adj[i] = reference_low - (atr_mult * atr[i])
        
        # Initialize signals
        signals = np.zeros(n, dtype=np.int32)
        
        # Long signal: price breaks above upper band AND above VWAP
        # Exit signal: price falls below lower band OR below VWAP
        valid_mask = ~np.isnan(upper_band) & ~np.isnan(lower_band_adj)
        
        long_condition = valid_mask & (close_prices > upper_band) & (close_prices > vwap)
        exit_condition = valid_mask & ((close_prices < lower_band_adj) | (close_prices < vwap))
        
        signals[long_condition] = 1
        signals[exit_condition] = -1
        
        return signals


    @staticmethod
    def generate_signals_trailing(
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        vwap: np.ndarray,
        atr_mult: float,
        atr_period: int = 14,
        lookback: int = 20
    ) -> np.ndarray:
        """
        Vectorized signal generation with trailing ATR bands.
        
        This variant uses rolling high/low over a lookback window as band 
        anchors instead of fixed reference prices, making bands dynamically
        trail the price action. Bands can both expand AND contract.
        
        Args:
            close_prices: Array of close prices
            high_prices: Array of high prices
            low_prices: Array of low prices
            vwap: Array of VWAP values
            atr_mult: ATR multiplier for band width
            atr_period: Period for ATR calculation (default: 14)
            lookback: Rolling window for high/low calculation (default: 20)
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
        """
        n = len(close_prices)
        
        # Calculate ATR
        atr = calculate_atr_numba(
            high_prices.astype(np.float64),
            low_prices.astype(np.float64),
            close_prices.astype(np.float64),
            atr_period
        )
        
        # Calculate trailing ATR bands with rolling window
        upper_band, lower_band = calculate_trailing_atr_bands_numba(
            close_prices.astype(np.float64),
            high_prices.astype(np.float64),
            low_prices.astype(np.float64),
            atr,
            atr_mult,
            lookback
        )
        
        # Initialize signals
        signals = np.zeros(n, dtype=np.int32)
        
        # Long signal: price breaks above upper band AND above VWAP
        # Exit signal: price falls below lower band OR below VWAP
        valid_mask = ~np.isnan(upper_band) & ~np.isnan(lower_band)
        
        long_condition = valid_mask & (close_prices > upper_band) & (close_prices > vwap)
        exit_condition = valid_mask & ((close_prices < lower_band) | (close_prices < vwap))
        
        signals[long_condition] = 1
        signals[exit_condition] = -1
        
        return signals


    @staticmethod
    def generate_signals_with_indicators(
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        vwap: np.ndarray,
        reference_high: float,
        reference_low: float,
        atr_mult: float,
        atr_period: int = 14
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate signals with indicator data for analysis and visualization.
        
        Args:
            close_prices: Array of close prices
            high_prices: Array of high prices
            low_prices: Array of low prices
            vwap: Array of VWAP values
            reference_high: Reference price for upper band anchor
            reference_low: Reference price for lower band anchor
            atr_mult: ATR multiplier for band width
            atr_period: Period for ATR calculation (default: 14)
            
        Returns:
            Tuple of (signals, atr, upper_band, lower_band)
        """
        n = len(close_prices)
        
        # Calculate ATR
        atr = calculate_atr_numba(
            high_prices.astype(np.float64),
            low_prices.astype(np.float64),
            close_prices.astype(np.float64),
            atr_period
        )
        
        # Calculate ATR-based bands
        upper_band = np.full(n, np.nan, dtype=np.float64)
        lower_band = np.full(n, np.nan, dtype=np.float64)
        
        for i in range(n):
            if not np.isnan(atr[i]):
                upper_band[i] = reference_high + (atr_mult * atr[i])
                lower_band[i] = reference_low - (atr_mult * atr[i])
        
        # Generate signals
        signals = np.zeros(n, dtype=np.int32)
        
        valid_mask = ~np.isnan(upper_band) & ~np.isnan(lower_band)
        long_condition = valid_mask & (close_prices > upper_band) & (close_prices > vwap)
        exit_condition = valid_mask & ((close_prices < lower_band) | (close_prices < vwap))
        
        signals[long_condition] = 1
        signals[exit_condition] = -1
        
        return signals, atr, upper_band, lower_band


    @staticmethod
    def generate_signals_keltner_style(
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        vwap: np.ndarray,
        atr_mult: float,
        atr_period: int = 14,
        ema_period: int = 20
    ) -> np.ndarray:
        """
        Keltner Channel style ATR momentum signals.
        
        Uses EMA as the channel center instead of fixed reference prices.
        This creates dynamic bands that follow the price trend.
        
        Upper Band = EMA + (atr_mult × ATR)
        Lower Band = EMA - (atr_mult × ATR)
        
        Args:
            close_prices: Array of close prices
            high_prices: Array of high prices
            low_prices: Array of low prices
            vwap: Array of VWAP values
            atr_mult: ATR multiplier for band width
            atr_period: Period for ATR calculation (default: 14)
            ema_period: Period for EMA calculation (default: 20)
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
        """
        n = len(close_prices)
        
        # Calculate EMA
        ema = np.full(n, np.nan, dtype=np.float64)
        multiplier = 2.0 / (ema_period + 1.0)
        
        # Initialize EMA with SMA
        if n >= ema_period:
            sma_sum = 0.0
            for i in range(ema_period):
                sma_sum += close_prices[i]
            ema[ema_period - 1] = sma_sum / ema_period
            
            for i in range(ema_period, n):
                ema[i] = (close_prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
        
        # Calculate ATR
        atr = calculate_atr_numba(
            high_prices.astype(np.float64),
            low_prices.astype(np.float64),
            close_prices.astype(np.float64),
            atr_period
        )
        
        # Calculate Keltner-style bands
        upper_band = np.full(n, np.nan, dtype=np.float64)
        lower_band = np.full(n, np.nan, dtype=np.float64)
        
        for i in range(n):
            if not np.isnan(ema[i]) and not np.isnan(atr[i]):
                upper_band[i] = ema[i] + (atr_mult * atr[i])
                lower_band[i] = ema[i] - (atr_mult * atr[i])
        
        # Initialize signals
        signals = np.zeros(n, dtype=np.int32)
        
        valid_mask = ~np.isnan(upper_band) & ~np.isnan(lower_band)
        long_condition = valid_mask & (close_prices > upper_band) & (close_prices > vwap)
        exit_condition = valid_mask & ((close_prices < lower_band) | (close_prices < vwap))
        
        signals[long_condition] = 1
        signals[exit_condition] = -1
        
        return signals
