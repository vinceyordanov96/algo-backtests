import numpy as np
from typing import Dict, Any, List


class Momentum:
    """
    Momentum (band breakout) strategy.
    
    Uses static methods for stateless signal generation,
    compatible with vectorized backtesting.
    """

    def __init__(
        self,
        band_mult: List[float] = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
    ):
        """
        Initialize the momentum strategy Class with default parameters.
        
        Args:
            band_mult: List of band multipliers to test (default: [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]).
        """
        self.band_mult = band_mult
    
    def get_parameter_grid(self) -> Dict[str, list]:
        """
        Get parameter grid for momentum strategy.
        
        Returns:
            Dictionary of parameter names to lists of values to test
        """
        return {
            'band_multipliers': self.band_mult
        }


    @staticmethod
    def generate_signals(
        close_prices: np.ndarray,
        vwap: np.ndarray,
        sigma_open: np.ndarray,
        reference_high: float,
        reference_low: float,
        band_mult: float
    ) -> np.ndarray:
        """
        Vectorized signal generation based on band breakouts (momentum strategy).
        
        Args:
            close_prices: Array of close prices
            vwap: Array of VWAP values
            sigma_open: Array of sigma values for band calculation
            reference_high: Reference price for upper band
            reference_low: Reference price for lower band
            band_mult: Band multiplier
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
        """
        # Calculate bands
        UB = reference_high * (1 + band_mult * sigma_open)
        LB = reference_low * (1 - band_mult * sigma_open)
        
        # Initialize signals
        signals = np.zeros(len(close_prices), dtype=np.int32)
        
        # Long signal: price breaks above upper band AND above VWAP
        # Exit signal: price falls below lower band OR below VWAP
        long_condition = (close_prices > UB) & (close_prices > vwap)
        exit_condition = (close_prices < LB) | (close_prices < vwap)
        
        signals[long_condition] = 1
        signals[exit_condition] = -1
        
        return signals
