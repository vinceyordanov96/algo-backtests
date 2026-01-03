"""
Abstract Base Strategy Class.

All trading strategies should inherit from this base class to ensure
consistent interface across momentum, mean reversion, stat arb, and ML strategies.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SignalResult:
    """
    Container for signal generation results.
    
    Provides a consistent return type across all strategies,
    with optional additional data for strategy-specific metrics.
    """
    signals: np.ndarray
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Defines the interface that all strategies must implement for
    compatibility with the backtesting and simulation infrastructure.
    
    Key Design Principles:
        1. Stateless signal generation (no look-ahead bias)
        2. Consistent interface for parameter optimization
        3. Factory pattern for configuration-based instantiation
        4. Support for both array-based and DataFrame-based inputs
    
    Example Implementation:
        class MyStrategy(BaseStrategy):
            def __init__(self, param1: float = 1.0):
                self.param1 = param1
            
            def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
                # Generate signals based on data
                return signals
            
            def get_parameter_grid(self) -> Dict[str, List[Any]]:
                return {'param1': [0.5, 1.0, 1.5]}
            
            @classmethod
            def from_config(cls, config: Dict[str, Any]) -> 'MyStrategy':
                return cls(param1=config.get('param1', 1.0))
    """
    
    # Class-level constants for signal values
    SIGNAL_LONG = 1
    SIGNAL_EXIT = -1
    SIGNAL_HOLD = 0
    
    @abstractmethod
    def generate_signals(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """
        Generate trading signals from input data.
        
        This is the primary method called during backtesting. It must:
        1. Process input data without look-ahead bias
        2. Return signals in a consistent format
        3. Handle edge cases (insufficient data, NaN values)
        
        Args:
            data: Input data - either DataFrame with OHLCV columns or
                  numpy array of prices
            **kwargs: Strategy-specific parameters
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
        """
        pass
    
    @abstractmethod
    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Get parameter grid for strategy optimization.
        
        Returns a dictionary mapping parameter names to lists of values
        to test during optimization/simulation sweeps.
        
        Returns:
            Dictionary of parameter names to lists of values
            
        Example:
            {
                'lookback_period': [10, 20, 30],
                'threshold': [1.5, 2.0, 2.5]
            }
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseStrategy':
        """
        Factory method to create strategy instance from configuration.
        
        Allows strategies to be instantiated from configuration dictionaries,
        enabling dynamic strategy creation during simulations.
        
        Args:
            config: Dictionary containing strategy parameters
            
        Returns:
            Initialized strategy instance
        """
        pass
    
    def generate_signals_for_backtest(
        self,
        current_day_info: Dict[str, Any],
        precomputed_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Generate signals from BackTest's precomputed data format.
        
        Provides compatibility with the BackTest class's internal
        data structures. Default implementation builds a DataFrame
        from the day info and calls generate_signals().
        
        Override this method if your strategy needs special handling
        of the backtest data format.
        
        Args:
            current_day_info: Dictionary with precomputed day data:
                - close_prices: np.ndarray
                - volumes: np.ndarray
                - high_prices: np.ndarray
                - low_prices: np.ndarray
                - vwap: np.ndarray (optional)
                - open: float
                - min_from_open: np.ndarray (optional)
            precomputed_data: Full precomputed data dict (for context)
            
        Returns:
            Array of signals for the day
        """
        # Build DataFrame from precomputed arrays
        close_prices = current_day_info['close_prices']
        n = len(close_prices)
        
        df = pd.DataFrame({
            'close': close_prices,
            'volume': current_day_info.get('volumes', np.zeros(n)),
            'open': np.full(n, current_day_info.get('open', close_prices[0])),
            'high': current_day_info.get('high_prices', np.maximum.accumulate(close_prices)),
            'low': current_day_info.get('low_prices', np.minimum.accumulate(close_prices)),
        })
        
        # Add optional columns
        if 'vwap' in current_day_info:
            df['vwap'] = current_day_info['vwap']
        
        if 'min_from_open' in current_day_info:
            df['min_from_open'] = current_day_info['min_from_open']
        
        return self.generate_signals(df)
    
    def validate_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        min_length: int = 1
    ) -> bool:
        """
        Validate input data meets minimum requirements.
        
        Args:
            data: Input data to validate
            min_length: Minimum required data length
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If data is invalid
        """
        if isinstance(data, pd.DataFrame):
            if len(data) < min_length:
                raise ValueError(
                    f"Insufficient data: got {len(data)}, need {min_length}"
                )
            return True
        elif isinstance(data, np.ndarray):
            if len(data) < min_length:
                raise ValueError(
                    f"Insufficient data: got {len(data)}, need {min_length}"
                )
            return True
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def get_min_required_bars(self) -> int:
        """
        Get minimum number of bars required for signal generation.
        
        Override this method to specify the minimum lookback period
        needed for your strategy to generate valid signals.
        
        Returns:
            Minimum number of bars required
        """
        return 1
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the strategy for logging/debugging.
        
        Returns:
            Dictionary with strategy metadata
        """
        return {
            'name': self.__class__.__name__,
            'min_required_bars': self.get_min_required_bars(),
            'parameter_grid': self.get_parameter_grid()
        }


class StatefulStrategy(BaseStrategy):
    """
    Base class for strategies that maintain state across bars.
    
    Some strategies need to track state (e.g., position, entry price)
    during signal generation. This base class provides infrastructure
    for state management.
    
    Note: State should be reset between days/sessions to avoid
    look-ahead bias in backtesting.
    """
    
    def __init__(self):
        super().__init__()
        self._state: Dict[str, Any] = {}
    
    def reset_state(self) -> None:
        """Reset strategy state. Call between trading sessions."""
        self._state = {}
    
    def get_state(self) -> Dict[str, Any]:
        """Get current strategy state."""
        return self._state.copy()
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set strategy state."""
        self._state = state.copy()


class PairsStrategy(BaseStrategy):
    """
    Base class for pairs/spread trading strategies.
    
    Extends BaseStrategy to handle two-asset signal generation
    required for statistical arbitrage and pairs trading.
    """
    
    @abstractmethod
    def generate_signals_pair(
        self,
        data_a: Union[pd.DataFrame, np.ndarray],
        data_b: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate signals for a pair of assets.
        
        Args:
            data_a: Data for first asset
            data_b: Data for second asset
            **kwargs: Strategy-specific parameters
            
        Returns:
            Tuple of (signals, spread_values, hedge_ratios)
        """
        pass
    
    def generate_signals(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """
        Single-asset signal generation not supported for pairs strategies.
        
        Raises:
            NotImplementedError: Pairs strategies require two assets
        """
        raise NotImplementedError(
            "Pairs strategies require generate_signals_pair() with two assets"
        )
