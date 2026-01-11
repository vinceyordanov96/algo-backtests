"""
Strategy factory and configuration utilities.

This module provides:
    - StrategyType: Enumeration of available strategies
    - StrategyFactory: Factory class for creating and managing strategies
    - Configuration validation and default generation
"""

from typing import Dict, Any, Callable, List
from enum import Enum

from .momentum import Momentum
from .momentum_atr import MomentumATR
from .mean_reversion import MeanReversion
from .mean_reversion_rsi import MeanReversionRSI
from .stat_arb import StatArb


class StrategyType(Enum):
    """
    Enumeration of available trading strategies.
    
    Values:
        MOMENTUM: Trend-following strategy based on price momentum (sigma_open)
        MOMENTUM_ATR: ATR-based trend-following strategy
        MEAN_REVERSION: Z-score based mean reversion
        MEAN_REVERSION_RSI: RSI + SMA mean reversion
        STAT_ARB: Statistical arbitrage (pairs trading)
        SUPERVISED: Machine learning based strategy
    """
    MOMENTUM = "momentum"
    MOMENTUM_ATR = "momentum_atr"
    MEAN_REVERSION = "mean_reversion"
    MEAN_REVERSION_RSI = "mean_reversion_rsi"
    STAT_ARB = "stat_arb"
    SUPERVISED = "supervised"


class StrategyFactory:
    """
    Factory class for creating and managing trading strategies.
    
    Provides a consistent interface for signal generation across 
    different strategy types and parameter grid generation for optimization.
    
    Example:
        # Get signal generation function
        signal_func = StrategyFactory.get_strategy(StrategyType.MOMENTUM)
        signals = signal_func(prices, vwap, sigma, ref_high, ref_low, band_mult)
        
        # Get default configuration
        config = StrategyFactory.get_default_config(StrategyType.MOMENTUM)
        
        # Validate configuration
        StrategyFactory.validate_config(config, StrategyType.MOMENTUM)
    """
    
    _strategies: Dict[StrategyType, Callable] = {
        StrategyType.MOMENTUM: Momentum.generate_signals,
        StrategyType.MOMENTUM_ATR: MomentumATR.generate_signals,
        StrategyType.MEAN_REVERSION: MeanReversion.generate_signals,
        StrategyType.MEAN_REVERSION_RSI: MeanReversionRSI.generate_signals,
        StrategyType.STAT_ARB: StatArb.generate_signals,
        # SUPERVISED strategy uses SupervisedStrategy class directly
    }
    
    @classmethod
    def get_strategy(cls, strategy_type: StrategyType) -> Callable:
        """
        Get the signal generation function for a given strategy type.
        
        Args:
            strategy_type: The type of strategy to retrieve
            
        Returns:
            Signal generation function
            
        Raises:
            ValueError: If strategy type is not registered or is SUPERVISED
        """
        if strategy_type == StrategyType.SUPERVISED:
            raise ValueError(
                "SUPERVISED strategy should be instantiated via SupervisedStrategy class, "
                "not through StrategyFactory.get_strategy(). "
                "Use: from ml import SupervisedStrategy"
            )
        
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return cls._strategies[strategy_type]
    
    @classmethod
    def register_strategy(
        cls,
        strategy_type: StrategyType,
        strategy_func: Callable
    ) -> None:
        """
        Register a new strategy or override an existing one.
        
        Args:
            strategy_type: The type identifier for the strategy
            strategy_func: The signal generation function
        """
        cls._strategies[strategy_type] = strategy_func
    
    @classmethod
    def list_strategies(cls) -> List[StrategyType]:
        """
        List all available strategy types.
        
        Returns:
            List of available StrategyType values
        """
        return list(StrategyType)
    
    @classmethod
    def get_strategy_class(cls, strategy_type: StrategyType):
        """
        Get the strategy class for a given strategy type.
        
        Args:
            strategy_type: The type of strategy
            
        Returns:
            Strategy class
        """
        strategy_classes = {
            StrategyType.MOMENTUM: Momentum,
            StrategyType.MOMENTUM_ATR: MomentumATR,
            StrategyType.MEAN_REVERSION: MeanReversion,
            StrategyType.MEAN_REVERSION_RSI: MeanReversionRSI,
            StrategyType.STAT_ARB: StatArb,
        }
        
        if strategy_type == StrategyType.SUPERVISED:
            # Lazy import to avoid circular dependencies
            from ml import SupervisedStrategy
            return SupervisedStrategy
        
        if strategy_type not in strategy_classes:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategy_classes[strategy_type]

    @staticmethod
    def get_common_parameter_grid() -> Dict[str, List]:
        """
        Get default parameter grid for common parameters.
        
        Returns:
            Dictionary of parameter names to lists of values
        """
        return {
            'tickers': ['NVDA'],
            'pairs': [('NVDA', 'AMD')],
            'trade_frequencies': [30, 45, 60, 90],
            'stop_loss_pcts': [0.01, 0.02, 0.03],
            'take_profit_pcts': [0.04, 0.05, 0.06],
            'max_drawdown_pcts': [0.10, 0.15, 0.20],
            'target_volatilities': [0.015, 0.02, 0.025]
        }

    @staticmethod
    def get_strategy_parameter_grid(strategy_type: StrategyType) -> Dict[str, List]:
        """
        Get default parameter grid for strategy-specific parameters.
        
        Args:
            strategy_type: The strategy type
            
        Returns:
            Dictionary of parameter names to lists of values
        """
        grids = {
            StrategyType.MOMENTUM: {
                'band_mult': [0.8, 1.0, 1.2, 1.5],
            },
            StrategyType.MOMENTUM_ATR: {
                'atr_mult': [1.5, 2.0, 2.5, 3.0],
                'atr_period': [10, 14, 20],
            },
            StrategyType.MEAN_REVERSION: {
                'zscore_lookback': [10, 20, 30],
                'n_std_upper': [1.5, 2.0, 2.5],
                'n_std_lower': [1.5, 2.0, 2.5],
                'exit_threshold': [0.0, 0.5],
            },
            StrategyType.MEAN_REVERSION_RSI: {
                'rsi_period': [7, 10, 14],
                'rsi_oversold': [25.0, 30.0, 35.0],
                'rsi_overbought': [65.0, 70.0, 75.0],
                'sma_period': [100, 200],
            },
            StrategyType.STAT_ARB: {
                'zscore_lookback': [30, 60, 90],
                'entry_threshold': [1.5, 2.0, 2.5],
                'exit_threshold': [0.0, 0.5],
                'use_dynamic_hedge': [True, False],
            },
            StrategyType.SUPERVISED: {
                'buy_threshold': [0.50, 0.55, 0.60, 0.65],
                'sell_threshold': [0.35, 0.40, 0.45, 0.50],
            },
        }
        
        return grids.get(strategy_type, {})

    @staticmethod
    def validate_config(
        config: Dict[str, Any],
        strategy_type: StrategyType
    ) -> bool:
        """
        Validate configuration parameters for a given strategy.
        
        Args:
            config: Configuration dictionary
            strategy_type: The strategy type to validate for
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If required parameters are missing
        """
        required_common = ['AUM', 'trade_freq', 'stop_loss_pct', 'take_profit_pct']
        
        required_by_strategy = {
            StrategyType.MOMENTUM: ['band_mult'],
            StrategyType.MOMENTUM_ATR: ['atr_mult', 'atr_period'],
            StrategyType.MEAN_REVERSION: ['zscore_lookback', 'n_std_upper', 'n_std_lower'],
            StrategyType.MEAN_REVERSION_RSI: ['rsi_period', 'rsi_oversold', 'rsi_overbought', 'sma_period'],
            StrategyType.STAT_ARB: ['zscore_lookback', 'entry_threshold'],
            StrategyType.SUPERVISED: ['model_path', 'scaler_path', 'features_path'],
        }

        # Check common parameters
        for param in required_common:
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Check strategy-specific parameters
        required_strategy = required_by_strategy.get(strategy_type, [])
        for param in required_strategy:
            if param not in config:
                raise ValueError(
                    f"Missing required {strategy_type.value} parameter: {param}"
                )
        
        return True

    @staticmethod
    def get_default_config(
        strategy_type: StrategyType,
        **overrides
    ) -> Dict[str, Any]:
        """
        Get a default configuration for a given strategy type.
        
        Args:
            strategy_type: The strategy type
            **overrides: Parameters to override defaults
            
        Returns:
            Configuration dictionary with defaults
        """
        # Common defaults
        config = {
            'AUM': 100000.0,
            'commission': 0.0035,
            'min_comm_per_order': 0.35,
            'trade_freq': 30,
            'sizing_type': 'vol_target',
            'target_vol': 0.02,
            'kelly_fraction': 0.5,
            'kelly_lookback': 60,
            'kelly_min_trades': 30,
            'kelly_vol_blend_weight': 0.5,
            'max_leverage': 1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'max_drawdown_pct': 0.15,
            'lookback_period': 20,
            'slippage_factor': 0.1,
            'var_confidence': 0.95,
            'strategy_type': strategy_type
        }
        
        # Strategy-specific defaults
        strategy_defaults = {
            StrategyType.MOMENTUM: {
                'band_mult': 1.0,
            },
            StrategyType.MOMENTUM_ATR: {
                'atr_mult': 2.0,
                'atr_period': 14,
            },
            StrategyType.MEAN_REVERSION: {
                'zscore_lookback': 20,
                'n_std_upper': 2.0,
                'n_std_lower': 2.0,
                'exit_threshold': 0.0,
            },
            StrategyType.MEAN_REVERSION_RSI: {
                'rsi_period': 10,
                'rsi_oversold': 30.0,
                'rsi_overbought': 70.0,
                'sma_period': 200,
                'exit_on_sma_cross': True,
            },
            StrategyType.STAT_ARB: {
                'zscore_lookback': 60,
                'entry_threshold': 2.0,
                'exit_threshold': 0.0,
                'use_dynamic_hedge': True,
                'hedge_ratio': None,
            },
            StrategyType.SUPERVISED: {
                'model_path': None,
                'scaler_path': None,
                'features_path': None,
                'buy_threshold': 0.55,
                'sell_threshold': 0.45,
            },
        }
        
        # Add strategy-specific defaults
        config.update(strategy_defaults.get(strategy_type, {}))
        
        # Apply overrides
        config.update(overrides)
        
        return config
