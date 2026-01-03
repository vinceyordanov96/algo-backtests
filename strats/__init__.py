# algo-backtests/strats/__init__.py

"""
Trading Strategies Module.

This module provides various trading strategy implementations:
    - Momentum: Band breakout strategy
    - Mean Reversion: Z-score based reversion strategy
    - Mean Reversion RSI: RSI + SMA trend-filtered reversion
    - Statistical Arbitrage: Pairs trading with dynamic hedge ratio
    - Supervised ML: Machine learning based strategies (via ml submodule)

All strategies follow a consistent interface defined by BaseStrategy.

Factory and Types:
    - StrategyType: Enumeration of available strategy types
    - StrategyFactory: Factory class for strategy management

Example usage:
    from strats import Momentum, MeanReversion, StrategyType, StrategyFactory
    
    # Use static signal generation
    signals = Momentum.generate_signals(close_prices, vwap, sigma_open, ...)
    
    # Use factory for configuration
    config = StrategyFactory.get_default_config(StrategyType.MOMENTUM)
    signal_func = StrategyFactory.get_strategy(StrategyType.MOMENTUM)
"""

from .base import BaseStrategy, StatefulStrategy, PairsStrategy, SignalResult
from .factory import StrategyType, StrategyFactory
from .momentum import Momentum
from .mean_reversion import MeanReversion
from .mean_reversion_rsi import MeanReversionRSI
from .stat_arb import StatArb

# ML strategy
from .classification import SupervisedStrategy, generate_ml_signals

# Re-export ML modules for convenience (lazy imports to avoid circular deps)
def __getattr__(name):
    """Lazy import for ML modules to avoid circular imports."""
    ml_exports = {
        'ModelArtifacts': 'ml.model_artifacts',
        'ArtifactManager': 'ml.model_artifacts', 
        'FeatureGeneration': 'ml.feature_generation',
        'FeatureConfig': 'ml.feature_generation',
        'FeatureNormalization': 'ml.feature_normalization',
        'FeatureSelection': 'ml.feature_selection',
        'TrainingPipeline': 'ml.training_pipeline',
        'TrainingConfig': 'ml.training_config',
        'TrainingMetrics': 'ml.training_metrics',
        'TrainingResult': 'ml.training_results',
    }
    
    if name in ml_exports:
        import importlib
        module = importlib.import_module(ml_exports[name])
        return getattr(module, name)
    
    raise AttributeError(f"module 'strats' has no attribute '{name}'")


__all__ = [
    # Base classes
    'BaseStrategy',
    'StatefulStrategy', 
    'PairsStrategy',
    'SignalResult',
    
    # Factory and types
    'StrategyType',
    'StrategyFactory',
    
    # Strategy implementations
    'Momentum',
    'MeanReversion',
    'MeanReversionRSI',
    'StatArb',
    
    # ML strategy
    'SupervisedStrategy',
    'generate_ml_signals',
    
    # ML modules (lazy loaded)
    'ModelArtifacts',
    'ArtifactManager',
    'FeatureGeneration',
    'FeatureConfig',
    'FeatureNormalization',
    'FeatureSelection',
    'TrainingPipeline',
    'TrainingConfig',
    'TrainingMetrics',
    'TrainingResult',
]
