"""
Simulation configuration management.

This module provides configuration classes and parameter grid
management for simulation sweeps.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class StrategyType(Enum):
    """Strategy type enumeration."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    MEAN_REVERSION_RSI = "mean_reversion_rsi"
    STAT_ARB = "stat_arb"
    SUPERVISED = "supervised"


@dataclass
class CommonParameters:
    """Common parameters shared across all strategies."""
    
    # Trading parameters
    trade_frequencies: List[int] = field(default_factory=lambda: [15, 30, 60])
    target_volatilities: List[float] = field(default_factory=lambda: [0.015, 0.02, 0.025])
    
    # Risk parameters
    stop_loss_pcts: List[float] = field(default_factory=lambda: [0.02])
    take_profit_pcts: List[float] = field(default_factory=lambda: [0.04])
    max_drawdown_pcts: List[float] = field(default_factory=lambda: [0.15])
    
    # Position sizing parameters
    sizing_types: List[str] = field(default_factory=lambda: ['vol_target'])
    kelly_fractions: List[float] = field(default_factory=lambda: [0.5])
    kelly_lookbacks: List[int] = field(default_factory=lambda: [60])
    kelly_min_trades: int = 30
    kelly_vol_blend_weight: float = 0.5
    
    # Execution parameters
    commission: float = 0.0035
    slippage_factor: float = 0.1
    max_leverage: int = 1
    
    # Backtest parameters
    initial_aum: float = 100000.0
    lookback_period: int = 20
    var_confidence: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_frequencies': self.trade_frequencies,
            'target_volatilities': self.target_volatilities,
            'stop_loss_pcts': self.stop_loss_pcts,
            'take_profit_pcts': self.take_profit_pcts,
            'max_drawdown_pcts': self.max_drawdown_pcts,
            'sizing_types': self.sizing_types,
            'kelly_fractions': self.kelly_fractions,
            'kelly_lookbacks': self.kelly_lookbacks,
            'kelly_min_trades': self.kelly_min_trades,
            'kelly_vol_blend_weight': self.kelly_vol_blend_weight,
            'commission': self.commission,
            'slippage_factor': self.slippage_factor,
            'max_leverage': self.max_leverage,
            'initial_aum': self.initial_aum,
            'lookback_period': self.lookback_period,
            'var_confidence': self.var_confidence,
        }


@dataclass
class MomentumParameters:
    """Parameters specific to momentum strategy."""
    band_multipliers: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5])


@dataclass
class MeanReversionParameters:
    """Parameters specific to mean reversion (z-score) strategy."""
    zscore_lookbacks: List[int] = field(default_factory=lambda: [10, 20, 30])
    n_std_uppers: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    n_std_lowers: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    exit_thresholds: List[float] = field(default_factory=lambda: [0.0, 0.5])


@dataclass
class MeanReversionRSIParameters:
    """Parameters specific to RSI mean reversion strategy."""
    rsi_periods: List[int] = field(default_factory=lambda: [7, 10, 14])
    rsi_oversold_levels: List[float] = field(default_factory=lambda: [25.0, 30.0, 35.0])
    rsi_overbought_levels: List[float] = field(default_factory=lambda: [65.0, 70.0, 75.0])
    sma_periods: List[int] = field(default_factory=lambda: [100, 200])


@dataclass
class StatArbParameters:
    """Parameters specific to statistical arbitrage strategy."""
    zscore_lookbacks: List[int] = field(default_factory=lambda: [30, 60, 90])
    entry_thresholds: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    exit_thresholds: List[float] = field(default_factory=lambda: [0.0, 0.5])
    use_dynamic_hedge_options: List[bool] = field(default_factory=lambda: [True, False])


@dataclass
class SupervisedParameters:
    """Parameters specific to supervised ML strategy."""
    model_paths: List[str] = field(default_factory=list)
    scaler_paths: List[str] = field(default_factory=list)
    features_paths: List[str] = field(default_factory=list)
    buy_thresholds: List[float] = field(default_factory=lambda: [0.50, 0.55, 0.60, 0.65])
    sell_thresholds: List[float] = field(default_factory=lambda: [0.35, 0.40, 0.45, 0.50])


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.
    
    This class manages all parameters for a simulation sweep including:
        - Tickers/pairs to test
        - Common parameters (trading frequency, risk limits, etc.)
        - Strategy-specific parameters
    
    Example:
        config = SimulationConfig(
            strategy_type=StrategyType.MOMENTUM,
            tickers=['NVDA', 'AAPL'],
            common=CommonParameters(
                trade_frequencies=[15, 30],
                target_volatilities=[0.02]
            ),
            momentum=MomentumParameters(
                band_multipliers=[0.8, 1.0, 1.2]
            )
        )
    """
    
    strategy_type: StrategyType = StrategyType.MOMENTUM
    
    # Tickers/pairs
    tickers: List[str] = field(default_factory=lambda: ['NVDA'])
    pairs: List[Tuple[str, str]] = field(default_factory=lambda: [('NVDA', 'AMD')])
    
    # Parameter groups
    common: CommonParameters = field(default_factory=CommonParameters)
    momentum: MomentumParameters = field(default_factory=MomentumParameters)
    mean_reversion: MeanReversionParameters = field(default_factory=MeanReversionParameters)
    mean_reversion_rsi: MeanReversionRSIParameters = field(default_factory=MeanReversionRSIParameters)
    stat_arb: StatArbParameters = field(default_factory=StatArbParameters)
    supervised: SupervisedParameters = field(default_factory=SupervisedParameters)
    
    # Parallel execution settings
    n_workers: Optional[int] = None  # None = auto (CPU count - 2)
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get parameters for the configured strategy type."""
        if self.strategy_type == StrategyType.MOMENTUM:
            return {
                'band_multipliers': self.momentum.band_multipliers,
            }
        elif self.strategy_type == StrategyType.MEAN_REVERSION:
            return {
                'zscore_lookbacks': self.mean_reversion.zscore_lookbacks,
                'n_std_uppers': self.mean_reversion.n_std_uppers,
                'n_std_lowers': self.mean_reversion.n_std_lowers,
                'exit_thresholds': self.mean_reversion.exit_thresholds,
            }
        elif self.strategy_type == StrategyType.MEAN_REVERSION_RSI:
            return {
                'rsi_periods': self.mean_reversion_rsi.rsi_periods,
                'rsi_oversold_levels': self.mean_reversion_rsi.rsi_oversold_levels,
                'rsi_overbought_levels': self.mean_reversion_rsi.rsi_overbought_levels,
                'sma_periods': self.mean_reversion_rsi.sma_periods,
            }
        elif self.strategy_type == StrategyType.STAT_ARB:
            return {
                'zscore_lookbacks': self.stat_arb.zscore_lookbacks,
                'entry_thresholds': self.stat_arb.entry_thresholds,
                'exit_thresholds': self.stat_arb.exit_thresholds,
                'use_dynamic_hedge_options': self.stat_arb.use_dynamic_hedge_options,
            }
        elif self.strategy_type == StrategyType.SUPERVISED:
            return {
                'model_paths': self.supervised.model_paths,
                'scaler_paths': self.supervised.scaler_paths,
                'features_paths': self.supervised.features_paths,
                'buy_thresholds': self.supervised.buy_thresholds,
                'sell_thresholds': self.supervised.sell_thresholds,
            }
        else:
            return {}
    
    def count_combinations(self) -> int:
        """
        Estimate the number of parameter combinations.
        
        Returns:
            Estimated number of backtest configurations
        """
        common = self.common
        base_count = (
            len(common.trade_frequencies) *
            len(common.target_volatilities) *
            len(common.stop_loss_pcts) *
            len(common.take_profit_pcts) *
            len(common.max_drawdown_pcts) *
            len(common.sizing_types) *
            len(common.kelly_fractions) *
            len(common.kelly_lookbacks)
        )
        
        if self.strategy_type == StrategyType.MOMENTUM:
            strategy_count = len(self.momentum.band_multipliers)
            ticker_count = len(self.tickers)
        elif self.strategy_type == StrategyType.MEAN_REVERSION:
            mr = self.mean_reversion
            strategy_count = (
                len(mr.zscore_lookbacks) *
                len(mr.n_std_uppers) *
                len(mr.n_std_lowers) *
                len(mr.exit_thresholds)
            )
            ticker_count = len(self.tickers)
        elif self.strategy_type == StrategyType.MEAN_REVERSION_RSI:
            rsi = self.mean_reversion_rsi
            strategy_count = (
                len(rsi.rsi_periods) *
                len(rsi.rsi_oversold_levels) *
                len(rsi.rsi_overbought_levels) *
                len(rsi.sma_periods)
            )
            ticker_count = len(self.tickers)
        elif self.strategy_type == StrategyType.STAT_ARB:
            sa = self.stat_arb
            strategy_count = (
                len(sa.zscore_lookbacks) *
                len(sa.entry_thresholds) *
                len(sa.exit_thresholds) *
                len(sa.use_dynamic_hedge_options)
            )
            ticker_count = len(self.pairs)
        elif self.strategy_type == StrategyType.SUPERVISED:
            sup = self.supervised
            strategy_count = (
                max(1, len(sup.model_paths)) *
                len(sup.buy_thresholds) *
                len(sup.sell_thresholds)
            )
            ticker_count = len(self.tickers)
        else:
            strategy_count = 1
            ticker_count = len(self.tickers)
        
        return base_count * strategy_count * ticker_count
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create configuration from dictionary."""
        strategy_type = data.get('strategy_type', 'momentum')
        if isinstance(strategy_type, str):
            strategy_type = StrategyType(strategy_type)
        
        config = cls(strategy_type=strategy_type)
        
        if 'tickers' in data:
            config.tickers = data['tickers']
        if 'pairs' in data:
            config.pairs = data['pairs']
        if 'n_workers' in data:
            config.n_workers = data['n_workers']
        
        # Common parameters
        common_data = data.get('common', {})
        for key, value in common_data.items():
            if hasattr(config.common, key):
                setattr(config.common, key, value)
        
        # Strategy-specific parameters
        if strategy_type == StrategyType.MOMENTUM:
            momentum_data = data.get('momentum', {})
            for key, value in momentum_data.items():
                if hasattr(config.momentum, key):
                    setattr(config.momentum, key, value)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            mr_data = data.get('mean_reversion', {})
            for key, value in mr_data.items():
                if hasattr(config.mean_reversion, key):
                    setattr(config.mean_reversion, key, value)
        elif strategy_type == StrategyType.MEAN_REVERSION_RSI:
            rsi_data = data.get('mean_reversion_rsi', {})
            for key, value in rsi_data.items():
                if hasattr(config.mean_reversion_rsi, key):
                    setattr(config.mean_reversion_rsi, key, value)
        elif strategy_type == StrategyType.STAT_ARB:
            sa_data = data.get('stat_arb', {})
            for key, value in sa_data.items():
                if hasattr(config.stat_arb, key):
                    setattr(config.stat_arb, key, value)
        elif strategy_type == StrategyType.SUPERVISED:
            sup_data = data.get('supervised', {})
            for key, value in sup_data.items():
                if hasattr(config.supervised, key):
                    setattr(config.supervised, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'strategy_type': self.strategy_type.value,
            'tickers': self.tickers,
            'pairs': self.pairs,
            'n_workers': self.n_workers,
            'common': self.common.to_dict(),
            'momentum': {
                'band_multipliers': self.momentum.band_multipliers,
            },
            'mean_reversion': {
                'zscore_lookbacks': self.mean_reversion.zscore_lookbacks,
                'n_std_uppers': self.mean_reversion.n_std_uppers,
                'n_std_lowers': self.mean_reversion.n_std_lowers,
                'exit_thresholds': self.mean_reversion.exit_thresholds,
            },
            'mean_reversion_rsi': {
                'rsi_periods': self.mean_reversion_rsi.rsi_periods,
                'rsi_oversold_levels': self.mean_reversion_rsi.rsi_oversold_levels,
                'rsi_overbought_levels': self.mean_reversion_rsi.rsi_overbought_levels,
                'sma_periods': self.mean_reversion_rsi.sma_periods,
            },
            'stat_arb': {
                'zscore_lookbacks': self.stat_arb.zscore_lookbacks,
                'entry_thresholds': self.stat_arb.entry_thresholds,
                'exit_thresholds': self.stat_arb.exit_thresholds,
                'use_dynamic_hedge_options': self.stat_arb.use_dynamic_hedge_options,
            },
            'supervised': {
                'model_paths': self.supervised.model_paths,
                'scaler_paths': self.supervised.scaler_paths,
                'features_paths': self.supervised.features_paths,
                'buy_thresholds': self.supervised.buy_thresholds,
                'sell_thresholds': self.supervised.sell_thresholds,
            },
        }
