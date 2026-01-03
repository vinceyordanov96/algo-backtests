# algo-backtests/backtesting/__init__.py
"""
Backtesting module for strategy evaluation.

This module provides the core backtesting infrastructure:
    - BacktestEngine: Main backtest execution loop
    - BacktestResults: Results container with analysis methods
    - BacktestConfig: Configuration management

Example:
    from backtesting import BacktestEngine, BacktestConfig
    
    config = BacktestConfig(
        strategy_type=StrategyType.MOMENTUM,
        initial_aum=100000.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    engine = BacktestEngine(ticker='NVDA')
    results = engine.run(df_intraday, df_daily, config)
    
    # Analyze results
    stats = results.calculate_statistics()
    results.plot()
"""

from .engine import BacktestEngine
from .results import BacktestResults, BacktestStatistics

__all__ = [
    'BacktestEngine',
    'BacktestResults',
    'BacktestStatistics',
]
