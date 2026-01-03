"""
Simulation module for parameter sweep backtesting.

This module provides infrastructure for running backtests across
multiple parameter combinations in parallel:
    - SimulationRunner: Parallel execution manager
    - TaskGenerator: Strategy-specific task generation
    - SimulationConfig: Configuration management

Example:
    from simulation import SimulationRunner, SimulationConfig
    
    config = SimulationConfig(
        strategy_type='momentum',
        tickers=['NVDA', 'AAPL'],
        band_multipliers=[0.8, 1.0, 1.2],
        trade_frequencies=[15, 30, 60]
    )
    
    runner = SimulationRunner(config)
    results = runner.run()
    
    # Get best configurations
    best = runner.get_best_by_ticker(results)
    
    # Run best strategy and save all outputs
    runner.run_best_strategy(results, 'NVDA', save_outputs=True)
    
    # Or run for all tickers
    runner.run_all_best_strategies(results, save_outputs=True)
"""

from .runner import SimulationRunner
from .tasks import TaskGenerator
from .config import SimulationConfig, StrategyType

__all__ = [
    'SimulationRunner',
    'TaskGenerator',
    'SimulationConfig',
    'StrategyType',
]
