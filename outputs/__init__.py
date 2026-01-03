"""
Output management module.

This module provides unified output management for all strategy artifacts:
    - Simulation results (parameter sweep CSVs)
    - Strategy signals (best performing strategy signals)
    - Portfolio results (trade-by-trade updates)
    - Benchmark plots (strategy vs benchmark visualization)

Directory Structure:
    outputs/
    ├── data/                           # Market data (managed by DataManager)
    │   ├── intraday/{ticker}/
    │   ├── daily/{ticker}/
    │   └── dividends/{ticker}/
    ├── models/                         # ML models (managed by ArtifactManager)
    │   └── {ticker}/
    └── strategies/                     # Strategy outputs (managed by StrategyOutputManager)
        └── {strategy}/
            ├── simulations/{ticker}/   # Parameter sweep results
            ├── signals/{ticker}/       # Best strategy signals
            ├── results/{ticker}/       # Trade-by-trade portfolio updates
            └── plots/{ticker}/         # Benchmark comparison plots

Example:
    from outputs import StrategyOutputManager
    
    manager = StrategyOutputManager()
    
    # Save simulation results
    manager.save_simulation_results('momentum', 'NVDA', results_df)
    
    # Save best strategy outputs
    manager.save_signals('momentum', 'NVDA', signals_df, config)
    manager.save_portfolio_results('momentum', 'NVDA', portfolio_df, config)
    manager.save_benchmark_plot('momentum', 'NVDA', backtest_results)
"""

from .manager import StrategyOutputManager

__all__ = [
    'StrategyOutputManager',
]
