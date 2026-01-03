"""
Core module for portfolio simulation, risk management, and performance metrics.

This module provides the fundamental building blocks for backtesting:
    - Portfolio simulation with Numba-optimized position tracking
    - Risk management (stop-loss, take-profit, drawdown)
    - Performance metrics (Sharpe, Sortino, Calmar, VaR)

Example:
    from core import simulate_positions, calculate_portfolio_values
    from core.metrics import MetricsCalculator
    from core.risk import RiskManager
"""

from .portfolio import (
    simulate_positions_numba,
    calculate_portfolio_values_numba,
)

from .risk import (
    RiskManager,
    check_stop_loss,
    check_take_profit,
    check_max_drawdown,
    calculate_drawdown_series_numba,
)

from .metrics import (
    MetricsCalculator,
    calculate_var_numba,
    calculate_rolling_sharpe_numba,
    calculate_sortino_components_numba,
)

from .benchmarks import (
    BenchmarkMetrics,
)

__all__ = [
    # Portfolio simulation
    'simulate_positions_numba',
    'calculate_portfolio_values_numba',
    
    # Risk management
    'RiskManager',
    'check_stop_loss',
    'check_take_profit',
    'check_max_drawdown',
    'calculate_drawdown_series_numba',
    
    # Metrics
    'MetricsCalculator',
    'calculate_var_numba',
    'calculate_rolling_sharpe_numba',
    'calculate_sortino_components_numba',

    # Benchmarks
    'BenchmarkMetrics',
]
