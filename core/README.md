# Core Module

Fundamental building blocks for portfolio simulation, risk management, and performance metrics.

## Overview

This module provides Numba-optimized functions for position tracking, transaction cost modeling, risk management, and performance analytics. All computationally intensive functions use `@njit` compilation for speed.

## Components

### 1) Portfolio Simulation (portfolio.py)

Core position management and portfolio value calculation.

**Functions:**

| Function | Description |
|----------|-------------|
| `simulate_positions_numba` | Position simulation with stop-loss/take-profit logic |
| `calculate_portfolio_values_numba` | Portfolio value tracking with transaction costs |
| `calculate_position_size_numba` | Position sizing based on rolling Sharpe |
| `calculate_kelly_position_size_numba` | Kelly Criterion position sizing |
| `calculate_kelly_continuous_numba` | Gaussian approximation Kelly sizing |

#

### 2) Risk Management (risk.py)

Position and portfolio-level risk monitoring.

**Functions:**

| Function | Description |
|----------|-------------|
| `check_stop_loss` | Check if stop-loss is triggered |
| `check_take_profit` | Check if take-profit is triggered |
| `check_max_drawdown` | Check if max drawdown limit is breached |
| `calculate_drawdown_series_numba` | Calculate drawdown series efficiently |
| `calculate_position_pnl` | Calculate position P&L |

**Classes:**

| Class | Description |
|-------|-------------|
| `RiskManager` | High-level risk management interface |
| `RiskLimits` | Container for risk limit parameters |
| `RiskState` | Container for current risk state |


#

### 3) Performance Metrics (`metrics.py`)

Risk-adjusted performance calculations.

**Functions:**

| Function | Description |
|----------|-------------|
| `calculate_var_numba` | Value at Risk (historical simulation) |
| `calculate_rolling_sharpe_numba` | Rolling Sharpe ratio |
| `calculate_sortino_components_numba` | Sortino ratio components |
| `calculate_max_drawdown_numba` | Maximum drawdown |

**Classes:**

| Class | Description |
|-------|-------------|
| `MetricsCalculator` | High-level metrics calculation interface |

#

### 4) Benchmark Comparison (`benchmarks.py`)

Strategy vs benchmark analytics.

**Class: `BenchmarkMetrics`**

| Metric | Description |
|--------|-------------|
| Beta | Sensitivity to benchmark movements |
| Alpha | Excess return beyond beta exposure (CAPM) |
| Correlation | Strategy-benchmark correlation |
| R-Squared | Variance explained by benchmark |
| Information Ratio | Risk-adjusted excess return |
| Treynor Ratio | Excess return per unit systematic risk |

---

## Usage

### 1) Position Simulation

```python
from core import simulate_positions_numba, calculate_portfolio_values_numba
import numpy as np

# Simulate positions with stop-loss/take-profit
positions, entry_prices, exit_reasons, stop_losses, take_profits = simulate_positions_numba(
    prices=close_prices,           # np.ndarray of prices
    signals=signals,               # np.ndarray: 1=buy, -1=sell, 0=hold
    trade_freq_mask=trade_mask,    # np.ndarray boolean
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    initial_position=0,
    initial_entry_price=0.0
)

# Calculate portfolio values with transaction costs
cash, shares, portfolio_values, commissions, slippages, buys, sells = calculate_portfolio_values_numba(
    prices=close_prices,
    positions=positions,
    entry_prices=entry_prices,
    initial_cash=100000.0,
    initial_shares=0,
    commission_rate=0.0035,
    slippage_bps=0.1,
    size_factor=1.0
)
```

#

### 2) Kelly Position Sizing

```python
from core.portfolio import calculate_kelly_position_size_numba

# Trade-based Kelly
position_size, raw_kelly = calculate_kelly_position_size_numba(
    returns=daily_returns,
    base_size=1000,
    lookback=60,
    kelly_fraction=0.5,    # Half-Kelly (conservative)
    max_leverage=1.0,
    min_trades=30
)
```

#

### 3) Risk Management

```python
from core import RiskManager, check_stop_loss, check_take_profit

# Using RiskManager class
risk_manager = RiskManager(
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    max_drawdown_pct=0.15,
    initial_aum=100000.0
)

# Check position risk
should_exit, reason = risk_manager.check_position_risk(
    current_price=98.0,
    entry_price=100.0
)

# Update portfolio state
risk_manager.update_portfolio_state(current_aum=95000.0)

# Check for max drawdown breach
if risk_manager.should_halt_trading():
    print("Max drawdown breached - halt trading")

# Or use low-level functions directly
if check_stop_loss(current_price=98.0, entry_price=100.0, stop_loss_pct=0.02):
    print("Stop loss triggered")
```

#

### 4) Performance Metrics

```python
from core import MetricsCalculator
import pandas as pd

calculator = MetricsCalculator(trading_days_per_year=252)

# Individual metrics
sharpe = calculator.calculate_sharpe_ratio(
    returns=strategy_returns,
    risk_free_rate_series=rf_series,
    annualize=True
)

sortino = calculator.calculate_sortino_ratio(returns, risk_free_rate_series=rf_series)
calmar = calculator.calculate_calmar_ratio(returns, aum_series=aum)
var_95 = calculator.calculate_var(returns, confidence_level=0.95)
volatility = calculator.calculate_volatility(returns, annualize=True)
win_rate = calculator.calculate_win_rate(returns)

# All metrics at once
metrics = calculator.calculate_all_metrics(
    returns=strategy_returns,
    aum_series=aum,
    risk_free_rate_series=rf_series
)
# Returns dict with: total_return, annualized_return, sharpe_ratio, sortino_ratio,
#                    calmar_ratio, max_drawdown, win_rate, var_95, var_99
```

#

### 5) Benchmark Comparison

```python
from core import BenchmarkMetrics

benchmark = BenchmarkMetrics(benchmark_ticker='SPY')

# Individual metrics
beta = benchmark.calculate_beta(strategy_returns, benchmark_returns)
alpha = benchmark.calculate_alpha(strategy_returns, benchmark_returns, risk_free_rate=0.03)
correlation = benchmark.calculate_correlation(strategy_returns, benchmark_returns)
info_ratio = benchmark.calculate_information_ratio(strategy_returns, benchmark_returns)

# All metrics at once
metrics = benchmark.calculate_all_metrics(
    returns=strategy_returns,
    benchmark_returns=benchmark_returns,
    risk_free_rate=0.03
)
# Returns dict with: Beta, Alpha (annualized), Correlation, R-Squared,
#                    Information Ratio, Treynor Ratio
```

---

## Exit Reason Codes

`simulate_positions_numba` returns an `exit_reasons` array:

| Code | Meaning |
|------|---------|
| 0 | No exit |
| 1 | Signal-based exit |
| 2 | Stop-loss triggered |
| 3 | Take-profit triggered |

---

## Performance Notes

All `*_numba` functions are JIT-compiled with `@njit(cache=True)` for optimal performance. 
- First call compiles the function
- Subsequent calls run at native speed.

Typical speedups vs pure Python: **10-100x** for position simulation and metric calculations.
