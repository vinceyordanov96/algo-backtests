# Utils Module

Utility class for formatted verbose output generation during backtests and simulations.

## Overview

The `Outputs` class provides standardized formatting for simulation results and logging.

## Components

### Outputs

Generates formatted output messages for simulation runs, including parameter logging, result formatting, and summary generation.

**Design Principle:** Configuration eliminates if-else blocks for each strategy type. Adding a new strategy only requires adding entries to the configuration dictionaries.

## Configuration Dictionaries

### 1. strategy_parameters

Maps strategy types to their configurable parameters for logging at simulation start.

```python
{
    StrategyType.MOMENTUM: [
        ("Tickers", "tickers"),
        ("Band Multipliers", "band_multipliers"),
        ...
    ],
    ...
}
```

#

### 2. strategy_result_columns

Maps strategy types to their result DataFrame columns for output formatting.

```python
{
    StrategyType.MOMENTUM: [
        ("Band Multiplier", "Band Multiplier", None),  # (display_name, column_name, format)
    ],
    ...
}
```

#

### 3. Common Column Groups

| Group | Purpose |
|-------|---------|
| `common_result_columns` | Trade freq, volatility, risk limits |
| `trading_activity_columns` | Trade counts, stops/profits hit |
| `performance_columns` | Returns, ratios, risk metrics |
| `benchmark_columns` | Buy & hold comparison |

---

## Usage

### 1. Log Simulation Parameters

```python
from utils import Outputs
from strats import StrategyType
import logging

logger = logging.getLogger(__name__)
outputs = Outputs()

# Log parameters at simulation start
outputs.log_simulation_parameters(
    strategy_type=StrategyType.MOMENTUM,
    simulation_instance=simulation,
    logger=logger
)
```

#

### 2. Format Best Results

```python
# Format single result row
formatted = outputs.format_best_strategy_result(
    strategy_type=StrategyType.MOMENTUM,
    row=best_results.iloc[0]
)
print(formatted)

# Format all best results
outputs.print_best_results(
    strategy_type=StrategyType.MOMENTUM,
    best_df=best_results
)
```

**Output Example:**
```
==================================================
BEST MOMENTUM STRATEGY CONFIGURATION (by Sharpe Ratio)
==================================================

--------------------------------------------------
NVDA
--------------------------------------------------

Strategy Parameters:
  • Band Multiplier: 1.0
  • Trading Frequency: 30 min
  • Target Volatility: 2.00%
  • Stop Loss: 2.00%
  • Take Profit: 4.00%

Trading Activity:
  • Total Trades: 156
  • Buy Trades: 78
  • Sell Trades: 78
  • Stop Losses Hit: 12
  • Take Profits Hit: 23

Strategy Performance:
  • Total Return: 45.23%
  • Sharpe Ratio: 1.8234
  • Sortino Ratio: 2.4512
  • Max Drawdown: 8.45%
  • Win Rate: 58.97%

Benchmark (Buy & Hold):
  • Total Return: 32.15%
  • Sharpe Ratio: 1.2341
  • Alpha: +13.08%

==================================================
```

#

### 3. Format Simulation Summary

```python
# Print summary for all tickers
outputs.print_simulation_summary(
    strategy_type=StrategyType.MOMENTUM,
    results_df=all_results,
    tickers=['NVDA', 'TSLA', 'AAPL']
)
```

**Output Example:**
```
====================================
MOMENTUM Simulation Summary
====================================

NVDA:
  Configurations tested: 324
  Sharpe Range: -0.45 to 2.12
  Return Range: -15.23% to 67.89%

TSLA:
  Configurations tested: 324
  Sharpe Range: -1.23 to 1.87
  Return Range: -32.45% to 89.12%
```

#

### 4. Format Individual Sections

```python
# Format specific sections separately
params_str = outputs.format_strategy_parameters(StrategyType.MOMENTUM, row)
activity_str = outputs.format_trading_activity(row)
perf_str = outputs.format_performance(row)
bench_str = outputs.format_benchmark(row)
```

---

## Format Specifications

The `_format_value()` method handles various format types:

| Format | Example | Output |
|--------|---------|--------|
| `None` | `123` | `123` |
| `".2f"` | `1.234` | `1.23` |
| `".4f"` | `1.23456` | `1.2346` |
| `".2%"` | `0.1234` | `12.34%` |
| `"min"` | `30` | `30 min` |
| N/A values | `NaN` | `N/A` |

---

## Methods Reference

| Method | Purpose |
|--------|---------|
| `log_simulation_parameters()` | Log params via logger at simulation start |
| `generate_parameters_string()` | Return params as formatted string |
| `format_strategy_parameters()` | Format strategy-specific params from result row |
| `format_trading_activity()` | Format trade counts and stop/profit hits |
| `format_performance()` | Format all performance metrics |
| `format_benchmark()` | Format buy & hold comparison with alpha |
| `format_ticker_header()` | Format ticker (handles pairs for stat arb) |
| `format_best_strategy_result()` | Complete formatted result for one ticker |
| `format_all_best_results()` | Format all best results for printing |
| `print_best_results()` | Print all best results |
| `format_ticker_summary()` | Summary stats for single ticker |
| `format_pair_summary()` | Summary stats for trading pair |
| `format_simulation_summary()` | Complete simulation summary |
| `print_simulation_summary()` | Print simulation summary |


---

## Extending for New Strategies

To add a new strategy type:

1. Add entry to `strategy_parameters`:
```python
StrategyType.NEW_STRATEGY: [
    ("Parameter Name", "attribute_name"),
    ...
]
```

2. Add entry to `strategy_result_columns`:
```python
StrategyType.NEW_STRATEGY: [
    ("Display Name", "Column Name", ".2f"),  # or None for default
    ...
]
```
