# Outputs Module

Unified output management for strategy artifacts, data, and model persistence.

## Overview

This module provides `StrategyOutputManager` for saving and loading all strategy-related outputs with consistent naming conventions and directory organization.

## Directory Structure

```
outputs/
├── data/                           # Market data (via DataManager)
│   ├── intraday/{ticker}/
│   ├── daily/{ticker}/
│   └── dividends/{ticker}/
├── models/                         # ML models (via ArtifactManager)
│   └── {ticker}/
└── strategies/                     # Strategy outputs (via StrategyOutputManager)
    └── {strategy}/
        ├── simulations/{ticker}/   # Parameter sweep results
        ├── signals/{ticker}/       # Best strategy signals
        ├── results/{ticker}/       # Trade-by-trade portfolio
        └── plots/{ticker}/         # Benchmark comparison plots
```

---

## Output Types

| Type | Description | Format |
|------|-------------|--------|
| Simulations | Parameter sweep results (all combinations) | CSV |
| Signals | Generated signals from best strategy | CSV + JSON config |
| Results | Trade-by-trade portfolio updates | CSV + JSON metadata |
| Plots | Strategy vs benchmark visualization | PNG |

---

## Usage

### 1. Save Simulation Results

```python
from outputs import StrategyOutputManager

manager = StrategyOutputManager()

# After parameter sweep
manager.save_simulation_results('momentum', 'NVDA', results_df)
```

#

### 2. Save Best Strategy Outputs

```python
# After running best configuration
manager.save_signals('momentum', 'NVDA', signals_df, config)
manager.save_portfolio_results('momentum', 'NVDA', portfolio_df, config, stats)
manager.save_benchmark_plot('momentum', 'NVDA', backtest_results)

# Or save all at once
paths = manager.save_all_best_strategy_outputs(
    strategy='momentum',
    ticker='NVDA',
    backtest_results=results,
    signals_df=signals_df,
    portfolio_df=portfolio_df,
    config=config
)
```

#

### 3. Load Outputs

```python
# Load latest simulation results
sim_df = manager.load_simulation_results('momentum', 'NVDA')

# Load latest signals
signals_df = manager.load_signals('momentum', 'NVDA')

# Load portfolio results
portfolio_df = manager.load_portfolio_results('momentum', 'NVDA')

# Get best config from simulation
best_config = manager.get_best_config_from_simulation(
    'momentum', 'NVDA', 
    metric='Sharpe Ratio (Strategy)'
)
```

#

### 4. List and Query

```python
# List all strategies with outputs
strategies = manager.list_strategies()

# List tickers for a strategy
tickers = manager.list_tickers('momentum')

# List all versions for a ticker
versions = manager.list_versions('momentum', 'NVDA', 'simulations')

# Get info about latest output
info = manager.get_output_info('momentum', 'NVDA', 'simulations')
```

#

### 5. Cleanup

```python
# Keep only 3 most recent versions
deleted = manager.cleanup_old_versions('momentum', 'NVDA', 'simulations', keep_n=3)
```

---

## File Naming Convention

All files follow the pattern:
```
{TICKER}_{strategy}_{output_type}_{YYYYMMDD}_{HHMMSS}.{ext}
```

Example: `NVDA_momentum_simulation_20240115_143022.csv`

---

## Integration

Works alongside:
- `connectors.DataManager` - Market data persistence
- `ml.ArtifactManager` - ML model artifacts
- `backtesting.BacktestResults` - Backtest output container
