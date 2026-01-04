# Backtesting Module

Core backtesting infrastructure for strategy evaluation and performance analysis.

## Overview

The backtesting module provides a complete framework for running historical simulations of trading strategies. It handles daily data iteration, signal generation across multiple strategy types, position simulation with risk management, and detailed performance analytics.

## Components

### BacktestEngine (engine.py)

The main execution engine that orchestrates the backtest loop.

**Key Responsibilities:**
- Data preprocessing and daily iteration over trading days
- Signal generation coordination for all strategy types (momentum, mean reversion, stat arb, ML/supervised)
- Calling the position simulation with stop-loss and take-profit logic from the `core` module.
- Applying portfolio value tracking with transaction costs (commissions, slippage)
- Applying risk management (max drawdown breaches, position sizing)
- Market calendar handling (holidays, early closes)

**Supported Strategy Types:**
- `MOMENTUM` - VWAP/volatility band breakout signals
- `MEAN_REVERSION` - Z-score based mean reversion
- `MEAN_REVERSION_RSI` - RSI + SMA mean reversion
- `STAT_ARB` - Pairs trading with dynamic hedge ratios
- `SUPERVISED` - ML model-based signals (with precomputed probability optimization)

**Position Sizing Methods:**
- `vol_target` - Volatility targeting (default)
- `kelly` - Kelly Criterion based sizing
- `kelly_vol_blend` - Blended Kelly + volatility targeting

### BacktestResults

Container for backtest output data with built-in analysis methods.

**Key Features:**
- Daily strategy metrics DataFrame (AUM, returns, positions, drawdowns)
- Automatic benchmark (buy & hold) tracking
- Performance statistics calculation
- Visualization utilities (equity curves, drawdown charts)
- CSV export functionality

**Properties:**
- `returns` - Strategy daily returns series
- `benchmark_returns` - Benchmark daily returns series  
- `aum` - Strategy AUM series
- `benchmark_aum` - Benchmark AUM series

### BacktestStatistics

Dataclass containing all calculated performance metrics.

**Return Metrics:**
- Total return, annualized return
- Benchmark return, benchmark annualized return

**Risk-Adjusted Metrics:**
- Sharpe ratio, Sortino ratio, Calmar ratio
- Benchmark equivalents for comparison

**Risk Metrics:**
- Max drawdown (strategy and benchmark)
- Annualized volatility

**Trade Metrics:**
- Win rate, total trades
- Buy/sell trade counts
- Stop losses hit, take profits hit

**Benchmark Comparison:**
- Alpha (annualized), Beta
- Correlation, R-squared
- Information ratio, Treynor ratio

## Configuration

The backtest is configured via a dictionary with the following parameters:

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_type` | str | Strategy type (see supported types above) |
| `AUM` | float | Initial portfolio value |
| `trade_freq` | int | Minutes between trade evaluations |

### Risk Management Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stop_loss_pct` | float | 0.02 | Stop loss threshold (2%) |
| `take_profit_pct` | float | 0.04 | Take profit threshold (4%) |
| `max_drawdown_pct` | float | 0.15 | Max drawdown before forced liquidation |

### Position Sizing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sizing_type` | str | 'vol_target' | Sizing method |
| `target_vol` | float | 0.02 | Target daily volatility |
| `max_leverage` | float | 1.0 | Maximum leverage allowed |
| `kelly_fraction` | float | 0.5 | Fraction of Kelly to use |
| `kelly_lookback` | int | 60 | Lookback period for Kelly calc |

### Transaction Cost Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `commission` | float | 0.0035 | Commission per share |
| `slippage_factor` | float | 0.1 | Slippage in basis points |

### Strategy-Specific Parameters

**Momentum:**
- `band_mult` - Bollinger band multiplier

**Mean Reversion:**
- `zscore_lookback` - Lookback period for z-score
- `n_std_upper` / `n_std_lower` - Entry thresholds

**Supervised (ML):**
- `model_path`, `scaler_path`, `features_path` - Model artifacts
- `buy_threshold`, `sell_threshold` - Probability thresholds
- `precomputed_probabilities` - Pre-computed predictions (for speed)

## Usage

### Basic Example

```python
from backtesting import BacktestEngine
from backtesting.engine import StrategyType

# Initialize engine
engine = BacktestEngine(ticker='TSLA')

# Configure backtest
config = {
    'strategy_type': StrategyType.MOMENTUM,
    'AUM': 100000.0,
    'trade_freq': 30,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
    'max_drawdown_pct': 0.15,
    'target_vol': 0.02,
    'band_mult': 1.0,
    'sizing_type': 'kelly'
}

# Run backtest
results = engine.run(
    df=df_intraday,
    ticker_daily_data=df_daily,
    all_days=trading_days,
    config=config,
    market_calendar=calendar,
    risk_free_rate_series=rf_rates,
    verbose=True
)

# Analyze results
stats = results.calculate_statistics()

print(f"Total Return: {stats.total_return:.2%}")
print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")
print(f"Max Drawdown: {stats.max_drawdown:.2%}")
print(f"Total Trades: {stats.total_trades}")

# Visualize
results.plot(show=True, save_path='backtest_results.png')
results.plot_drawdown(save_path='drawdown.png')

# Export
results.to_csv('strategy_results.csv')
```

### ML Strategy Example

```python
from backtesting import BacktestEngine
from backtesting.engine import StrategyType

engine = BacktestEngine(ticker='TSLA')

config = {
    'strategy_type': StrategyType.SUPERVISED,
    'AUM': 100000.0,
    'trade_freq': 60,  # Must match model's forecast horizon
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
    
    # ML model artifacts
    'model_path': 'outputs/models/TSLA/model.pkl',
    'scaler_path': 'outputs/models/TSLA/scalers.pkl',
    'features_path': 'outputs/models/TSLA/features.pkl',
    
    # Signal thresholds
    'buy_threshold': 0.55,
    'sell_threshold': 0.45,
}

results = engine.run(
    df=df_resampled,  # Use resampled data matching training
    ticker_daily_data=df_daily,
    all_days=trading_days,
    config=config
)

# Access benchmark comparison metrics
stats = results.calculate_statistics()
print(f"Alpha: {stats.alpha:.2%}")
print(f"Beta: {stats.beta:.2f}")
print(f"Information Ratio: {stats.information_ratio:.2f}")
```

### Pairs Trading Example

```python
from backtesting import BacktestEngine
from backtesting.engine import StrategyType

engine = BacktestEngine(ticker='AAPL', ticker_b='MSFT')

config = {
    'strategy_type': StrategyType.STAT_ARB,
    'AUM': 100000.0,
    'trade_freq': 30,
    'stop_loss_pct': 0.03,
    'take_profit_pct': 0.05,
    'zscore_lookback': 60,
    'entry_threshold': 2.0,
    'exit_threshold': 0.0,
    'use_dynamic_hedge': True,
}

results = engine.run(
    df=df_aapl,
    ticker_daily_data=df_aapl_daily,
    all_days=trading_days,
    config=config,
    df_b=df_msft,
    ticker_b_daily_data=df_msft_daily
)
```

### Accessing Raw Data

```python
# Get the full strategy DataFrame
strategy_df = results.strategy_df

# Available columns:
# - ret: Daily strategy return
# - AUM: End of day portfolio value
# - balance: Cash balance
# - ret_ticker: Benchmark daily return
# - position: End of day position (0 or 1)
# - drawdown: Current drawdown from peak
# - shares_held: Number of shares held
# - slippage_cost: Daily slippage costs
# - size_factor: Position sizing factor used
# - kelly_raw: Raw Kelly fraction (if using Kelly sizing)

# For stat arb strategies:
# - spread_zscore: End of day spread z-score
# - hedge_ratio: Dynamic hedge ratio
```

## Integration with SimulationRunner

For parameter sweeps, use the `SimulationRunner` which wraps `BacktestEngine`:

```python
from simulation import SimulationRunner, SimulationConfig, StrategyType
from simulation.config import CommonParameters, MomentumParameters

config = SimulationConfig(
    strategy_type=StrategyType.MOMENTUM,
    tickers=['TSLA'],
    common=CommonParameters(
        trade_frequencies=[15, 30, 60],
        stop_loss_pcts=[0.015, 0.02, 0.025],
        take_profit_pcts=[0.03, 0.04, 0.05],
    ),
    momentum=MomentumParameters(
        band_multipliers=[0.8, 1.0, 1.2],
    )
)

runner = SimulationRunner(config)
results_df = runner.run(parallel=True)

# Find best configuration
best = runner.get_best_by_ticker(results_df, metric='Sharpe Ratio (Strategy)')
```

## Performance Optimizations

The engine includes several optimizations for speed:

1. **Precomputed Daily Data** - OHLCV data is grouped and cached by day before the main loop
2. **Numba JIT Compilation** - Position simulation and portfolio calculations use `@njit` compiled functions
3. **Precomputed ML Probabilities** - For supervised strategies, probabilities can be pre-computed once and reused across parameter sweeps (10-20x speedup)
4. **Day-to-Index Mapping** - Fast lookup of DataFrame indices by trading day

## Output Files

When using `results.save_outputs()` or the `SimulationRunner`, outputs are organized as:

```
outputs/strategies/{strategy_type}/
├── simulations/{ticker}/    # Parameter sweep results
├── signals/{ticker}/        # Daily signals CSV + config JSON
├── results/{ticker}/        # Portfolio results CSV + metadata JSON
└── plots/{ticker}/          # Equity curve PNG
```