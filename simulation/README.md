# Simulation Module

Parallel parameter sweep backtesting for strategy optimization.

## Overview

This module runs backtests across multiple parameter combinations in parallel, collecting results for analysis and identifying optimal configurations.

**Workflow:**
1. Define parameter ranges in `SimulationConfig`
2. `TaskGenerator` creates all parameter combinations
3. `SimulationRunner` executes backtests in parallel
4. Results are collected, analyzed, and saved
5. Best configurations can be re-run with full output generation

## Components

### 1. SimulationConfig

Dataclass holding all simulation parameters.

**Parameter Groups:**
- `CommonParameters` - Shared across all strategies (trade frequency, risk limits, position sizing)
- `MomentumParameters` - Band multipliers
- `MeanReversionParameters` - Z-score lookbacks, std thresholds
- `MeanReversionRSIParameters` - RSI periods, oversold/overbought levels
- `StatArbParameters` - Entry/exit thresholds, hedge options
- `SupervisedParameters` - Model paths, probability thresholds

#

### 2. SimulationRunner

Main execution engine for parallel backtesting.

**Key Methods:**
- `run()` - Execute all parameter combinations
- `get_best_by_ticker()` - Find best config per ticker
- `run_best_strategy()` - Re-run best config with full outputs
- `run_all_best_strategies()` - Process all tickers

#

### 3. TaskGenerator

Creates task dictionaries for each parameter combination.

**Optimization:** For ML strategies, pre-computes model probabilities once and reuses across all threshold combinations (10-20x speedup).

## Supported Strategies

| Strategy | Config Class | Key Parameters |
|----------|--------------|----------------|
| `MOMENTUM` | `MomentumParameters` | `band_multipliers` |
| `MEAN_REVERSION` | `MeanReversionParameters` | `zscore_lookbacks`, `n_std_uppers/lowers` |
| `MEAN_REVERSION_RSI` | `MeanReversionRSIParameters` | `rsi_periods`, `oversold/overbought` |
| `STAT_ARB` | `StatArbParameters` | `entry/exit_thresholds`, `use_dynamic_hedge` |
| `SUPERVISED` | `SupervisedParameters` | `model_paths`, `buy/sell_thresholds` |

---

## Usage

### 1. Basic Parameter Sweep

```python
from simulation import SimulationRunner, SimulationConfig, StrategyType
from simulation.config import CommonParameters, MomentumParameters

config = SimulationConfig(
    strategy_type=StrategyType.MOMENTUM,
    tickers=['NVDA', 'TSLA'],
    common=CommonParameters(
        trade_frequencies=[15, 30, 60],
        stop_loss_pcts=[0.015, 0.02],
        take_profit_pcts=[0.03, 0.04],
    ),
    momentum=MomentumParameters(
        band_multipliers=[0.8, 1.0, 1.2, 1.5]
    )
)

runner = SimulationRunner(config)
results_df = runner.run(parallel=True)

# Find best configurations
best = runner.get_best_by_ticker(results_df, metric='Sharpe Ratio (Strategy)')
print(best[['Ticker', 'Sharpe Ratio (Strategy)', 'Total % Return (Strategy)']])
```

#

### 2. Run Best Strategy with Outputs

```python
# Run best config for a specific ticker
output = runner.run_best_strategy(
    results_df=results_df,
    ticker='NVDA',
    metric='Sharpe Ratio (Strategy)',
    save_outputs=True,
    show_plot=True
)

# Access results
print(f"Sharpe: {output['statistics'].sharpe_ratio:.2f}")
print(f"Return: {output['statistics'].total_return:.2%}")
print(f"Saved to: {output['paths']}")

# Or run for all tickers
all_outputs = runner.run_all_best_strategies(results_df, save_outputs=True)
```

#

### 3. ML Strategy Sweep

```python
from simulation.config import SupervisedParameters

config = SimulationConfig(
    strategy_type=StrategyType.SUPERVISED,
    tickers=['NVDA'],
    common=CommonParameters(
        trade_frequencies=[60],  # Must match model's forecast_horizon
        stop_loss_pcts=[0.015, 0.02],
        take_profit_pcts=[0.03, 0.04],
    ),
    supervised=SupervisedParameters(
        model_paths=['outputs/models/NVDA/model.pkl'],
        scaler_paths=['outputs/models/NVDA/scalers.pkl'],
        features_paths=['outputs/models/NVDA/features.pkl'],
        buy_thresholds=[0.50, 0.55, 0.60],
        sell_thresholds=[0.40, 0.45, 0.50],
    )
)

runner = SimulationRunner(config)
results = runner.run()
```

#

### 4. Pairs Trading (Stat Arb)

```python
config = SimulationConfig(
    strategy_type=StrategyType.STAT_ARB,
    pairs=[('NVDA', 'AMD'), ('AAPL', 'MSFT')],
    stat_arb=StatArbParameters(
        zscore_lookbacks=[30, 60],
        entry_thresholds=[1.5, 2.0],
        exit_thresholds=[0.0, 0.5],
        use_dynamic_hedge_options=[True, False]
    )
)

runner = SimulationRunner(config)
results = runner.run()

# Get best pair configurations
best_pairs = runner.get_best_by_pair(results)
```

#

### 5. Estimate Combinations

```python
# Before running, check how many backtests will execute
n_combinations = config.count_combinations()
print(f"Will run {n_combinations} backtests")
```

---

## Configuration Reference

### 1. CommonParameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trade_frequencies` | List[int] | [15, 30, 60] | Minutes between trades |
| `target_volatilities` | List[float] | [0.015, 0.02, 0.025] | Target daily vol |
| `stop_loss_pcts` | List[float] | [0.02] | Stop loss thresholds |
| `take_profit_pcts` | List[float] | [0.04] | Take profit thresholds |
| `max_drawdown_pcts` | List[float] | [0.15] | Max drawdown limits |
| `sizing_types` | List[str] | ['vol_target'] | Position sizing methods |
| `kelly_fractions` | List[float] | [0.5] | Kelly fractions |
| `kelly_lookbacks` | List[int] | [60] | Kelly lookback periods |
| `commission` | float | 0.0035 | Commission per share |
| `slippage_factor` | float | 0.1 | Slippage in bps |
| `initial_aum` | float | 100000.0 | Starting capital |

#

### 2. Strategy-Specific Parameters

**MomentumParameters:**
- `band_multipliers` - Bollinger band multipliers

**MeanReversionParameters:**
- `zscore_lookbacks` - Z-score calculation windows
- `n_std_uppers/lowers` - Entry threshold std deviations
- `exit_thresholds` - Exit z-score thresholds

**MeanReversionRSIParameters:**
- `rsi_periods` - RSI calculation periods
- `rsi_oversold_levels` - Buy signal thresholds
- `rsi_overbought_levels` - Sell signal thresholds
- `sma_periods` - Trend filter periods

**StatArbParameters:**
- `zscore_lookbacks` - Spread z-score windows
- `entry_thresholds` - Entry z-score thresholds
- `exit_thresholds` - Exit z-score thresholds
- `use_dynamic_hedge_options` - Dynamic vs fixed hedge ratio

**SupervisedParameters:**
- `model_paths` - Paths to trained model files
- `scaler_paths` - Paths to normalizer files
- `features_paths` - Paths to feature name files
- `buy_thresholds` - Probability thresholds for buy signals
- `sell_thresholds` - Probability thresholds for sell signals

---

## Results DataFrame

The `run()` method returns a DataFrame with these columns:

**Identifiers:**
- `Ticker`, `Strategy Type`

**Configuration:**
- `Trading Frequency`, `Target Volatility`, `Stop Loss`, `Take Profit`
- Strategy-specific parameters (e.g., `Band Multiplier`, `Z-Score Lookback`)

**Performance Metrics:**
- `Total % Return (Strategy)`, `Total % Return (Buy & Hold)`
- `Sharpe Ratio (Strategy)`, `Sharpe Ratio (Buy & Hold)`
- `Sortino Ratio`, `Calmar Ratio`
- `Max Drawdown (%)`, `Win Rate (%)`

**Trade Statistics:**
- `Total Trades`, `Buy Trades`, `Sell Trades`
- `Stop Losses Hit`, `Take Profits Hit`

**Benchmark Comparison:**
- `Beta`, `Alpha (annualized %)`, `Correlation`
- `Information Ratio`, `R-Squared`

---

## Output Files

When `save_outputs=True`, results are saved to:

```
outputs/strategies/{strategy}/
├── simulations/{ticker}/    # Parameter sweep CSV
├── signals/{ticker}/        # Best strategy signals + config JSON
├── results/{ticker}/        # Portfolio results + metadata JSON
└── plots/{ticker}/          # Benchmark comparison PNG
```

---

## Performance Tips

1. **Parallel Execution**: Set `n_workers` in config (default: CPU count - 2)
   ```python
   config = SimulationConfig(..., n_workers=8)
   ```

2. **ML Optimization**: Probabilities are pre-computed once per model, reducing runtime significantly for threshold sweeps

3. **Reduce Combinations**: Start with fewer parameter values to validate, then expand
   ```python
   # Quick test
   config = SimulationConfig(
       common=CommonParameters(trade_frequencies=[30])  # Single value
   )
   ```

4. **Monitor Progress**: Logger shows tasks/second and ETA during execution

---

## Command Line Usage

```bash
python -m simulation.runner \
    --strategy momentum \
    --tickers NVDA \ 
    --workers 6
```
