# Strategies Module

Trading strategy implementations with a consistent interface for backtesting and simulation.

## Overview

This module provides five strategy types, all following a common interface defined by `BaseStrategy`. Strategies generate trading signals (1=buy, -1=sell/exit, 0=hold) from price data.

## Architecture

### 1. BaseStrategy (base.py)

Abstract base class that enforces a consistent interface across all strategies.

**Benefits:**
- **Uniform API**: All strategies implement `generate_signals()`, `get_parameter_grid()`, and `from_config()`
- **Backtesting compatibility**: Consistent signal format (1/-1/0) works with BacktestEngine
- **Parameter optimization**: `get_parameter_grid()` enables automated sweeps via SimulationRunner
- **Factory pattern support**: `from_config()` enables dynamic instantiation
- **Validation helpers**: Built-in data validation and minimum bar requirements

**Class Hierarchy:**
```
BaseStrategy (abstract)
├── Momentum
├── MeanReversion
├── MeanReversionRSI
├── SupervisedStrategy
└── PairsStrategy (abstract)
    └── StatArb
```

#

### 2. StrategyFactory (factory.py)

Factory class for creating and managing strategies.

**Benefits:**
- **Centralized configuration**: `get_default_config()` provides sensible defaults
- **Dynamic instantiation**: Create strategies from config dicts at runtime
- **Validation**: `validate_config()` catches missing parameters early
- **Extensibility**: `register_strategy()` allows adding custom strategies
- **Decoupling**: Simulation code doesn't need to import specific strategy classes

```python
from strats import StrategyFactory, StrategyType

# Get default configuration
config = StrategyFactory.get_default_config(StrategyType.MOMENTUM)

# Get signal function
signal_func = StrategyFactory.get_strategy(StrategyType.MOMENTUM)

# Validate configuration
StrategyFactory.validate_config(config, StrategyType.MOMENTUM)
```

---


## Strategies

### 1. Momentum (Band Breakout)

Trend-following strategy that enters when price breaks above volatility bands and exits on band violation or VWAP cross.

**Signal Logic:**
- **Buy**: Price > Upper Band AND Price > VWAP
- **Exit**: Price < Lower Band OR Price < VWAP

**Parameters:**

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `band_mult` | float | 1.0 | Band width multiplier. Higher = fewer signals, requires stronger breakouts. Lower = more signals, more noise. |

**Band Calculation:**
```
Upper Band = reference_high × (1 + band_mult × sigma_open)
Lower Band = reference_low × (1 - band_mult × sigma_open)
```

**Trading Impact:**
- `band_mult < 1.0`: More frequent entries, higher turnover, captures smaller moves
- `band_mult > 1.0`: Fewer entries, lower turnover, only enters on strong momentum
- Optimal value depends on volatility regime and transaction costs

```python
from strats import Momentum

signals = Momentum.generate_signals(
    close_prices=prices,
    vwap=vwap,
    sigma_open=sigma,
    reference_high=prev_high,
    reference_low=prev_low,
    band_mult=1.0
)
```

#

### 2. Mean Reversion (Z-Score)

Counter-trend strategy that buys when price is statistically cheap (below mean) and sells when expensive (above mean).

**Signal Logic:**
- **Buy**: Z-score < -n_std_lower (price is n standard deviations below mean)
- **Exit**: Z-score > n_std_upper OR Z-score returns to exit_threshold

**Parameters:**

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `lookback_period` | int | 20 | Bars for mean/std calculation. Longer = smoother mean, slower reaction. Shorter = faster reaction, more noise. |
| `n_std_lower` | float | 2.0 | Entry threshold. Higher = fewer entries, only extreme deviations. Lower = more entries, smaller deviations. |
| `n_std_upper` | float | 2.0 | Exit threshold. Higher = holds longer, waits for mean reversion. Lower = exits earlier. |
| `exit_threshold` | float | 0.0 | Z-score level to close position. 0.0 = exit at mean. Positive = exit before mean. |

**Trading Impact:**
- `lookback_period`: Short (10-20) for high-frequency mean reversion, long (50-100) for swing trading
- `n_std` thresholds: 1.5-2.0 for active trading, 2.5-3.0 for conservative/rare entries
- `exit_threshold`: 0.0 captures full reversion, 0.5 exits earlier (reduces risk but also profit)

```python
from strats import MeanReversion

signals = MeanReversion.generate_signals(
    close_prices=prices,
    lookback_period=20,
    n_std_upper=2.0,
    n_std_lower=2.0,
    exit_threshold=0.0
)
```
#

### 3. Mean Reversion RSI

RSI-based mean reversion with SMA trend filter. Buys oversold conditions in uptrends only.

**Signal Logic:**
- **Buy**: RSI < oversold AND Price > SMA (buy dips in uptrend)
- **Exit**: RSI > overbought OR Price < SMA (trend reversal)

**Parameters:**

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `rsi_period` | int | 10 | RSI calculation period. Shorter = more sensitive/volatile. Longer = smoother/slower. |
| `rsi_oversold` | float | 30.0 | RSI level for buy signal. Lower = fewer entries, only extreme oversold. Higher = more entries. |
| `rsi_overbought` | float | 70.0 | RSI level for exit. Lower = exits earlier. Higher = holds longer. |
| `sma_period` | int | 200 | Trend filter period. Longer = stronger trend confirmation. Shorter = more responsive. |
| `exit_on_sma_cross` | bool | True | Exit when price crosses below SMA. False = only exit on RSI overbought. |

**Trading Impact:**
- `rsi_period`: 5-10 for short-term trading, 14-21 for swing trading
- `rsi_oversold/overbought`: Wider range (25/75) = fewer signals, tighter (35/65) = more signals
- `sma_period`: 50-100 for medium-term trends, 200 for long-term trend filter
- SMA filter prevents buying falling knives in downtrends

```python
from strats import MeanReversionRSI

signals = MeanReversionRSI.generate_signals(
    close_prices=prices,
    rsi_period=10,
    rsi_oversold=30.0,
    rsi_overbought=70.0,
    sma_period=200,
    exit_on_sma_cross=True
)
```

#

### 4. Statistical Arbitrage (Pairs Trading)

Trades the spread between two correlated assets, betting on mean reversion of their price relationship.

**Signal Logic:**
- **Buy Spread**: Spread Z-score < -entry_threshold (spread is cheap)
- **Exit**: Spread Z-score > entry_threshold OR returns to exit_threshold

**Parameters:**

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `lookback_period` | int | 60 | Bars for spread mean/std. Longer = more stable relationship estimate. |
| `entry_threshold` | float | 2.0 | Z-score for entry. Higher = fewer trades, larger dislocations only. |
| `exit_threshold` | float | 0.0 | Z-score for exit. 0.0 = exit at mean. Positive = exit earlier. |
| `hedge_ratio` | float | None | Fixed ratio. None = calculate dynamically. |
| `use_dynamic_hedge` | bool | True | Recalculate hedge ratio rolling. False = use initial ratio. |

**Trading Impact:**
- `lookback_period`: 30-60 for shorter relationships, 90-120 for stable pairs
- `entry_threshold`: 1.5-2.0 for active trading, 2.5-3.0 for conservative
- `use_dynamic_hedge`: True adapts to changing correlations, False for stable pairs

```python
from strats import StatArb

signals, zscore, spread, hedge_ratios = StatArb.generate_signals(
    prices_a=nvda_prices,
    prices_b=amd_prices,
    lookback_period=60,
    entry_threshold=2.0,
    exit_threshold=0.0,
    use_dynamic_hedge=True
)
```

#

### 5. Supervised ML Strategy

Machine learning-based strategy using trained classifiers to predict price direction.

**Signal Logic:**
- **Buy**: Model probability ≥ buy_threshold AND not in position
- **Exit**: Model probability ≤ sell_threshold AND in position

**Parameters:**

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `model_path` | str | required | Path to trained model (.pkl) |
| `scaler_path` | str | required | Path to fitted normalizer |
| `features_path` | str | required | Path to feature names list |
| `buy_threshold` | float | 0.55 | Probability threshold for buy. Higher = fewer entries, higher confidence. |
| `sell_threshold` | float | 0.45 | Probability threshold for exit. Lower = exits earlier, less confident. |

**Trading Impact:**
- `buy_threshold`: 0.50-0.55 for active trading, 0.60-0.70 for high-confidence only
- `sell_threshold`: 0.40-0.50 for quick exits, 0.30-0.40 for holding longer
- Thresholds at 0.50/0.50 = maximum responsiveness to model signals
- Wider gap (0.60/0.40) = buffer zone, fewer trades, avoids whipsaws

```python
from strats import SupervisedStrategy

strategy = SupervisedStrategy.from_artifacts(
    model_path='outputs/models/NVDA/model.pkl',
    scaler_path='outputs/models/NVDA/scalers.pkl',
    features_path='outputs/models/NVDA/features.pkl',
    buy_threshold=0.55,
    sell_threshold=0.45
)

signals = strategy.generate_signals(df_ohlcv)

# Or use pre-computed probabilities (faster for parameter sweeps)
probabilities = strategy.get_probabilities(df_ohlcv)
signals = strategy.generate_signals_from_probabilities(probabilities, buy_threshold=0.60)
```

---

## Signal Values

All strategies use consistent signal values:

| Signal | Value | Meaning |
|--------|-------|---------|
| `SIGNAL_LONG` | 1 | Enter long position |
| `SIGNAL_EXIT` | -1 | Exit position |
| `SIGNAL_HOLD` | 0 | No action |

---

## Usage Examples

### 1. Direct Signal Generation

```python
from strats import Momentum, MeanReversion

# Momentum signals
momentum_signals = Momentum.generate_signals(
    close_prices, vwap, sigma_open, ref_high, ref_low, band_mult=1.0
)

# Mean reversion signals
mr_signals = MeanReversion.generate_signals(
    close_prices, lookback_period=20, n_std_upper=2.0, n_std_lower=2.0
)
```

#

### 2. Using StrategyFactory

```python
from strats import StrategyFactory, StrategyType

# Get default config and customize
config = StrategyFactory.get_default_config(StrategyType.MEAN_REVERSION)
config['lookback_period'] = 30
config['n_std_upper'] = 2.5

# Validate before use
StrategyFactory.validate_config(config, StrategyType.MEAN_REVERSION)

# Get parameter grid for sweeps
grid = StrategyFactory.get_strategy_parameter_grid(StrategyType.MEAN_REVERSION)
```

#

### 3. With BacktestEngine

```python
from backtesting import BacktestEngine

config = {
    'strategy_type': 'momentum',
    'band_mult': 1.0,
    'AUM': 100000,
    'trade_freq': 30,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
}

engine = BacktestEngine(ticker='NVDA')
results = engine.run(df, df_daily, all_days, config)
```

---

## Performance Considerations

1. **Numba JIT compilation**: RSI, SMA, Z-score calculations use `@njit(cache=True)` for 10-100x speedup
2. **Static methods**: Signal generation is stateless, enabling vectorized operations
3. **Pre-computed probabilities**: ML strategy can compute probabilities once and generate signals for multiple threshold combinations
4. **Minimal allocations**: Strategies return numpy arrays directly without intermediate DataFrame conversions
