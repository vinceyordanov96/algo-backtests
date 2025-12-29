# Trading Strategy Backtesting Framework

Stemming from a personal project, the code in this repository aims to demonstrate how we can implement a scalable and reusable backtesting framework which simulates realistic historical trading, provided an algorithmic trading strategy (i.e., Momentum, Mean Reversion, RL etc.). Running a simulation amounts to selecting one of the available strategies and testing its historical performance on a given asset. Upon finishing a simulation, the user gets several outputs: 

1) The best performing strategy (according to Sharpe/Sortino/Calmar ratio) along with its parameters.
2) A granular overview detailing the `BUY`, `HOLD`, and `SELL` events of the best performing strategy. 
3) An overview containing strategy performance metrics & benchmarks.

## Overview

This framework aims to enable rigorous backtesting of intraday trading strategies with realistic execution modeling, risk management, and detailed performance analytics. It supports parallel execution for rapid parameter optimization and includes safeguards against common backtesting pitfalls.

### Key Considerations

- **Vectorized Operations**: NumPy-based signal calculations.
- **JIT Compilation**: Numba-accelerated position simulation and portfolio calculations.
- **Parallel Execution**: Multi-process parameter sweep using `ProcessPoolExecutor`.
- **Realistic Execution**: Next-bar execution, slippage modeling, and commission costs.
- **Risk Management**: Stop-loss, take-profit, max drawdown circuit breakers.
- **Performance Metrics**: Sharpe, Sortino, Calmar, Beta, Alpha, Information Ratio, etc.

---

## Supported Strategies

### 1. Momentum (Band Breakout)

Enters long positions when price breaks above a volatility-adjusted band and exits when price falls below.

| Parameter | Description |
|-----------|-------------|
| `band_mult` | Multiplier for volatility bands (higher = fewer trades, stronger signals) |
| `trade_freq` | Minutes between signal evaluations |
| `target_vol` | Target portfolio volatility for position sizing |

**Signal Logic**:
- **Entry**: Price > Upper Band AND Price > VWAP
- **Exit**: Price < Lower Band OR Price < VWAP

---

### 2. Mean Reversion (Z-Score)

Buys when price deviates significantly below its rolling mean and exits on reversion.

| Parameter | Description |
|-----------|-------------|
| `zscore_lookback` | Lookback period for mean and standard deviation |
| `n_std_upper` | Std devs above mean for exit signal |
| `n_std_lower` | Std devs below mean for entry signal |
| `exit_threshold` | Z-score level to exit position (typically 0) |

**Signal Logic**:
- **Entry**: Z-score < -n_std_lower (oversold)
- **Exit**: Z-score > n_std_upper OR Z-score reverts to exit_threshold

---

### 3. Mean Reversion RSI + SMA

Combines RSI momentum indicator with SMA trend filter for mean reversion in uptrends.

| Parameter | Description |
|-----------|-------------|
| `rsi_period` | Lookback period for RSI calculation |
| `rsi_oversold` | RSI threshold for entry (e.g., 30) |
| `rsi_overbought` | RSI threshold for exit (e.g., 70) |
| `sma_period` | SMA period for trend confirmation |

**Signal Logic**:
- **Entry**: RSI < oversold AND Price > SMA (buy dips in uptrend)
- **Exit**: RSI > overbought OR Price < SMA

---

### 4. Statistical Arbitrage (Pairs Trading)

Trades the spread between two correlated assets based on z-score deviation.

| Parameter | Description |
|-----------|-------------|
| `zscore_lookback` | Lookback for spread mean and std |
| `entry_threshold` | Z-score level to enter spread trade |
| `exit_threshold` | Z-score level to exit (typically 0) |
| `use_dynamic_hedge` | Recalculate hedge ratio dynamically |

**Signal Logic**:
- **Entry**: Spread Z-score exceeds entry threshold
- **Exit**: Spread Z-score reverts to exit threshold

---

### 5. Supervised Learning (XGBoost, Random Forest)

Generates signals based on a trained model with 50+ features consisting of volume, volatility and price action intraday data.

| Parameter | Description |
|-----------|-------------|
| `lookback_window` | Lookback window for observations |
| `forecast_horizon` | The number of ticks before generating signal |
| `return_threshold` | Label for position exit to take profit (0.1%) |
| `class_weight` | How to handle class imbalance during training |

**Signal Logic**:
- **Entry**: Signal = 1
- **Hold**: Signal = 0
- **Exit**: Signal = -1 (threshold > 0.1%)

---

## Key Considerations for Realistic Backtests

The backtesting logic implements several safeguards to ensure backtest results are realistic and not inflated by common pitfalls.

### 1. Look-Ahead Bias Prevention

The `simulate_positions_numba` method uses **pending order logic** where:
- Signals are evaluated at valid trading times
- Orders queue for execution at the *next* valid trading time
- Stop-loss and take-profit execute immediately (realistic for market orders)

### 2. Realistic Execution Costs

| Cost Component | Implementation |
|----------------|----------------|
| **Slippage** | Configurable basis points applied symmetrically to buys/sells |
| **Commission** | Per-share commission rate |
| **Market Impact** | Implicit in slippage modeling |

```python
# Buy: Pay more than quoted price
effective_price = price * (1 + slippage_bps / 10000)

# Sell: Receive less than quoted price  
net_proceeds = shares * price * (1 - slippage_bps / 10000)
```

### 3. Position Valuation

Positions are always valued at the **current close price**, not the entry price. This ensures accurate P&L even on entry days.

### 4. Signal Priority

When conflicting signals occur (e.g., both entry and exit conditions true), **exit signals take priority**. This prevents overriding risk management decisions.

### 5. Trading Frequency Enforcement

Signals are only evaluated at specified intervals (e.g., every 30 minutes), preventing unrealistic high-frequency trading assumptions (this is not a HFT project).

### 6. Risk Management

| Control | Description |
|---------|-------------|
| **Stop-Loss** | Exit when position loss exceeds threshold |
| **Take-Profit** | Exit when position gain exceeds threshold |
| **Max Drawdown** | Circuit breaker - flatten all positions if portfolio drawdown exceeds limit |

### 7. Volatility-Based Position Sizing

Position sizes scale inversely with volatility to maintain consistent portfolio risk:

```python
size_factor = min(target_volatility / current_volatility, max_leverage)
```

Optionally, **intraday rebalancing** adjusts positions when volatility changes significantly.

---

## Project Structure

```
algo-backtests/
├── backtest.py           # Core backtesting module
├── simulation.py         # Parameter sweep with parallel execution
├── strategies.py         # Signal generation and position simulation
├── benchmarks.py         # Benchmark comparison metrics
├── outputs.py            # Formatted output generation
├── utils.py              # Utility functions (Sharpe, VaR, etc.)
├── constants.py          # Configuration constants
├── data.py               # Polygon.io data fetching
│
├── strats/
│   ├── momentum.py            # Momentum strategy implementation
│   ├── mean_reversion.py      # Z-score mean reversion
│   ├── mean_reversion_rsi.py  # RSI + SMA mean reversion
│   └── stat_arb.py            # Statistical arbitrage (pairs)
└── 
```

---

## Usage

### Installation

```bash
pip install numpy pandas numba matplotlib python-dotenv requests pytz
```

### Environment Setup

Create a `.env` file with your Polygon.io API key:

```
POLYGON_API_KEY=your_api_key_here
```

### Running Simulations

#### Single Strategy

```bash
# Run momentum strategy
python simulation.py --strategy momentum

# Run mean reversion
python simulation.py --strategy mean_reversion

# Run RSI mean reversion
python simulation.py --strategy mean_reversion_rsi

# Run statistical arbitrage
python simulation.py --strategy stat_arb
```

#### All Strategies

```bash
python simulation.py --strategy all
```

#### Sequential Mode (for debugging)

```bash
python simulation.py --strategy momentum --sequential
```

---

## Performance Metrics

### Strategy Metrics

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative percentage return |
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) |
| **Sortino Ratio** | Downside risk-adjusted return |
| **Calmar Ratio** | Annualized return / max drawdown |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of profitable trading days |
| **Annualized Volatility** | Standard deviation of returns × √252 |

### Benchmark Comparison Metrics

| Metric | Description |
|--------|-------------|
| **Beta** | Sensitivity to benchmark movements |
| **Alpha** | Excess return beyond CAPM prediction |
| **Correlation** | Linear relationship with benchmark |
| **R-Squared** | Variance explained by benchmark |
| **Information Ratio** | Risk-adjusted excess return vs benchmark |
| **Treynor Ratio** | Excess return per unit of beta risk |

---

## Example Output

```
====================================================================================================
BEST MEAN_REVERSION STRATEGY CONFIGURATION (by Sharpe Ratio)
====================================================================================================

--------------------------------------------------
NVDA
--------------------------------------------------

Strategy Parameters:
  • Z-Score Lookback: 120
  • N Std Upper: 1.5
  • N Std Lower: 2.0
  • Trading Frequency: 30 min
  • Target Volatility: 0.015%
  • Max Drawdown Limit: -14.0%

Trading Activity:
  • Total Trades: 404
  • Buy Trades: 202
  • Sell Trades: 202
  • Stop Losses Hit: 62
  • Take Profits Hit: 3

Strategy Performance:
  • Total Return: 93.23%
  • Sharpe Ratio: 1.7507
  • Sortino Ratio: 3.1113
  • Calmar Ratio: 2.8665
  • Max Drawdown: -13.97%
  • Win Rate: 24.54%
  • Beta: 0.1302
  • Alpha (annualized): 20.98%
  • Information Ratio: -0.9195

Benchmark (Buy & Hold):
  • Total Return: 261.53%
  • Sharpe Ratio: 1.4485
  • Sortino Ratio: 1.4825
  • Calmar Ratio: 2.5181
```

---

## Configuration Reference

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `AUM` | 100,000 | Starting capital |
| `trade_freq` | 30 | Minutes between signal evaluations |
| `stop_loss_pct` | 0.02 | Stop-loss threshold (2%) |
| `take_profit_pct` | 0.04 | Take-profit threshold (4%) |
| `max_drawdown_pct` | 0.15 | Circuit breaker threshold (15%) |
| `target_vol` | 0.02 | Target portfolio volatility |
| `max_leverage` | 1.0 | Maximum position leverage |
| `commission` | 0.0035 | Commission per share |
| `slippage_factor` | 0.1 | Slippage in basis points |

### Volatility Rebalancing (Optional)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_vol_rebalance` | False | Enable intraday rebalancing |
| `vol_rebalance_threshold` | 0.25 | Volatility change to trigger rebalance |
| `min_rebalance_size` | 0.10 | Minimum position change to execute |

---

## Acknowledgments

- [Polygon.io](https://polygon.io/) for market data
- [Numba](https://numba.pydata.org/) for JIT compilation
- [NumPy](https://numpy.org/) for vectorized operations