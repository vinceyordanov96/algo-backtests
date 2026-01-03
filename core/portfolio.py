"""
Portfolio simulation module with Numba-optimized functions.

This module provides the core position management and portfolio value
calculation functions used by all trading strategies.

Functions:
    - simulate_positions_numba: Position simulation with stop-loss/take-profit
    - calculate_portfolio_values_numba: Portfolio value tracking with transaction costs
    - calculate_position_size_numba: Position size calculation based on rolling Sharpe ratio
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def simulate_positions_numba(
    prices: np.ndarray,
    signals: np.ndarray,
    trade_freq_mask: np.ndarray,
    stop_loss_pct: float,
    take_profit_pct: float,
    initial_position: int,
    initial_entry_price: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    JIT-compiled position simulation with stop-loss and take-profit logic.
    
    This is the core position management function used by all non-pairs
    strategies. It handles:
        - Signal-based entry/exit
        - Stop-loss triggers
        - Take-profit triggers
        - Trade frequency constraints
    
    Args:
        prices: Array of close prices
        signals: Array of raw signals (1=long, -1=exit, 0=hold)
        trade_freq_mask: Boolean array indicating valid trading times
        stop_loss_pct: Stop-loss threshold as decimal (e.g., 0.02 for 2%)
        take_profit_pct: Take-profit threshold as decimal
        initial_position: Starting position (0=flat, 1=long)
        initial_entry_price: Entry price if already in position
        
    Returns:
        Tuple of:
            - positions: Array of position states (0 or 1)
            - entry_prices: Array of entry prices
            - exit_reasons: 0=none, 1=signal, 2=stop_loss, 3=take_profit
            - stop_losses_count: Number of stop-loss exits
            - take_profits_count: Number of take-profit exits
    """
    n = len(prices)
    positions = np.zeros(n, dtype=np.int32)
    entry_prices = np.zeros(n, dtype=np.float64)
    exit_reasons = np.zeros(n, dtype=np.int32)
    
    current_pos = initial_position
    entry_price = initial_entry_price
    stop_losses = 0
    take_profits = 0
    
    # Pending order state: 0=none, 1=pending entry, -1=pending exit
    pending_action = 0
    pending_signal_bar = -1
    
    for i in range(n):
        # Execute pending orders only at valid trading times after signal bar
        if pending_action != 0 and i > pending_signal_bar:
            if not trade_freq_mask[i]:
                # Wait for next valid trading time
                positions[i] = current_pos
                entry_prices[i] = entry_price
                continue
            
            # Execute pending entry
            if pending_action == 1 and current_pos == 0:
                current_pos = 1
                entry_price = prices[i]
                pending_action = 0
                pending_signal_bar = -1
            # Execute pending exit
            elif pending_action == -1 and current_pos == 1:
                current_pos = 0
                entry_price = 0.0
                exit_reasons[i] = 1
                pending_action = 0
                pending_signal_bar = -1
        
        # Check stop-loss and take-profit (immediate execution)
        if current_pos > 0 and entry_price > 0:
            pnl_pct = (prices[i] - entry_price) / entry_price
            
            # Stop-loss triggered
            if pnl_pct <= -stop_loss_pct:
                current_pos = 0
                entry_price = 0.0
                exit_reasons[i] = 2
                stop_losses += 1
                pending_action = 0
                pending_signal_bar = -1
                positions[i] = current_pos
                entry_prices[i] = entry_price
                continue
            
            # Take-profit triggered
            if pnl_pct >= take_profit_pct:
                current_pos = 0
                entry_price = 0.0
                exit_reasons[i] = 3
                take_profits += 1
                pending_action = 0
                pending_signal_bar = -1
                positions[i] = current_pos
                entry_prices[i] = entry_price
                continue
        
        # Evaluate signals only at valid trading times
        if trade_freq_mask[i] and pending_action == 0:
            signal = signals[i]
            
            # Exit signals take priority over entry signals
            if signal == -1 and current_pos == 1:
                # Queue exit for next valid trading time
                pending_action = -1
                pending_signal_bar = i
            elif signal == 1 and current_pos == 0:
                # Queue entry for next valid trading time
                pending_action = 1
                pending_signal_bar = i
        
        # Record current state
        positions[i] = current_pos
        entry_prices[i] = entry_price
    
    return positions, entry_prices, exit_reasons, stop_losses, take_profits


@njit(cache=True)
def calculate_portfolio_values_numba(
    prices: np.ndarray,
    positions: np.ndarray,
    entry_prices: np.ndarray,
    initial_cash: float,
    initial_shares: int,
    commission_rate: float,
    slippage_bps: float,
    size_factor: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    JIT-compiled portfolio value calculation with transaction costs.
    
    Calculates mark-to-market portfolio values accounting for:
        - Commission costs per share
        - Slippage on order execution
        - Cash and position tracking
        - Position sizing via size_factor
    
    Args:
        prices: Array of close prices
        positions: Array of target positions (0 or 1)
        entry_prices: Array of entry prices (from simulate_positions_numba)
        initial_cash: Starting cash balance
        initial_shares: Starting shares held
        commission_rate: Commission per share
        slippage_bps: Slippage in basis points
        size_factor: Fraction of portfolio to allocate (0.0 to 1.0+)
        
    Returns:
        Tuple of:
            - cash: Array of cash balances
            - shares: Array of share holdings
            - portfolio_values: Array of total portfolio values
            - commissions: Array of commission costs
            - slippages: Array of slippage costs
            - buy_count: Number of buy trades
            - sell_count: Number of sell trades
    """
    n = len(prices)
    cash = np.zeros(n, dtype=np.float64)
    shares = np.zeros(n, dtype=np.int32)
    portfolio_values = np.zeros(n, dtype=np.float64)
    commissions = np.zeros(n, dtype=np.float64)
    slippages = np.zeros(n, dtype=np.float64)
    
    running_cash = initial_cash
    running_shares = initial_shares
    last_position = 1 if initial_shares > 0 else 0
    buy_count = 0
    sell_count = 0
    
    slippage_factor = slippage_bps / 10000.0
    
    for i in range(n):
        price = prices[i]
        target_pos = positions[i]
        
        if target_pos != last_position:
            if target_pos == 1 and last_position == 0:
                effective_price = price * (1.0 + slippage_factor)
                
                total_equity = running_cash + (running_shares * price if running_shares > 0 else 0.0)
                target_investment = total_equity * size_factor
                
                max_affordable = int(running_cash * 0.99 / (effective_price + commission_rate))
                target_shares = int(target_investment / effective_price)
                
                order_size = min(max_affordable, target_shares)
                
                if order_size > 0:
                    slip = order_size * price * slippage_factor
                    comm = order_size * commission_rate
                    total_cost = order_size * price + slip + comm
                    
                    running_cash -= total_cost
                    running_shares = order_size
                    
                    commissions[i] = comm
                    slippages[i] = slip
                    buy_count += 1
            
            elif target_pos == 0 and last_position == 1:
                if running_shares > 0:
                    order_size = running_shares
                    
                    slip = order_size * price * slippage_factor
                    comm = order_size * commission_rate
                    net_proceeds = order_size * price - slip - comm
                    
                    running_cash += net_proceeds
                    running_shares = 0
                    
                    commissions[i] = comm
                    slippages[i] = slip
                    sell_count += 1
        
        if running_shares > 0:
            position_value = running_shares * price
        else:
            position_value = 0.0
        
        cash[i] = running_cash
        shares[i] = running_shares
        portfolio_values[i] = running_cash + position_value
        last_position = target_pos
    
    return cash, shares, portfolio_values, commissions, slippages, buy_count, sell_count


@njit(cache=True)
def calculate_position_size_numba(
    returns: np.ndarray,
    base_size: int,
    lookback: int
) -> int:
    """
    JIT-compiled position size calculation based on rolling Sharpe ratio.
    
    Args:
        returns: Array of previous returns
        base_size: Base position size
        lookback: Lookback period
        
    Returns:
        Adjusted position size
    """
    if len(returns) < lookback:
        return base_size
    
    recent = returns[-lookback:]
    mean_ret = np.mean(recent)
    std_ret = np.std(recent)
    
    if std_ret <= 0:
        sharpe = 0.0
    else:
        sharpe = mean_ret / std_ret
    
    # Scale factor based on Sharpe
    scale = 1.0 + 0.2 * sharpe
    scale = max(0.5, min(1.5, scale))  # Clip to [0.5, 1.5]
    
    return int(base_size * scale)


@njit(cache=True)
def calculate_kelly_position_size_numba(
    returns: np.ndarray,
    base_size: int,
    lookback: int,
    kelly_fraction: float = 0.5,
    max_leverage: float = 1.0,
    min_trades: int = 30
) -> Tuple[int, float]:
    """
    JIT-compiled position size calculation using Kelly Criterion.
    
    Uses the trade-based Kelly formula:
        f* = W - (1-W)/R
    
    Where:
        W = win rate (probability of positive return)
        R = win/loss ratio (avg win / avg loss)
    
    Args:
        returns: Array of period returns (can be trade returns or bar returns)
        base_size: Base position size (e.g., max shares or units)
        lookback: Number of periods to use for estimation
        kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly, common choice)
        max_leverage: Maximum position as multiple of base (1.0 = no leverage)
        min_trades: Minimum observations before applying Kelly sizing
        
    Returns:
        Tuple of:
            - position_size: Adjusted position size
            - raw_kelly: The raw Kelly fraction before scaling
            
    Notes:
        - Half-Kelly (0.5) is commonly used to reduce volatility and drawdowns
        - Quarter-Kelly (0.25) is even more conservative
        - Returns a minimum of 0 if Kelly is negative (edge is negative)
    """
    n = len(returns)
    
    # Insufficient data - return base size
    if n < min_trades:
        return base_size, 0.0
    
    # Get recent returns
    if n > lookback:
        recent = returns[-lookback:]
    else:
        recent = returns
    
    # Separate wins and losses
    n_total = len(recent)
    n_wins = 0
    n_losses = 0
    sum_wins = 0.0
    sum_losses = 0.0
    
    for i in range(n_total):
        ret = recent[i]
        if ret > 0.0:
            n_wins += 1
            sum_wins += ret
        elif ret < 0.0:
            n_losses += 1
            sum_losses += ret  # This will be negative
    
    # Handle edge cases
    if n_wins == 0:
        # No winning trades - Kelly says don't bet
        return 0, -1.0
    
    if n_losses == 0:
        # All winners - use full allocation (capped)
        kelly_raw = 1.0
        kelly_scaled = min(kelly_fraction, max_leverage)
        return int(base_size * kelly_scaled), kelly_raw
    
    # Calculate win rate
    win_prob = n_wins / n_total
    loss_prob = 1.0 - win_prob
    
    # Calculate average win and average loss (absolute)
    avg_win = sum_wins / n_wins
    avg_loss = -sum_losses / n_losses  # Make positive
    
    # Win/loss ratio (R)
    if avg_loss < 1e-10:
        avg_loss = 1e-10  # Avoid division by zero
    
    win_loss_ratio = avg_win / avg_loss
    
    # Kelly formula: f* = W - (1-W)/R
    kelly_raw = win_prob - (loss_prob / win_loss_ratio)
    
    # Apply fractional Kelly and bounds
    kelly_scaled = kelly_raw * kelly_fraction
    kelly_bounded = max(0.0, min(max_leverage, kelly_scaled))
    
    # Calculate final position size
    position_size = int(base_size * kelly_bounded)
    
    return position_size, kelly_raw


@njit(cache=True)
def calculate_kelly_continuous_numba(
    returns: np.ndarray,
    base_size: int,
    lookback: int,
    kelly_fraction: float = 0.5,
    max_leverage: float = 1.0,
    min_periods: int = 30,
    annualization_factor: float = 252.0
) -> Tuple[int, float]:
    """
    JIT-compiled Kelly sizing using continuous/Gaussian approximation.
    
    Uses the formula for normally distributed returns:
        f* = μ / σ²
    
    This is derived from maximizing log-wealth growth rate and is
    appropriate when returns are approximately normally distributed.
    
    Args:
        returns: Array of period returns
        base_size: Base position size
        lookback: Lookback period for estimation
        kelly_fraction: Fraction of full Kelly to use
        max_leverage: Maximum leverage allowed
        min_periods: Minimum periods before applying Kelly
        annualization_factor: Periods per year (252 for daily, 52 for weekly)
        
    Returns:
        Tuple of:
            - position_size: Adjusted position size
            - raw_kelly: Raw Kelly fraction (can be > 1 for high Sharpe strategies)
            
    Notes:
        - For a Sharpe ratio of S, Kelly ≈ S/σ annually, or S²/μ
        - This formulation can suggest leverage (f* > 1) for high Sharpe strategies
    """
    n = len(returns)
    
    if n < min_periods:
        return base_size, 0.0
    
    # Get recent returns
    if n > lookback:
        recent = returns[-lookback:]
    else:
        recent = returns
    
    # Calculate mean and variance
    mean_ret = np.mean(recent)
    var_ret = np.var(recent)
    
    # Handle zero or near-zero variance
    if var_ret < 1e-16:
        if mean_ret > 0:
            return int(base_size * min(kelly_fraction, max_leverage)), 1.0
        else:
            return 0, 0.0
    
    # Kelly formula for continuous returns: f* = μ / σ²
    kelly_raw = mean_ret / var_ret
    
    # Apply fractional Kelly and bounds
    kelly_scaled = kelly_raw * kelly_fraction
    kelly_bounded = max(0.0, min(max_leverage, kelly_scaled))
    
    position_size = int(base_size * kelly_bounded)
    
    return position_size, kelly_raw
