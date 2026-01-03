# algo-backtests/strats/stat_arb/stat_arb.py
import numpy as np
from numba import njit
from typing import Tuple, Dict, Any


@njit(cache=True)
def calculate_spread_numba(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    hedge_ratio: float
) -> np.ndarray:
    """
    JIT-compiled spread calculation between two price series.
    
    Args:
        prices_a: Array of prices for asset A
        prices_b: Array of prices for asset B
        hedge_ratio: Fixed hedge ratio (if None, uses dynamic calculation)

    Returns:
        Array of spreads
    """
    n = len(prices_a)
    spread = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        spread[i] = prices_a[i] - (hedge_ratio * prices_b[i])
    
    return spread


@njit(cache=True)
def calculate_rolling_hedge_ratio_numba(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    lookback: int
) -> np.ndarray:
    """
    JIT-compiled rolling hedge ratio calculation.
    
    Args:
        prices_a: Array of prices for asset A
        prices_b: Array of prices for asset B
        lookback: Lookback period
        
    Returns:
        Array of hedge ratios
    """
    n = len(prices_a)
    hedge_ratios = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        window_a = prices_a[i - lookback:i]
        window_b = prices_b[i - lookback:i]
        mean_b = np.mean(window_b)
        if mean_b > 0:
            hedge_ratios[i] = np.mean(window_a) / mean_b
        else:
            hedge_ratios[i] = 1.0
    
    return hedge_ratios


@njit(cache=True)
def calculate_spread_zscore_numba(
    spread: np.ndarray,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JIT-compiled z-score calculation for spread series.
    
    Args:
        spread: Array of spreads
        lookback: Lookback period
        
    Returns:
        Tuple of (zscore, rolling_mean, rolling_std)
    """
    n = len(spread)
    rolling_mean = np.full(n, np.nan, dtype=np.float64)
    rolling_std = np.full(n, np.nan, dtype=np.float64)
    zscore = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        window = spread[i - lookback:i]
        rolling_mean[i] = np.mean(window)
        rolling_std[i] = np.std(window)
        
        if rolling_std[i] > 0:
            zscore[i] = (spread[i] - rolling_mean[i]) / rolling_std[i]
    
    return zscore, rolling_mean, rolling_std


@njit(cache=True)
def simulate_positions_stat_arb_numba(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    signals: np.ndarray,
    hedge_ratios: np.ndarray,
    trade_freq_mask: np.ndarray,
    stop_loss_pct: float,
    take_profit_pct: float,
    initial_position: int,
    initial_entry_spread: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """JIT-compiled position simulation for statistical arbitrage.
    
    Args:
        prices_a: Array of prices for asset A
        prices_b: Array of prices for asset B
        signals: Array of signals
        hedge_ratios: Array of hedge ratios
        trade_freq_mask: Array of trade frequency mask
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        initial_position: Initial position
        initial_entry_spread: Initial entry spread
        
    Returns:
        Tuple of (positions, entry_spreads, exit_reasons, stop_losses, take_profits)
    """
    n = len(prices_a)
    positions = np.zeros(n, dtype=np.int32)
    entry_spreads = np.zeros(n, dtype=np.float64)
    exit_reasons = np.zeros(n, dtype=np.int32)
    
    current_pos = initial_position
    entry_spread = initial_entry_spread
    stop_losses = 0
    take_profits = 0
    
    pending_action = 0
    pending_signal_bar = -1
    
    for i in range(n):
        # Calculate current spread
        if not np.isnan(hedge_ratios[i]):
            current_spread = prices_a[i] - (hedge_ratios[i] * prices_b[i])
        else:
            current_spread = np.nan
        
        # Execute pending orders
        if pending_action != 0 and i > pending_signal_bar:
            if not trade_freq_mask[i]:
                positions[i] = current_pos
                entry_spreads[i] = entry_spread
                continue
            
            if pending_action == 1 and current_pos == 0:
                current_pos = 1
                entry_spread = current_spread
                pending_action = 0
                pending_signal_bar = -1
            elif pending_action == -1 and current_pos == 1:
                current_pos = 0
                entry_spread = 0.0
                exit_reasons[i] = 1
                pending_action = 0
                pending_signal_bar = -1
        
        # Check stop-loss and take-profit based on spread
        if current_pos > 0 and entry_spread != 0 and not np.isnan(current_spread):
            spread_pnl_pct = (current_spread - entry_spread) / abs(entry_spread)
            
            if spread_pnl_pct <= -stop_loss_pct:
                current_pos = 0
                entry_spread = 0.0
                exit_reasons[i] = 2
                stop_losses += 1
                pending_action = 0
                pending_signal_bar = -1
                positions[i] = current_pos
                entry_spreads[i] = entry_spread
                continue
            
            if spread_pnl_pct >= take_profit_pct:
                current_pos = 0
                entry_spread = 0.0
                exit_reasons[i] = 3
                take_profits += 1
                pending_action = 0
                pending_signal_bar = -1
                positions[i] = current_pos
                entry_spreads[i] = entry_spread
                continue
        
        # Evaluate signals
        if trade_freq_mask[i] and pending_action == 0:
            signal = signals[i]
            
            if signal == 1 and current_pos == 0:
                pending_action = 1
                pending_signal_bar = i
            elif signal == -1 and current_pos == 1:
                pending_action = -1
                pending_signal_bar = i
        
        positions[i] = current_pos
        entry_spreads[i] = entry_spread
    
    return positions, entry_spreads, exit_reasons, stop_losses, take_profits


@njit(cache=True)
def calculate_portfolio_values_stat_arb_numba(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    positions: np.ndarray,
    hedge_ratios: np.ndarray,
    entry_spreads: np.ndarray,
    initial_cash: float,
    initial_shares_a: int,
    initial_shares_b: int,
    commission_rate: float,
    slippage_bps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """JIT-compiled portfolio value calculation for statistical arbitrage.
    
    Args:
        prices_a: Array of prices for asset A
        prices_b: Array of prices for asset B
        positions: Array of positions
        hedge_ratios: Array of hedge ratios
        entry_spreads: Array of entry spreads
        initial_cash: Initial cash
        initial_shares_a: Initial shares of asset A
        initial_shares_b: Initial shares of asset B
        commission_rate: Commission rate
        slippage_bps: Slippage basis points
        
    Returns:
        Tuple of (cash, shares_a, shares_b, portfolio_values, commissions, slippages, buy_count, sell_count)
    """
    n = len(prices_a)
    cash = np.zeros(n, dtype=np.float64)
    shares_a = np.zeros(n, dtype=np.int32)
    shares_b = np.zeros(n, dtype=np.int32)
    portfolio_values = np.zeros(n, dtype=np.float64)
    commissions = np.zeros(n, dtype=np.float64)
    slippages = np.zeros(n, dtype=np.float64)
    
    running_cash = initial_cash
    running_shares_a = initial_shares_a
    running_shares_b = initial_shares_b
    last_position = 1 if initial_shares_a > 0 else 0
    buy_count = 0
    sell_count = 0
    
    entry_bar = -1
    actual_entry_price_a = 0.0
    
    for i in range(n):
        price_a = prices_a[i]
        target_pos = positions[i]
        
        if target_pos != last_position:
            # BUY spread: Long A
            if target_pos == 1 and last_position == 0:
                slippage_factor = slippage_bps / 10000.0
                effective_price = price_a * (1 + slippage_factor)
                
                max_shares = int(running_cash * 0.99 / (effective_price + commission_rate))
                
                if max_shares > 0:
                    order_size = max_shares
                    cost = order_size * price_a
                    slip = order_size * price_a * slippage_factor
                    comm = order_size * commission_rate
                    
                    running_cash -= (cost + slip + comm)
                    running_shares_a = order_size
                    
                    commissions[i] = comm
                    slippages[i] = slip
                    buy_count += 1
                    
                    entry_bar = i
                    actual_entry_price_a = effective_price
            
            # SELL spread: Close long A
            elif target_pos == 0 and last_position == 1:
                if running_shares_a > 0:
                    order_size = running_shares_a
                    proceeds = order_size * price_a
                    slippage_factor = slippage_bps / 10000.0
                    slip = order_size * price_a * slippage_factor
                    comm = order_size * commission_rate
                    
                    running_cash += (proceeds - slip - comm)
                    running_shares_a = 0
                    
                    commissions[i] = comm
                    slippages[i] = slip
                    sell_count += 1
                    
                    entry_bar = -1
                    actual_entry_price_a = 0.0
        
        # Portfolio valuation
        if running_shares_a > 0:
            if i == entry_bar:
                position_value = running_shares_a * actual_entry_price_a
            else:
                position_value = running_shares_a * price_a
        else:
            position_value = 0.0
        
        cash[i] = running_cash
        shares_a[i] = running_shares_a
        shares_b[i] = running_shares_b
        portfolio_values[i] = running_cash + position_value
        last_position = target_pos
    
    return cash, shares_a, shares_b, portfolio_values, commissions, slippages, buy_count, sell_count


class StatArb:
    """
    Statistical Arbitrage (Pairs Trading) Strategy.
    Uses static methods for stateless signal generation.
    """
    
    def __init__(
        self,
        zscore_lookback: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        hedge_ratio: float = None,
        use_dynamic_hedge: bool = True
    ):
        self.zscore_lookback = zscore_lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.hedge_ratio = hedge_ratio
        self.use_dynamic_hedge = use_dynamic_hedge

    
    def get_parameter_grid(self) -> Dict[str, list]:
        """
        Get parameter grid for stat arb strategy.
        
        Returns:
            Dictionary of parameter names to lists of values to test
        """
        return {
            'stat_arb_zscore_lookbacks': self.zscore_lookback,
            'entry_thresholds': self.entry_threshold,
            'exit_thresholds': self.exit_threshold,
            'hedge_ratio': self.hedge_ratio,
            'use_dynamic_hedge': self.use_dynamic_hedge
        }

    
    @staticmethod
    def generate_signals(
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        lookback_period: int,
        entry_threshold: float,
        exit_threshold: float = 0.0,
        hedge_ratio: float = None,
        use_dynamic_hedge: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized signal generation for statistical arbitrage (pairs trading).
        
        Args:
            prices_a: Array of prices for asset A (primary asset to trade)
            prices_b: Array of prices for asset B (reference/hedge asset)
            lookback_period: Number of periods for MA and StdDev calculation
            entry_threshold: Z-score threshold for entry (e.g., 2.0)
            exit_threshold: Z-score threshold for exit (default 0.0)
            hedge_ratio: Fixed hedge ratio (if None, uses dynamic calculation)
            use_dynamic_hedge: Whether to recalculate hedge ratio rolling
            
        Returns:
            Tuple of (signals, zscore, spread, hedge_ratios)
        """
        n = len(prices_a)
        
        # Calculate hedge ratio
        if use_dynamic_hedge or hedge_ratio is None:
            hedge_ratios = calculate_rolling_hedge_ratio_numba(
                prices_a.astype(np.float64),
                prices_b.astype(np.float64),
                lookback_period
            )
        else:
            hedge_ratios = np.full(n, hedge_ratio, dtype=np.float64)
        
        # Calculate spread
        spread = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if not np.isnan(hedge_ratios[i]):
                spread[i] = prices_a[i] - (hedge_ratios[i] * prices_b[i])
            else:
                spread[i] = np.nan
        
        # Calculate z-score of spread
        zscore, spread_ma, spread_std = calculate_spread_zscore_numba(
            spread.astype(np.float64),
            lookback_period
        )
        
        # Initialize signals
        signals = np.zeros(n, dtype=np.int32)
        
        # Long spread signal: z-score below -entry_threshold
        long_condition = zscore < -entry_threshold
        
        # Exit signal
        exit_condition = (zscore > entry_threshold) | (
            (zscore >= exit_threshold) & (zscore <= entry_threshold)
        )
        
        signals[long_condition] = 1
        signals[exit_condition] = -1
        
        return signals, zscore, spread, hedge_ratios

    
    @staticmethod
    def generate_signals_symmetric(
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        lookback_period: int,
        entry_threshold: float,
        exit_threshold: float = 0.0,
        hedge_ratio: float = None,
        use_dynamic_hedge: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Symmetric statistical arbitrage signal generation.
        Generates both long and short spread signals.
        
        Args:
            prices_a: Array of prices for asset A
            prices_b: Array of prices for asset B
            lookback_period: Number of periods for MA and StdDev calculation
            entry_threshold: Z-score threshold for entry (e.g., 2.0)
            exit_threshold: Z-score threshold for exit (default 0.0)
            hedge_ratio: Fixed hedge ratio (if None, uses dynamic calculation)
            use_dynamic_hedge: Whether to recalculate hedge ratio rolling
            
        Returns:
            Tuple of (signals, zscore, spread, hedge_ratios)
        """
        n = len(prices_a)
        
        if use_dynamic_hedge or hedge_ratio is None:
            hedge_ratios = calculate_rolling_hedge_ratio_numba(
                prices_a.astype(np.float64),
                prices_b.astype(np.float64),
                lookback_period
            )
        else:
            hedge_ratios = np.full(n, hedge_ratio, dtype=np.float64)
        
        spread = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if not np.isnan(hedge_ratios[i]):
                spread[i] = prices_a[i] - (hedge_ratios[i] * prices_b[i])
            else:
                spread[i] = np.nan
        
        zscore, spread_ma, spread_std = calculate_spread_zscore_numba(
            spread.astype(np.float64),
            lookback_period
        )
        
        signals = np.zeros(n, dtype=np.int32)
        
        long_condition = zscore < -entry_threshold
        short_condition = zscore > entry_threshold
        exit_condition = (zscore >= -abs(exit_threshold)) & (zscore <= abs(exit_threshold))
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        signals[exit_condition] = 0
        
        return signals, zscore, spread, hedge_ratios

    
    @staticmethod
    def simulate_positions(
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        signals: np.ndarray,
        hedge_ratios: np.ndarray,
        trade_freq_mask: np.ndarray,
        stop_loss_pct: float,
        take_profit_pct: float,
        initial_position: int,
        initial_entry_spread: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Simulate positions for statistical arbitrage strategy.
        
        Args:
            prices_a: Array of prices for asset A
            prices_b: Array of prices for asset B
            signals: Array of signals
            hedge_ratios: Array of hedge ratios
            trade_freq_mask: Array of trade frequency mask
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            initial_position: Initial position
            initial_entry_spread: Initial entry spread
            
        Returns:
            Tuple of (positions, entry_spreads, exit_reasons, stop_losses, take_profits)
        """
        return simulate_positions_stat_arb_numba(
            prices_a.astype(np.float64),
            prices_b.astype(np.float64),
            signals,
            hedge_ratios,
            trade_freq_mask,
            stop_loss_pct,
            take_profit_pct,
            initial_position,
            initial_entry_spread
        )

    
    @staticmethod
    def calculate_portfolio_values(
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        positions: np.ndarray,
        hedge_ratios: np.ndarray,
        entry_spreads: np.ndarray,
        initial_cash: float,
        initial_shares_a: int,
        initial_shares_b: int,
        commission_rate: float,
        slippage_bps: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Calculate portfolio values for statistical arbitrage strategy.
        
        Args:
            prices_a: Array of prices for asset A
            prices_b: Array of prices for asset B
            positions: Array of positions
            hedge_ratios: Array of hedge ratios
            entry_spreads: Array of entry spreads
            initial_cash: Initial cash
            initial_shares_a: Initial shares of asset A
            initial_shares_b: Initial shares of asset B
            commission_rate: Commission rate
            slippage_bps: Slippage basis points
            
        Returns:
            Tuple of (cash, shares_a, shares_b, portfolio_values, commissions, slippages, buy_count, sell_count)
        """
        return calculate_portfolio_values_stat_arb_numba(
            prices_a.astype(np.float64),
            prices_b.astype(np.float64),
            positions,
            hedge_ratios,
            entry_spreads,
            initial_cash,
            initial_shares_a,
            initial_shares_b,
            commission_rate,
            slippage_bps
        )
