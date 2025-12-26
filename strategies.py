import numpy as np
from numba import njit
from typing import Dict, Tuple, Any, Callable
from enum import Enum

from strats.momentum import Momentum
from strats.mean_reversion import MeanReversion
from strats.mean_reversion_rsi import MeanReversionRSI
from strats.stat_arb import StatArb


class StrategyType(Enum):
    """
    Enumeration of available trading strategies.
    """
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    MEAN_REVERSION_RSI = "mean_reversion_rsi"
    STAT_ARB = "stat_arb"


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
    
    Args:
        prices: Array of close prices
        signals: Array of raw signals (1=long, -1=exit, 0=hold)
        trade_freq_mask: Boolean array indicating valid trading times
        stop_loss_pct: Stop-loss threshold as decimal
        take_profit_pct: Take-profit threshold as decimal
        initial_position: Starting position (0 or 1)
        initial_entry_price: Entry price if already in position
        
    Returns:
        Tuple of (positions, entry_prices, exit_reasons, stop_losses_count, take_profits_count)
        exit_reasons: 0=none, 1=signal, 2=stop_loss, 3=take_profit
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
            # This handles cases where both conditions might be true
            if signal == -1 and current_pos == 1:
                # Queue exit for next valid trading time
                pending_action = -1
                pending_signal_bar = i
            elif signal == 1 and current_pos == 0:
                # Queue entry for next valid trading time (only if not exiting)
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
    slippage_bps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    JIT-compiled portfolio value calculation with transaction costs.
    
    Args:
        prices: Array of close prices
        positions: Array of target positions (0 or 1)
        entry_prices: Array of entry prices (from simulate_positions_numba)
        initial_cash: Starting cash balance
        initial_shares: Starting shares held
        commission_rate: Commission per share
        slippage_bps: Slippage in basis points
        
    Returns:
        Tuple of (cash, shares, portfolio_values, commissions, slippages, buy_count, sell_count)
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
        
        # Position change detection
        if target_pos != last_position:
            # BUY: Going from flat to long
            if target_pos == 1 and last_position == 0:
                # Calculate effective price with slippage upfront
                effective_price = price * (1.0 + slippage_factor)
                
                # Calculate max shares we can buy with slippage and commission
                max_shares = int(running_cash * 0.99 / (effective_price + commission_rate))
                
                if max_shares > 0:
                    order_size = max_shares
                    
                    # Symmetric cost calculation
                    # Total cost = shares * effective_price + commission
                    slip = order_size * price * slippage_factor
                    comm = order_size * commission_rate
                    total_cost = order_size * price + slip + comm
                    
                    running_cash -= total_cost
                    running_shares = order_size
                    
                    commissions[i] = comm
                    slippages[i] = slip
                    buy_count += 1
            
            # SELL: Going from long to flat
            elif target_pos == 0 and last_position == 1:
                if running_shares > 0:
                    order_size = running_shares
                    
                    # Symmetric proceeds calculation
                    # Net proceeds = shares * price - slippage - commission
                    slip = order_size * price * slippage_factor
                    comm = order_size * commission_rate
                    net_proceeds = order_size * price - slip - comm
                    
                    running_cash += net_proceeds
                    running_shares = 0
                    
                    commissions[i] = comm
                    slippages[i] = slip
                    sell_count += 1
        
        # Always value position at current market price
        # This provides accurate mark-to-market P&L, even on entry day
        if running_shares > 0:
            position_value = running_shares * price
        else:
            position_value = 0.0
        
        # Update tracking arrays
        cash[i] = running_cash
        shares[i] = running_shares
        portfolio_values[i] = running_cash + position_value
        last_position = target_pos
    
    return cash, shares, portfolio_values, commissions, slippages, buy_count, sell_count


class StrategyFactory:
    """
    Factory class for creating and managing trading strategies.
    Provides a consistent interface for signal generation across 
    different strategies.
    """
    
    _strategies: Dict[StrategyType, Callable] = {
        StrategyType.MOMENTUM: Momentum.generate_signals,
        StrategyType.MEAN_REVERSION: MeanReversion.generate_signals_with_zscore,
        StrategyType.MEAN_REVERSION_RSI: MeanReversionRSI.generate_signals,
        StrategyType.STAT_ARB: StatArb.generate_signals
    }
    
    @classmethod
    def get_strategy(cls, strategy_type: StrategyType) -> Callable:
        """
        Get the signal generation function for a given strategy type.
        
        Args:
            strategy_type: The type of strategy to retrieve
            
        Returns:
            Signal generation function
            
        Raises:
            ValueError: If strategy type is not registered
        """
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        return cls._strategies[strategy_type]
    
    @classmethod
    def register_strategy(cls, strategy_type: StrategyType, strategy_func: Callable) -> None:
        """
        Register a new strategy or override an existing one.
        
        Args:
            strategy_type: The type identifier for the strategy
            strategy_func: The signal generation function
        """
        cls._strategies[strategy_type] = strategy_func
    
    @classmethod
    def list_strategies(cls) -> list:
        """
        List all available strategy types.
        
        Returns:
            List of available StrategyType values
        """
        return list(cls._strategies.keys())

    
    @staticmethod
    def get_common_parameter_grid() -> Dict[str, list]:
        """Get default parameter grid for common parameters."""
        return {
            'tickers': ['NVDA'],
            'pairs': [('NVDA', 'AMD')],
            'trade_frequencies': [30, 45, 60, 90],
            'stop_loss_pcts': [0.01, 0.02, 0.03],
            'take_profit_pcts': [0.04, 0.05, 0.06],
            'max_drawdown_pcts': [0.10, 0.15, 0.20],
            'target_volatilities': [0.015, 0.02, 0.025]
        }

    @staticmethod
    def validate_config(config: Dict[str, Any], strategy_type: StrategyType) -> bool:
        """
        Validate configuration parameters for a given strategy.
        
        Args:
            config: Configuration dictionary
            strategy_type: The strategy type to validate for
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_common = ['AUM', 'trade_freq', 'stop_loss_pct', 'take_profit_pct']
        
        required_momentum = ['band_mult']
        required_mean_reversion = ['zscore_lookback', 'n_std_upper', 'n_std_lower']
        required_stat_arb = ['zscore_lookback', 'entry_threshold']
        
        # Check common parameters
        for param in required_common:
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Check strategy-specific parameters
        if strategy_type == StrategyType.MOMENTUM:
            for param in required_momentum:
                if param not in config:
                    raise ValueError(f"Missing required momentum parameter: {param}")
        elif strategy_type == StrategyType.MEAN_REVERSION:
            for param in required_mean_reversion:
                if param not in config:
                    raise ValueError(f"Missing required mean reversion parameter: {param}")
        elif strategy_type == StrategyType.STAT_ARB:
            for param in required_stat_arb:
                if param not in config:
                    raise ValueError(f"Missing required stat arb parameter: {param}")
        
        return True
