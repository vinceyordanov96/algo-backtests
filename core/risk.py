"""
Risk management module.

This module provides risk management functionality including:
    - Stop-loss monitoring
    - Take-profit monitoring  
    - Maximum drawdown tracking
    - Position-level risk checks

Classes:
    - RiskManager: High-level risk management interface

Functions:
    - check_stop_loss: Check if stop-loss is triggered
    - check_take_profit: Check if take-profit is triggered
    - check_max_drawdown: Check if max drawdown limit is breached
    - calculate_drawdown_series_numba: Calculate drawdown series efficiently
"""

import numpy as np
from numba import njit
from typing import Tuple, Optional
from dataclasses import dataclass


@njit(cache=True)
def calculate_drawdown_series_numba(aum: np.ndarray) -> np.ndarray:
    """
    JIT-compiled drawdown calculation for an AUM series.
    
    Args:
        aum: Array of AUM values
        
    Returns:
        Array of drawdown values (as negative decimals)
    """
    n = len(aum)
    drawdowns = np.zeros(n, dtype=np.float64)
    peak = aum[0]
    
    for i in range(n):
        if aum[i] > peak:
            peak = aum[i]
        drawdowns[i] = (aum[i] - peak) / peak if peak > 0 else 0.0
    
    return drawdowns


@njit(cache=True)
def check_stop_loss(
    current_price: float,
    entry_price: float,
    stop_loss_pct: float
) -> bool:
    """
    Check if stop-loss condition is triggered.
    
    Args:
        current_price: Current market price
        entry_price: Entry price of the position
        stop_loss_pct: Stop-loss threshold as decimal (e.g., 0.02 for 2%)
        
    Returns:
        True if stop-loss is triggered
    """
    if entry_price <= 0:
        return False
    
    pnl_pct = (current_price - entry_price) / entry_price
    return pnl_pct <= -stop_loss_pct


@njit(cache=True)
def check_take_profit(
    current_price: float,
    entry_price: float,
    take_profit_pct: float
) -> bool:
    """
    Check if take-profit condition is triggered.
    
    Args:
        current_price: Current market price
        entry_price: Entry price of the position
        take_profit_pct: Take-profit threshold as decimal (e.g., 0.04 for 4%)
        
    Returns:
        True if take-profit is triggered
    """
    if entry_price <= 0:
        return False
    
    pnl_pct = (current_price - entry_price) / entry_price
    return pnl_pct >= take_profit_pct


@njit(cache=True)
def check_max_drawdown(
    current_aum: float,
    peak_aum: float,
    max_drawdown_pct: float
) -> bool:
    """
    Check if maximum drawdown limit is breached.
    
    Args:
        current_aum: Current portfolio value
        peak_aum: Peak portfolio value
        max_drawdown_pct: Maximum allowed drawdown as decimal (e.g., 0.15 for 15%)
        
    Returns:
        True if max drawdown is breached
    """
    if peak_aum <= 0:
        return False
    
    current_drawdown = (current_aum - peak_aum) / peak_aum
    return abs(current_drawdown) > max_drawdown_pct


@njit(cache=True)
def calculate_position_pnl(
    current_price: float,
    entry_price: float,
    shares: int
) -> Tuple[float, float]:
    """
    Calculate position P&L.
    
    Args:
        current_price: Current market price
        entry_price: Entry price of the position
        shares: Number of shares held
        
    Returns:
        Tuple of (absolute_pnl, percentage_pnl)
    """
    if entry_price <= 0 or shares <= 0:
        return 0.0, 0.0
    
    absolute_pnl = (current_price - entry_price) * shares
    percentage_pnl = (current_price - entry_price) / entry_price
    
    return absolute_pnl, percentage_pnl


@dataclass
class RiskLimits:
    """Container for risk limit parameters."""
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_drawdown_pct: float = 0.15
    max_position_size: float = 1.0  # As fraction of portfolio
    max_leverage: float = 1.0


@dataclass 
class RiskState:
    """Container for current risk state."""
    peak_aum: float = 0.0
    current_drawdown: float = 0.0
    stop_losses_hit: int = 0
    take_profits_hit: int = 0
    max_drawdown_breaches: int = 0


class RiskManager:
    """
    High-level risk management interface.
    
    Manages portfolio-level and position-level risk including:
        - Stop-loss monitoring and execution
        - Take-profit monitoring and execution
        - Maximum drawdown tracking and breach handling
        - Position sizing based on risk parameters
    
    Example:
        risk_manager = RiskManager(
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            max_drawdown_pct=0.15
        )
        
        # Check position risk
        should_exit, reason = risk_manager.check_position_risk(
            current_price=105.0,
            entry_price=100.0
        )
        
        # Update portfolio state
        risk_manager.update_portfolio_state(current_aum=98000.0)
        
        # Check if trading should be halted
        if risk_manager.should_halt_trading():
            # Close all positions
            pass
    """
    
    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_drawdown_pct: float = 0.15,
        max_leverage: float = 1.0,
        initial_aum: float = 100000.0
    ):
        """
        Initialize the RiskManager.
        
        Args:
            stop_loss_pct: Stop-loss threshold as decimal
            take_profit_pct: Take-profit threshold as decimal
            max_drawdown_pct: Maximum drawdown threshold as decimal
            max_leverage: Maximum allowed leverage
            initial_aum: Initial portfolio value
        """
        self.limits = RiskLimits(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_drawdown_pct=max_drawdown_pct,
            max_leverage=max_leverage
        )
        
        self.state = RiskState(peak_aum=initial_aum)
        self._initial_aum = initial_aum
    
    def check_position_risk(
        self,
        current_price: float,
        entry_price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position should be exited due to risk limits.
        
        Args:
            current_price: Current market price
            entry_price: Entry price of the position
            
        Returns:
            Tuple of (should_exit, reason)
            reason is None if should_exit is False
        """
        # Check stop-loss
        if check_stop_loss(current_price, entry_price, self.limits.stop_loss_pct):
            self.state.stop_losses_hit += 1
            return True, "stop_loss"
        
        # Check take-profit
        if check_take_profit(current_price, entry_price, self.limits.take_profit_pct):
            self.state.take_profits_hit += 1
            return True, "take_profit"
        
        return False, None
    
    def update_portfolio_state(self, current_aum: float) -> None:
        """
        Update portfolio state with current AUM.
        
        Args:
            current_aum: Current portfolio value
        """
        # Update peak
        if current_aum > self.state.peak_aum:
            self.state.peak_aum = current_aum
        
        # Update drawdown
        if self.state.peak_aum > 0:
            self.state.current_drawdown = (current_aum - self.state.peak_aum) / self.state.peak_aum
    
    def check_max_drawdown_breach(self, current_aum: float) -> bool:
        """
        Check if maximum drawdown has been breached.
        
        Args:
            current_aum: Current portfolio value
            
        Returns:
            True if max drawdown is breached
        """
        is_breached = check_max_drawdown(
            current_aum, 
            self.state.peak_aum, 
            self.limits.max_drawdown_pct
        )
        
        if is_breached:
            self.state.max_drawdown_breaches += 1
        
        return is_breached
    
    def should_halt_trading(self) -> bool:
        """
        Check if trading should be halted due to risk limits.
        
        Returns:
            True if trading should be halted
        """
        return abs(self.state.current_drawdown) > self.limits.max_drawdown_pct
    
    def calculate_position_size(
        self,
        available_capital: float,
        price: float,
        volatility: float,
        target_volatility: float
    ) -> int:
        """
        Calculate position size based on volatility targeting.
        
        Args:
            available_capital: Available capital for trading
            price: Current price
            volatility: Asset volatility
            target_volatility: Target portfolio volatility
            
        Returns:
            Number of shares to trade
        """
        if volatility <= 0 or np.isnan(volatility):
            size_factor = self.limits.max_leverage
        else:
            size_factor = min(target_volatility / volatility, self.limits.max_leverage)
        
        position_value = available_capital * size_factor
        shares = int(position_value / price)
        
        return shares
    
    def get_risk_summary(self) -> dict:
        """
        Get summary of current risk state.
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            'peak_aum': self.state.peak_aum,
            'current_drawdown': self.state.current_drawdown,
            'stop_losses_hit': self.state.stop_losses_hit,
            'take_profits_hit': self.state.take_profits_hit,
            'max_drawdown_breaches': self.state.max_drawdown_breaches,
            'limits': {
                'stop_loss_pct': self.limits.stop_loss_pct,
                'take_profit_pct': self.limits.take_profit_pct,
                'max_drawdown_pct': self.limits.max_drawdown_pct,
                'max_leverage': self.limits.max_leverage
            }
        }
    
    def reset(self, initial_aum: Optional[float] = None) -> None:
        """
        Reset risk manager state.
        
        Args:
            initial_aum: New initial AUM (optional, uses original if not provided)
        """
        aum = initial_aum if initial_aum is not None else self._initial_aum
        self.state = RiskState(peak_aum=aum)
