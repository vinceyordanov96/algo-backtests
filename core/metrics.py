

"""
Performance metrics module.

This module provides performance metric calculations including:
    - Sharpe ratio
    - Sortino ratio
    - Calmar ratio
    - Value at Risk (VaR)
    - Volatility calculations
    - Benchmark comparison metrics

Classes:
    - MetricsCalculator: High-level metrics calculation interface
    - BenchmarkMetrics: Benchmark comparison calculations

Functions:
    - calculate_var_numba: Numba-optimized VaR calculation
    - calculate_rolling_sharpe_numba: Numba-optimized rolling Sharpe
    - calculate_sortino_components_numba: Numba-optimized Sortino components
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Optional, Union, Dict, Any
from numpy.lib.stride_tricks import sliding_window_view
from .benchmarks import BenchmarkMetrics


@njit(cache=True)
def calculate_var_numba(returns: np.ndarray, confidence_level: float) -> float:
    """
    JIT-compiled Value at Risk calculation using historical simulation.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95)
        
    Returns:
        VaR value (positive number representing potential loss)
    """
    if len(returns) < 2:
        return np.nan
    
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Calculate the index for the percentile
    percentile_idx = int((1 - confidence_level) * len(sorted_returns))
    
    return -sorted_returns[percentile_idx]


@njit(cache=True)
def calculate_rolling_sharpe_numba(
    returns: np.ndarray,
    lookback: int
) -> float:
    """
    JIT-compiled rolling Sharpe ratio calculation (not annualized).
    
    Args:
        returns: Array of returns
        lookback: Lookback period
        
    Returns:
        Sharpe ratio (not annualized)
    """
    if len(returns) < lookback:
        return np.nan
    
    recent = returns[-lookback:]
    mean_ret = np.mean(recent)
    std_ret = np.std(recent)
    
    if std_ret <= 0:
        return 0.0
    
    return mean_ret / std_ret


@njit(cache=True)
def calculate_sortino_components_numba(
    excess_returns: np.ndarray
) -> tuple:
    """
    JIT-compiled Sortino ratio component calculation.
    
    Args:
        excess_returns: Array of excess returns (returns - risk-free rate)
        
    Returns:
        Tuple of (mean_excess, downside_deviation)
    """
    mean_excess = np.mean(excess_returns)
    
    # Calculate downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        return mean_excess, np.nan
    
    downside_var = np.mean(negative_returns ** 2)
    downside_dev = np.sqrt(downside_var)
    
    return mean_excess, downside_dev


@njit(cache=True)
def calculate_max_drawdown_numba(aum: np.ndarray) -> float:
    """
    JIT-compiled maximum drawdown calculation.
    
    Args:
        aum: Array of portfolio values
        
    Returns:
        Maximum drawdown as a negative decimal
    """
    n = len(aum)
    if n < 2:
        return 0.0
    
    peak = aum[0]
    max_dd = 0.0
    
    for i in range(n):
        if aum[i] > peak:
            peak = aum[i]
        
        if peak > 0:
            dd = (aum[i] - peak) / peak
            if dd < max_dd:
                max_dd = dd
    
    return max_dd


class MetricsCalculator:
    """
    High-level performance metrics calculator.
    
    Provides a unified interface for calculating various performance
    metrics with proper handling of risk-free rates and annualization.
    
    Example:
        calculator = MetricsCalculator()
        
        # Calculate Sharpe ratio
        sharpe = calculator.calculate_sharpe_ratio(
            returns=strategy_returns,
            risk_free_rate_series=rf_series,
            annualize=True
        )
        
        # Calculate all metrics
        metrics = calculator.calculate_all_metrics(
            returns=strategy_returns,
            aum_series=aum,
            risk_free_rate_series=rf_series
        )
    """
    
    def __init__(self, trading_days_per_year: int = 252):
        """
        Initialize the MetricsCalculator.
        
        Args:
            trading_days_per_year: Number of trading days per year for annualization
        """
        self.trading_days = trading_days_per_year
    
    def get_daily_risk_free_rate(
        self, 
        risk_free_rate_series: pd.Series, 
        date: pd.Timestamp
    ) -> float:
        """
        Get the daily risk-free rate for a given date.
        
        Args:
            risk_free_rate_series: Series with date index and annualized rates in decimal form
            date: The date to get the rate for
            
        Returns:
            Daily risk-free rate (decimal)
        """
        if risk_free_rate_series is None or risk_free_rate_series.empty:
            return 0.0
        
        date_normalized = pd.Timestamp(date).normalize()
        
        # Try exact match first
        if date_normalized in risk_free_rate_series.index:
            annual_rate = risk_free_rate_series.loc[date_normalized]
        else:
            # Find most recent available rate
            available_dates = risk_free_rate_series.index[risk_free_rate_series.index <= date_normalized]
            if len(available_dates) == 0:
                return 0.0
            annual_rate = risk_free_rate_series.loc[available_dates[-1]]
        
        if pd.isna(annual_rate):
            return 0.0
        
        return annual_rate / self.trading_days
    
    def subtract_risk_free_rate(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = None,
        risk_free_rate_series: pd.Series = None
    ) -> pd.Series:
        """
        Subtract the daily risk-free rate from the returns.
        
        Args:
            returns: Series of daily returns
            risk_free_rate: Single annualized risk-free rate (decimal)
            risk_free_rate_series: Series of annualized rates indexed by date (decimal)
            
        Returns:
            Series of excess returns
        """
        if len(returns) < 2:
            return returns
        
        if risk_free_rate_series is not None and not risk_free_rate_series.empty:
            # Vectorized approach using reindex
            rf_aligned = risk_free_rate_series.reindex(returns.index, method='ffill') / self.trading_days
            rf_aligned = rf_aligned.fillna(0)
            return returns - rf_aligned
        elif risk_free_rate is not None:
            daily_rf = risk_free_rate / self.trading_days
            return returns - daily_rf
        else:
            return returns
    
    def calculate_volatility(
        self, 
        returns: Union[pd.Series, np.ndarray], 
        rolling: bool = False, 
        lookback_period: int = 20,
        annualize: bool = False
    ) -> Union[float, pd.Series, np.ndarray]:
        """
        Calculate volatility for a returns series.
        
        Args:
            returns: Series or array of returns
            rolling: If True, calculate rolling volatility
            lookback_period: Window for rolling calculation
            annualize: If True, annualize the volatility
            
        Returns:
            Volatility value, Series, or array
        """
        factor = np.sqrt(self.trading_days) if annualize else 1
        
        if isinstance(returns, pd.Series):
            if rolling:
                return returns.rolling(lookback_period).std() * factor
            return returns.std() * factor
        else:
            # NumPy array
            if rolling:
                if len(returns) >= lookback_period:
                    windows = sliding_window_view(returns, lookback_period)
                    rolling_std = np.std(windows, axis=1) * factor
                    result = np.full(len(returns), np.nan)
                    result[lookback_period-1:] = rolling_std
                    return result
                return np.full(len(returns), np.nan)
            return np.nanstd(returns) * factor
    
    def calculate_var(
        self, 
        returns: Union[pd.Series, np.ndarray], 
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk using historical simulation.
        
        Args:
            returns: Returns data
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            VaR value
        """
        if isinstance(returns, pd.Series):
            returns_arr = returns.dropna().values.astype(np.float64)
        else:
            returns_arr = returns.astype(np.float64)
        
        return calculate_var_numba(returns_arr, confidence_level)
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        lookback_period: int = 20,
        risk_free_rate: float = None,
        risk_free_rate_series: pd.Series = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate the Sharpe ratio.
        
        Args:
            returns: Series of daily returns
            lookback_period: Minimum observations required
            risk_free_rate: Single annualized risk-free rate (decimal)
            risk_free_rate_series: Series of annualized rates indexed by date (decimal)
            annualize: Whether to annualize the ratio
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < lookback_period:
            return np.nan
        
        # Calculate excess returns
        excess_returns = self.subtract_risk_free_rate(
            returns, 
            risk_free_rate=risk_free_rate,
            risk_free_rate_series=risk_free_rate_series
        )
        
        excess_returns_clean = excess_returns.dropna()
        
        if len(excess_returns_clean) < lookback_period:
            return np.nan
        
        ret_std = excess_returns_clean.std()
        
        if ret_std <= 0:
            return 0.0
        
        sharpe = excess_returns_clean.mean() / ret_std
        
        if annualize:
            sharpe *= np.sqrt(self.trading_days)
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        lookback_period: int = 20,
        risk_free_rate: float = None,
        risk_free_rate_series: pd.Series = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate the Sortino ratio.
        
        Args:
            returns: Series of daily returns
            lookback_period: Minimum observations required
            risk_free_rate: Single annualized risk-free rate (decimal)
            risk_free_rate_series: Series of annualized rates indexed by date (decimal)
            annualize: Whether to annualize the ratio
            
        Returns:
            Sortino ratio
        """
        if len(returns) < lookback_period:
            return np.nan
        
        # Calculate excess returns
        excess_returns = self.subtract_risk_free_rate(
            returns,
            risk_free_rate=risk_free_rate,
            risk_free_rate_series=risk_free_rate_series
        )
        
        excess_returns_clean = excess_returns.dropna().values.astype(np.float64)
        
        if len(excess_returns_clean) < lookback_period:
            return np.nan
        
        mean_excess, downside_dev = calculate_sortino_components_numba(excess_returns_clean)
        
        if np.isnan(downside_dev) or downside_dev <= 0:
            return np.nan
        
        sortino = mean_excess / downside_dev
        
        if annualize:
            sortino *= np.sqrt(self.trading_days)
        
        return sortino
    
    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        aum_series: pd.Series = None
    ) -> float:
        """
        Calculate the Calmar ratio (annualized return / max drawdown).
        
        Args:
            returns: Series of daily returns
            aum_series: Series of portfolio values (optional)
            
        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return np.nan
        
        # Calculate annualized return
        total_return = (1 + returns).prod() - 1
        num_years = len(returns) / self.trading_days
        
        if num_years <= 0:
            return np.nan
        
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        
        # Calculate max drawdown
        if aum_series is not None:
            max_dd = calculate_max_drawdown_numba(aum_series.values.astype(np.float64))
        else:
            # Reconstruct AUM from returns
            aum = (1 + returns).cumprod()
            max_dd = calculate_max_drawdown_numba(aum.values.astype(np.float64))
        
        if max_dd == 0 or np.isnan(max_dd):
            return np.nan
        
        return annualized_return / abs(max_dd)
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """
        Calculate win rate (percentage of positive return days).
        
        Args:
            returns: Series of daily returns
            
        Returns:
            Win rate as decimal
        """
        valid_returns = returns.dropna()
        if len(valid_returns) == 0:
            return 0.0
        
        winning_days = (valid_returns > 0).sum()
        return winning_days / len(valid_returns)
    
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        aum_series: pd.Series = None,
        risk_free_rate: float = None,
        risk_free_rate_series: pd.Series = None,
        lookback_period: int = 20
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Args:
            returns: Series of daily returns
            aum_series: Series of portfolio values (optional)
            risk_free_rate: Single annualized risk-free rate (decimal)
            risk_free_rate_series: Series of annualized rates indexed by date (decimal)
            lookback_period: Minimum observations for ratio calculations
            
        Returns:
            Dictionary with all metrics
        """
        valid_returns = returns.dropna()
        
        return {
            'total_return': (1 + valid_returns).prod() - 1,
            'annualized_return': ((1 + valid_returns).prod() ** (self.trading_days / len(valid_returns))) - 1 if len(valid_returns) > 0 else np.nan,
            'annualized_volatility': self.calculate_volatility(valid_returns, annualize=True),
            'sharpe_ratio': self.calculate_sharpe_ratio(
                returns, lookback_period, risk_free_rate, risk_free_rate_series
            ),
            'sortino_ratio': self.calculate_sortino_ratio(
                returns, lookback_period, risk_free_rate, risk_free_rate_series
            ),
            'calmar_ratio': self.calculate_calmar_ratio(returns, aum_series),
            'max_drawdown': calculate_max_drawdown_numba(
                aum_series.values.astype(np.float64)
            ) if aum_series is not None else np.nan,
            'win_rate': self.calculate_win_rate(returns),
            'var_95': self.calculate_var(returns, 0.95),
            'var_99': self.calculate_var(returns, 0.99),
        }
