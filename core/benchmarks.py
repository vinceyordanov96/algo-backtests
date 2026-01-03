import pandas as pd
import numpy as np
from typing import Dict

class BenchmarkMetrics:
    """
    Benchmark comparison metrics calculator.
    
    Calculates metrics for comparing strategy performance against a benchmark:
        - Beta
        - Alpha
        - Correlation
        - R-Squared
        - Information Ratio
        - Treynor Ratio
    """
    
    def __init__(self, benchmark_ticker: str = 'SPY'):
        """
        Initialize BenchmarkMetrics.
        
        Args:
            benchmark_ticker: Ticker symbol for the benchmark
        """
        self.benchmark_ticker = benchmark_ticker
    
    def _align_returns(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> tuple:
        """
        Align two return series by dropping rows where either has NaN.
        """
        combined = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(combined) < 2:
            return None, None
        return combined.iloc[:, 0], combined.iloc[:, 1]
    
    def calculate_beta(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate beta (sensitivity to benchmark movements).
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Beta coefficient
        """
        aligned_ret, aligned_bench = self._align_returns(returns, benchmark_returns)
        if aligned_ret is None:
            return np.nan
        cov_matrix = np.cov(aligned_ret, aligned_bench)
        return cov_matrix[0, 1] / cov_matrix[1, 1]
    
    def calculate_alpha(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate alpha (CAPM alpha - excess return beyond beta exposure).
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Annualized risk-free rate
            
        Returns:
            Daily alpha
        """
        aligned_ret, aligned_bench = self._align_returns(returns, benchmark_returns)
        if aligned_ret is None:
            return np.nan
        
        beta = self.calculate_beta(returns, benchmark_returns)
        excess_strategy = aligned_ret.mean() - risk_free_rate / 252
        excess_benchmark = aligned_bench.mean() - risk_free_rate / 252
        
        return excess_strategy - beta * excess_benchmark
    
    def calculate_correlation(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate correlation between strategy and benchmark.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Correlation coefficient
        """
        aligned_ret, aligned_bench = self._align_returns(returns, benchmark_returns)
        if aligned_ret is None:
            return np.nan
        return aligned_ret.corr(aligned_bench)
    
    def calculate_r_squared(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate R-squared (variance explained by benchmark).
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            R-squared value
        """
        corr = self.calculate_correlation(returns, benchmark_returns)
        return corr ** 2 if not np.isnan(corr) else np.nan
    
    def calculate_information_ratio(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Information Ratio (risk-adjusted excess return).
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        aligned_ret, aligned_bench = self._align_returns(returns, benchmark_returns)
        if aligned_ret is None:
            return np.nan
        
        active_returns = aligned_ret - aligned_bench
        tracking_error = active_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return np.nan
        
        return (active_returns.mean() * 252) / tracking_error
    
    def calculate_treynor_ratio(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Treynor Ratio (excess return per unit of systematic risk).
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Annualized risk-free rate
            
        Returns:
            Treynor ratio
        """
        aligned_ret, aligned_bench = self._align_returns(returns, benchmark_returns)
        if aligned_ret is None:
            return np.nan
        
        beta = self.calculate_beta(returns, benchmark_returns)
        if beta == 0 or np.isnan(beta):
            return np.nan
        
        excess_return = aligned_ret.mean() * 252 - risk_free_rate
        return excess_return / beta
    
    def calculate_all_metrics(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate all benchmark comparison metrics.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Annualized risk-free rate
            
        Returns:
            Dictionary with all benchmark metrics
        """
        return {
            'Beta': self.calculate_beta(returns, benchmark_returns),
            'Alpha (annualized)': self.calculate_alpha(returns, benchmark_returns, risk_free_rate) * 252,
            'Correlation': self.calculate_correlation(returns, benchmark_returns),
            'R-Squared': self.calculate_r_squared(returns, benchmark_returns),
            'Information Ratio': self.calculate_information_ratio(returns, benchmark_returns),
            'Treynor Ratio': self.calculate_treynor_ratio(returns, benchmark_returns, risk_free_rate)
        }
