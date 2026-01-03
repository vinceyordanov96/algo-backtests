# algo-backtests/backtesting/results.py
"""
Backtest results container and analysis.

This module provides:
    - BacktestResults: Container for backtest output data
    - BacktestStatistics: Calculated performance statistics
    - Plotting and visualization utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

from core.metrics import MetricsCalculator
from core.benchmarks import BenchmarkMetrics


@dataclass
class BacktestStatistics:
    """Container for calculated backtest statistics."""
    
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    benchmark_return: float = 0.0
    benchmark_annualized_return: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = np.nan
    sortino_ratio: float = np.nan
    calmar_ratio: float = np.nan
    benchmark_sharpe: float = np.nan
    benchmark_sortino: float = np.nan
    benchmark_calmar: float = np.nan
    
    # Risk metrics
    max_drawdown: float = 0.0
    benchmark_max_drawdown: float = 0.0
    annualized_volatility: float = 0.0
    benchmark_volatility: float = 0.0
    
    # Trade metrics
    win_rate: float = 0.0
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    stop_losses_hit: int = 0
    take_profits_hit: int = 0
    
    # Benchmark comparison
    beta: float = np.nan
    alpha: float = np.nan
    correlation: float = np.nan
    r_squared: float = np.nan
    information_ratio: float = np.nan
    treynor_ratio: float = np.nan
    
    # Risk-free rate
    avg_risk_free_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'Total Return (%)': f"{self.total_return * 100:.2f}%",
            'Benchmark Return (%)': f"{self.benchmark_return * 100:.2f}%",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}" if not np.isnan(self.sharpe_ratio) else "N/A",
            'Benchmark Sharpe': f"{self.benchmark_sharpe:.2f}" if not np.isnan(self.benchmark_sharpe) else "N/A",
            'Sortino Ratio': f"{self.sortino_ratio:.2f}" if not np.isnan(self.sortino_ratio) else "N/A",
            'Benchmark Sortino': f"{self.benchmark_sortino:.2f}" if not np.isnan(self.benchmark_sortino) else "N/A",
            'Calmar Ratio': f"{self.calmar_ratio:.2f}" if not np.isnan(self.calmar_ratio) else "N/A",
            'Benchmark Calmar': f"{self.benchmark_calmar:.2f}" if not np.isnan(self.benchmark_calmar) else "N/A",
            'Max Drawdown (%)': f"{self.max_drawdown * 100:.2f}%",
            'Benchmark Max DD (%)': f"{self.benchmark_max_drawdown * 100:.2f}%",
            'Annualized Vol (%)': f"{self.annualized_volatility * 100:.2f}%",
            'Benchmark Vol (%)': f"{self.benchmark_volatility * 100:.2f}%",
            'Win Rate (%)': f"{self.win_rate * 100:.2f}%",
            'Total Trades': str(self.total_trades),
            'Buy Trades': str(self.buy_trades),
            'Sell Trades': str(self.sell_trades),
            'Stop Losses Hit': str(self.stop_losses_hit),
            'Take Profits Hit': str(self.take_profits_hit),
            'Beta': f"{self.beta:.4f}" if not np.isnan(self.beta) else "N/A",
            'Alpha (annualized %)': f"{self.alpha * 100:.4f}%" if not np.isnan(self.alpha) else "N/A",
            'Correlation': f"{self.correlation:.4f}" if not np.isnan(self.correlation) else "N/A",
            'Information Ratio': f"{self.information_ratio:.4f}" if not np.isnan(self.information_ratio) else "N/A",
            'Avg Risk-Free Rate (%)': f"{self.avg_risk_free_rate * 100:.2f}%",
        }
    
    def __str__(self) -> str:
        """Pretty print statistics."""
        lines = []
        for key, value in self.to_dict().items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


@dataclass
class BacktestResults:
    """
    Container for backtest results with analysis methods.
    
    Attributes:
        strategy_df: DataFrame with daily strategy metrics
        ticker: Primary ticker symbol
        ticker_b: Secondary ticker (for pairs trading)
        strategy_type: Type of strategy that was run
        initial_aum: Starting portfolio value
        trade_stats: Dictionary with trade statistics
    """
    
    strategy_df: pd.DataFrame
    ticker: str
    strategy_type: str
    initial_aum: float = 100000.0
    ticker_b: Optional[str] = None
    trade_stats: Dict[str, int] = field(default_factory=dict)
    risk_free_rate_series: Optional[pd.Series] = None
    
    def __post_init__(self):
        """Initialize calculated fields."""
        self._statistics: Optional[BacktestStatistics] = None
        self._metrics_calculator = MetricsCalculator()
        self._benchmark_metrics = BenchmarkMetrics(self.ticker)
    
    @property
    def returns(self) -> pd.Series:
        """Get strategy returns series."""
        return self.strategy_df['ret'].dropna()
    
    @property
    def benchmark_returns(self) -> pd.Series:
        """Get benchmark returns series."""
        return self.strategy_df['ret_ticker'].dropna()
    
    @property
    def aum(self) -> pd.Series:
        """Get AUM series."""
        return self.strategy_df['AUM']
    
    @property
    def benchmark_aum(self) -> pd.Series:
        """Get benchmark AUM series."""
        if 'AUM_ticker' not in self.strategy_df.columns:
            # Handle NaN values properly - fill with 0 return (no change)
            ret_ticker = self.strategy_df['ret_ticker'].fillna(0)
            self.strategy_df['AUM_ticker'] = self.initial_aum * (1 + ret_ticker).cumprod()
        return self.strategy_df['AUM_ticker']
    
    def calculate_statistics(self) -> BacktestStatistics:
        """
        Calculate comprehensive performance statistics.
        
        Returns:
            BacktestStatistics object with all metrics
        """
        if self._statistics is not None:
            return self._statistics
        
        stats = BacktestStatistics()
        
        valid_returns = self.returns
        valid_benchmark = self.benchmark_returns
        
        # Calculate benchmark AUM if not present
        if 'AUM_ticker' not in self.strategy_df.columns:
            ret_ticker = self.strategy_df['ret_ticker'].fillna(0)
            self.strategy_df['AUM_ticker'] = self.initial_aum * (1 + ret_ticker).cumprod()
        
        # Return metrics
        stats.total_return = (self.aum.iloc[-1] - self.initial_aum) / self.initial_aum
        stats.benchmark_return = (self.benchmark_aum.iloc[-1] - self.initial_aum) / self.initial_aum
        
        num_years = len(valid_returns) / 252
        benchmark_years = len(valid_benchmark) / 252
        
        if num_years > 0:
            stats.annualized_return = (1 + stats.total_return) ** (1 / num_years) - 1
        if benchmark_years > 0:
            stats.benchmark_annualized_return = (1 + stats.benchmark_return) ** (1 / benchmark_years) - 1
        
        # Risk-adjusted metrics
        stats.sharpe_ratio = self._metrics_calculator.calculate_sharpe_ratio(
            valid_returns,
            risk_free_rate_series=self.risk_free_rate_series
        )
        stats.sortino_ratio = self._metrics_calculator.calculate_sortino_ratio(
            valid_returns,
            risk_free_rate_series=self.risk_free_rate_series
        )
        stats.calmar_ratio = self._metrics_calculator.calculate_calmar_ratio(
            valid_returns, self.aum
        )
        
        stats.benchmark_sharpe = self._metrics_calculator.calculate_sharpe_ratio(
            valid_benchmark,
            risk_free_rate_series=self.risk_free_rate_series
        )
        stats.benchmark_sortino = self._metrics_calculator.calculate_sortino_ratio(
            valid_benchmark,
            risk_free_rate_series=self.risk_free_rate_series
        )
        stats.benchmark_calmar = self._metrics_calculator.calculate_calmar_ratio(
            valid_benchmark, self.benchmark_aum
        )
        
        # Risk metrics
        self.strategy_df['cummax'] = self.aum.cummax()
        self.strategy_df['drawdown_pct'] = (self.aum - self.strategy_df['cummax']) / self.strategy_df['cummax']
        stats.max_drawdown = self.strategy_df['drawdown_pct'].min()
        
        self.strategy_df['ticker_cummax'] = self.benchmark_aum.cummax()
        self.strategy_df['ticker_drawdown_pct'] = (self.benchmark_aum - self.strategy_df['ticker_cummax']) / self.strategy_df['ticker_cummax']
        stats.benchmark_max_drawdown = self.strategy_df['ticker_drawdown_pct'].min()
        
        stats.annualized_volatility = valid_returns.std() * np.sqrt(252)
        stats.benchmark_volatility = valid_benchmark.std() * np.sqrt(252)
        
        # Win rate
        stats.win_rate = self._metrics_calculator.calculate_win_rate(valid_returns)
        
        # Trade stats
        stats.total_trades = self.trade_stats.get('total_trades', 0)
        stats.buy_trades = self.trade_stats.get('buy_trades', 0)
        stats.sell_trades = self.trade_stats.get('sell_trades', 0)
        stats.stop_losses_hit = self.trade_stats.get('stop_losses', 0)
        stats.take_profits_hit = self.trade_stats.get('take_profits', 0)
        
        # Benchmark comparison metrics
        rf_rate = self.risk_free_rate_series.mean() if self.risk_free_rate_series is not None else 0.0
        benchmark_metrics = self._benchmark_metrics.calculate_all_metrics(
            valid_returns, valid_benchmark, rf_rate
        )
        
        stats.beta = benchmark_metrics['Beta']
        stats.alpha = benchmark_metrics['Alpha (annualized)']
        stats.correlation = benchmark_metrics['Correlation']
        stats.r_squared = benchmark_metrics['R-Squared']
        stats.information_ratio = benchmark_metrics['Information Ratio']
        stats.treynor_ratio = benchmark_metrics['Treynor Ratio']
        
        # Risk-free rate
        if self.risk_free_rate_series is not None and not self.risk_free_rate_series.empty:
            stats.avg_risk_free_rate = self.risk_free_rate_series.mean()
        
        self._statistics = stats
        return stats
    
    def plot(
        self,
        figsize: Tuple[int, int] = (12, 6),
        show: bool = True,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot backtest results.
        
        Args:
            figsize: Figure size
            show: Whether to display the plot
            save_path: Path to save the figure (optional)
            
        Returns:
            Tuple of (figure, axes)
        """
        # Calculate statistics if not already done
        stats = self.calculate_statistics()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Strategy name for title
        strategy_name = self.strategy_type.replace('_', ' ').title()
        
        # Title symbol
        if self.ticker_b:
            title_symbol = f"{self.ticker}/{self.ticker_b}"
        else:
            title_symbol = self.ticker
        
        # Plot AUM
        ax.plot(
            self.strategy_df.index, 
            self.aum, 
            label=f'{strategy_name} Strategy', 
            linewidth=2, 
            color='k'
        )
        ax.plot(
            self.strategy_df.index, 
            self.benchmark_aum, 
            label=f'{self.ticker} Buy & Hold', 
            linewidth=1, 
            color='r'
        )
        
        # Formatting
        ax.grid(True, linestyle=':')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.xticks(rotation=90)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.set_ylabel('AUM ($)')
        ax.set_xlabel('Date')
        ax.legend()
        ax.set_title(f'{title_symbol} {strategy_name} Strategy Backtest')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig, ax
    
    def plot_drawdown(
        self,
        figsize: Tuple[int, int] = (12, 4),
        show: bool = True,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot drawdown chart.
        
        Args:
            figsize: Figure size
            show: Whether to display the plot
            save_path: Path to save the figure (optional)
            
        Returns:
            Tuple of (figure, axes)
        """
        # Ensure drawdown is calculated
        if 'drawdown_pct' not in self.strategy_df.columns:
            self.calculate_statistics()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.fill_between(
            self.strategy_df.index,
            self.strategy_df['drawdown_pct'] * 100,
            0,
            color='red',
            alpha=0.3,
            label='Strategy Drawdown'
        )
        ax.fill_between(
            self.strategy_df.index,
            self.strategy_df['ticker_drawdown_pct'] * 100,
            0,
            color='blue',
            alpha=0.2,
            label='Benchmark Drawdown'
        )
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Date')
        ax.legend()
        ax.set_title(f'{self.ticker} Drawdown Analysis')
        ax.grid(True, linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig, ax
    
    def to_csv(self, path: str) -> None:
        """
        Save results to CSV file.
        
        Args:
            path: File path for CSV output
        """
        self.strategy_df.to_csv(path)
    

    def save_outputs(
        self,
        output_dir: str = 'outputs/strategies',
        config: Optional[Dict[str, Any]] = None,
        signals_df: Optional[pd.DataFrame] = None,
        show_plot: bool = False
    ) -> Dict[str, str]:
        """
        Save all outputs for this backtest result.
        
        Args:
            output_dir: Base directory for strategy outputs
            config: Strategy configuration used
            signals_df: Optional signals DataFrame (will be derived if not provided)
            show_plot: Whether to display the plot
            
        Returns:
            Dictionary mapping output type to file path
        """
        from outputs import StrategyOutputManager
        
        manager = StrategyOutputManager(base_dir=output_dir)
        
        # Derive signals if not provided
        if signals_df is None:
            signals_df = pd.DataFrame(index=self.strategy_df.index)
            signals_df['position'] = self.strategy_df['position']
            signals_df['position_change'] = self.strategy_df['position'].diff().fillna(0)
            signals_df['signal'] = 0
            signals_df.loc[signals_df['position_change'] > 0, 'signal'] = 1
            signals_df.loc[signals_df['position_change'] < 0, 'signal'] = -1
        
        # Create portfolio DataFrame
        portfolio_df = self.strategy_df.copy()
        portfolio_df['direction'] = 'WAIT'
        portfolio_df.loc[self.strategy_df['position'].diff() > 0, 'direction'] = 'BUY'
        portfolio_df.loc[self.strategy_df['position'].diff() < 0, 'direction'] = 'SELL'
        portfolio_df['cumulative_returns'] = (1 + portfolio_df['ret']).cumprod() - 1
        portfolio_df['PnL'] = portfolio_df['AUM'] - self.initial_aum
        
        # Get statistics
        stats = self.calculate_statistics()
        stats_dict = {
            'total_return': stats.total_return,
            'sharpe_ratio': stats.sharpe_ratio,
            'sortino_ratio': stats.sortino_ratio,
            'calmar_ratio': stats.calmar_ratio,
            'max_drawdown': stats.max_drawdown,
            'win_rate': stats.win_rate,
            'total_trades': stats.total_trades,
        }
        
        # Save outputs
        paths = manager.save_all_best_strategy_outputs(
            strategy=self.strategy_type,
            ticker=self.ticker,
            backtest_results=self,
            signals_df=signals_df,
            portfolio_df=portfolio_df,
            config=config or {},
            show_plot=show_plot
        )
        
        return paths
    

    def summary(self) -> str:
        """
        Get a summary string of the backtest results.
        
        Returns:
            Formatted summary string
        """
        stats = self.calculate_statistics()
        
        lines = [
            "=" * 60,
            f"BACKTEST RESULTS - {self.ticker} ({self.strategy_type.upper()})",
            "=" * 60,
            "",
            str(stats),
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)
