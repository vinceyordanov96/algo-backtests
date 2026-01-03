"""
Strategy output management module.

This module provides:
    - StrategyOutputManager: Central class for saving strategy outputs
    - Consistent naming conventions across all output types
    - Support for loading and listing saved outputs

Output Types:
    - Simulations: Parameter sweep results (all tested combinations)
    - Signals: Generated signals from best performing strategy
    - Results: Trade-by-trade portfolio updates
    - Plots: Benchmark comparison visualizations

Directory Structure:
    outputs/strategies/{strategy}/
    ├── simulations/{ticker}/
    │   └── {ticker}_{strategy}_simulation_YYYYMMDD_HHMMSS.csv
    ├── signals/{ticker}/
    │   └── {ticker}_{strategy}_signals_YYYYMMDD_HHMMSS.csv
    ├── results/{ticker}/
    │   └── {ticker}_{strategy}_results_YYYYMMDD_HHMMSS.csv
    └── plots/{ticker}/
        └── {ticker}_{strategy}_plot_YYYYMMDD_HHMMSS.png

Example:
    from outputs import StrategyOutputManager
    
    manager = StrategyOutputManager()
    
    # After running simulation
    manager.save_simulation_results('momentum', 'NVDA', results_df)
    
    # After running best strategy backtest
    manager.save_signals('momentum', 'NVDA', signals_df, best_config)
    manager.save_portfolio_results('momentum', 'NVDA', portfolio_df, best_config)
    manager.save_benchmark_plot('momentum', 'NVDA', backtest_results)
    
    # Load latest outputs
    sim_df = manager.load_simulation_results('momentum', 'NVDA')
    signals_df = manager.load_signals('momentum', 'NVDA')
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger(__name__)


class StrategyOutputManager:
    """
    Central manager for strategy output persistence and retrieval.
    
    Handles saving and loading of:
        - Simulation results (parameter sweeps)
        - Strategy signals
        - Portfolio results (trade-by-trade)
        - Benchmark comparison plots
    
    Attributes:
        base_dir: Root directory for all strategy outputs
    
    Example:
        manager = StrategyOutputManager()
        
        # Save outputs after simulation
        manager.save_simulation_results('momentum', 'NVDA', results_df)
        
        # Run best strategy and save detailed outputs
        manager.save_signals('momentum', 'NVDA', signals_df, config)
        manager.save_portfolio_results('momentum', 'NVDA', portfolio_df, config)
        manager.save_benchmark_plot('momentum', 'NVDA', backtest_results)
    """
    
    # Output subdirectory names
    SIMULATIONS_DIR = 'simulations'
    SIGNALS_DIR = 'signals'
    RESULTS_DIR = 'results'
    PLOTS_DIR = 'plots'
    
    def __init__(self, base_dir: str = 'outputs/strategies'):
        """
        Initialize the StrategyOutputManager.
        
        Args:
            base_dir: Root directory for all strategy outputs
        """
        self.base_dir = Path(base_dir)
        self._ensure_base_directory()
    

    def _ensure_base_directory(self) -> None:
        """Create base directory if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    

    def _get_strategy_dir(self, strategy: str) -> Path:
        """
        Get the directory for a specific strategy.
        
        Args:
            strategy: Strategy name (e.g., 'momentum', 'mean_reversion')
            
        Returns:
            Path to strategy directory
        """
        strategy_dir = self.base_dir / strategy.lower()
        strategy_dir.mkdir(parents=True, exist_ok=True)
        return strategy_dir
    

    def _get_output_dir(self, strategy: str, output_type: str, ticker: str) -> Path:
        """
        Get the directory for a specific output type and ticker.
        
        Args:
            strategy: Strategy name
            output_type: Output type (simulations, signals, results, plots)
            ticker: Stock ticker symbol
            
        Returns:
            Path to output directory
        """
        output_dir = self._get_strategy_dir(strategy) / output_type / ticker.upper()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    

    def _generate_filename(
        self,
        ticker: str,
        strategy: str,
        output_type: str,
        extension: str = 'csv'
    ) -> str:
        """
        Generate a filename with timestamp.
        
        Args:
            ticker: Stock ticker symbol
            strategy: Strategy name
            output_type: Output type identifier
            extension: File extension
            
        Returns:
            Filename string
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{ticker.upper()}_{strategy.lower()}_{output_type}_{timestamp}.{extension}"
    

    def _parse_filename_timestamp(self, filename: str) -> Optional[datetime]:
        """
        Parse timestamp from filename.
        
        Args:
            filename: Filename to parse
            
        Returns:
            datetime or None if parsing fails
        """
        try:
            # Extract timestamp portion (YYYYMMDD_HHMMSS)
            parts = filename.replace('.csv', '').replace('.png', '').split('_')
            if len(parts) >= 4:
                date_str = parts[-2]
                time_str = parts[-1]
                return datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
        except (ValueError, IndexError):
            pass
        return None
    

    def _get_latest_file(
        self,
        strategy: str,
        output_type: str,
        ticker: str,
        extension: str = 'csv'
    ) -> Optional[Path]:
        """
        Get the most recent file for a strategy/output_type/ticker.
        
        Args:
            strategy: Strategy name
            output_type: Output type (simulations, signals, results, plots)
            ticker: Stock ticker symbol
            extension: File extension to look for
            
        Returns:
            Path to latest file or None if no files exist
        """
        output_dir = self._get_output_dir(strategy, output_type, ticker)
        pattern = f"{ticker.upper()}_{strategy.lower()}_*.{extension}"
        files = list(output_dir.glob(pattern))
        
        if not files:
            return None
        
        # Sort by timestamp in filename (newest first)
        files_with_times = []
        for f in files:
            ts = self._parse_filename_timestamp(f.name)
            if ts:
                files_with_times.append((f, ts))
        
        if not files_with_times:
            # Fallback to modification time
            return max(files, key=lambda x: x.stat().st_mtime)
        
        files_with_times.sort(key=lambda x: x[1], reverse=True)
        return files_with_times[0][0]
    

    def _get_all_files(
        self,
        strategy: str,
        output_type: str,
        ticker: str,
        extension: str = 'csv'
    ) -> List[Path]:
        """
        Get all files for a strategy/output_type/ticker, sorted newest first.
        
        Args:
            strategy: Strategy name
            output_type: Output type
            ticker: Stock ticker symbol
            extension: File extension
            
        Returns:
            List of file paths sorted by timestamp (newest first)
        """
        output_dir = self._get_output_dir(strategy, output_type, ticker)
        pattern = f"{ticker.upper()}_{strategy.lower()}_*.{extension}"
        files = list(output_dir.glob(pattern))
        
        files_with_times = []
        for f in files:
            ts = self._parse_filename_timestamp(f.name)
            if ts:
                files_with_times.append((f, ts))
        
        files_with_times.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in files_with_times]
    
    
    def save_simulation_results(
        self,
        strategy: str,
        ticker: str,
        results_df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save simulation results (parameter sweep) to CSV.
        
        Args:
            strategy: Strategy name (e.g., 'momentum', 'mean_reversion')
            ticker: Stock ticker symbol
            results_df: DataFrame with simulation results for all parameter combinations
            metadata: Optional metadata to include (not saved, for logging)
            
        Returns:
            Path to saved file
        """
        output_dir = self._get_output_dir(strategy, self.SIMULATIONS_DIR, ticker)
        filename = self._generate_filename(ticker, strategy, 'simulation', 'csv')
        filepath = output_dir / filename
        
        results_df.to_csv(filepath, index=False)
        
        logger.info(f"Saved simulation results to: {output_dir}")
        #logger.info(f"  - {len(results_df)} parameter combinations")
        
        # if metadata:
        #     logger.info(f"  - Metadata: {metadata}")
        
        return str(filepath)
    

    def load_simulation_results(
        self,
        strategy: str,
        ticker: str,
        version: str = 'latest'
    ) -> Optional[pd.DataFrame]:
        """
        Load simulation results for a strategy/ticker.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            version: 'latest' for most recent, or specific filename
            
        Returns:
            DataFrame or None if not found
        """
        if version == 'latest':
            filepath = self._get_latest_file(strategy, self.SIMULATIONS_DIR, ticker)
        else:
            output_dir = self._get_output_dir(strategy, self.SIMULATIONS_DIR, ticker)
            filepath = output_dir / version
        
        if filepath is None or not filepath.exists():
            logger.warning(f"No simulation results found for {strategy}/{ticker}")
            return None
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} simulation results for {strategy}/{ticker}")
        return df
    

    def get_best_config_from_simulation(
        self,
        strategy: str,
        ticker: str,
        metric: str = 'Sharpe Ratio (Strategy)',
        version: str = 'latest'
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best performing configuration from simulation results.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            metric: Metric to optimize (default: Sharpe Ratio)
            version: 'latest' or specific filename
            
        Returns:
            Dictionary with best configuration parameters or None
        """
        df = self.load_simulation_results(strategy, ticker, version)
        
        if df is None or df.empty:
            return None
        
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in results")
            return None
        
        # Filter for this ticker
        ticker_df = df[df['Ticker'] == ticker.upper()]
        if ticker_df.empty:
            ticker_df = df  # Use all if ticker filter fails
        
        # Get best row
        best_idx = ticker_df[metric].idxmax()
        best_row = df.loc[best_idx]
        
        return best_row.to_dict()
    
    
    def save_signals(
        self,
        strategy: str,
        ticker: str,
        signals_df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save strategy signals to CSV.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            signals_df: DataFrame with signals (index=datetime, columns include 'signal')
            config: Strategy configuration used to generate signals
            
        Returns:
            Path to saved file
        """
        output_dir = self._get_output_dir(strategy, self.SIGNALS_DIR, ticker)
        filename = self._generate_filename(ticker, strategy, 'signals', 'csv')
        filepath = output_dir / filename
        
        # Add config info as header comments or separate metadata
        signals_df.to_csv(filepath)
        
        logger.info(f"Saved signals to: {output_dir}")
        #logger.info(f"  - {len(signals_df)} signal records")
        
        # Optionally save config alongside
        if config:
            config_path = filepath.with_suffix('.json')
            import json
            with open(config_path, 'w') as f:
                # Convert numpy types for JSON serialization
                clean_config = {}
                for k, v in config.items():
                    if isinstance(v, (np.integer, np.floating)):
                        clean_config[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        clean_config[k] = v.tolist()
                    else:
                        try:
                            json.dumps(v)
                            clean_config[k] = v
                        except (TypeError, ValueError):
                            clean_config[k] = str(v)
                json.dump(clean_config, f, indent=2, default=str)
            #logger.info(f"  - Config saved to: {config_path}")
        
        return str(filepath)
    

    def load_signals(
        self,
        strategy: str,
        ticker: str,
        version: str = 'latest'
    ) -> Optional[pd.DataFrame]:
        """
        Load signals for a strategy/ticker.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            version: 'latest' or specific filename
            
        Returns:
            DataFrame or None if not found
        """
        if version == 'latest':
            filepath = self._get_latest_file(strategy, self.SIGNALS_DIR, ticker)
        else:
            output_dir = self._get_output_dir(strategy, self.SIGNALS_DIR, ticker)
            filepath = output_dir / version
        
        if filepath is None or not filepath.exists():
            logger.warning(f"No signals found for {strategy}/{ticker}")
            return None
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} signal records for {strategy}/{ticker}")
        return df
    

    def save_portfolio_results(
        self,
        strategy: str,
        ticker: str,
        portfolio_df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        statistics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save trade-by-trade portfolio results to CSV.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            portfolio_df: DataFrame with portfolio updates (positions, cash, values, etc.)
            config: Strategy configuration used
            statistics: Summary statistics to include
            
        Returns:
            Path to saved file
        """
        output_dir = self._get_output_dir(strategy, self.RESULTS_DIR, ticker)
        filename = self._generate_filename(ticker, strategy, 'results', 'csv')
        filepath = output_dir / filename
        
        portfolio_df.to_csv(filepath)
        
        logger.info(f"Saved portfolio results to: {output_dir}")
        #logger.info(f"  - {len(portfolio_df)} records")
        
        # Save config and statistics
        if config or statistics:
            import json
            meta_path = filepath.with_suffix('.json')
            meta = {}
            
            if config:
                # Clean config for JSON
                clean_config = {}
                for k, v in config.items():
                    if isinstance(v, (np.integer, np.floating)):
                        clean_config[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        clean_config[k] = v.tolist()
                    else:
                        try:
                            json.dumps(v)
                            clean_config[k] = v
                        except (TypeError, ValueError):
                            clean_config[k] = str(v)
                meta['config'] = clean_config
            
            if statistics:
                # Clean statistics for JSON
                clean_stats = {}
                for k, v in statistics.items():
                    if isinstance(v, (np.integer, np.floating)):
                        clean_stats[k] = float(v) if not np.isnan(v) else None
                    elif isinstance(v, np.ndarray):
                        clean_stats[k] = v.tolist()
                    else:
                        clean_stats[k] = v
                meta['statistics'] = clean_stats
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2, default=str)
            #logger.info(f"  - Metadata saved to: {meta_path}")
        
        return str(filepath)
    

    def load_portfolio_results(
        self,
        strategy: str,
        ticker: str,
        version: str = 'latest'
    ) -> Optional[pd.DataFrame]:
        """
        Load portfolio results for a strategy/ticker.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            version: 'latest' or specific filename
            
        Returns:
            DataFrame or None if not found
        """
        if version == 'latest':
            filepath = self._get_latest_file(strategy, self.RESULTS_DIR, ticker)
        else:
            output_dir = self._get_output_dir(strategy, self.RESULTS_DIR, ticker)
            filepath = output_dir / version
        
        if filepath is None or not filepath.exists():
            logger.warning(f"No portfolio results found for {strategy}/{ticker}")
            return None
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} portfolio records for {strategy}/{ticker}")
        return df
    

    def save_benchmark_plot(
        self,
        strategy: str,
        ticker: str,
        backtest_results: Any,  # BacktestResults
        figsize: Tuple[int, int] = (12, 6),
        show: bool = False
    ) -> str:
        """
        Save benchmark comparison plot.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            backtest_results: BacktestResults object from backtest engine
            figsize: Figure size
            show: Whether to display the plot
            
        Returns:
            Path to saved file
        """
        output_dir = self._get_output_dir(strategy, self.PLOTS_DIR, ticker)
        filename = self._generate_filename(ticker, strategy, 'plot', 'png')
        filepath = output_dir / filename
        
        # Generate plot
        fig, ax = self._create_benchmark_plot(
            backtest_results=backtest_results,
            strategy=strategy,
            ticker=ticker,
            figsize=figsize
        )
        
        # Save
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved benchmark plot to: {output_dir}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return str(filepath)
    

    def _create_benchmark_plot(
        self,
        backtest_results: Any,
        strategy: str,
        ticker: str,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create benchmark comparison plot.
        
        Args:
            backtest_results: BacktestResults object
            strategy: Strategy name
            ticker: Stock ticker symbol
            figsize: Figure size
            
        Returns:
            Tuple of (figure, axes)
        """
        # Calculate statistics if needed
        stats = backtest_results.calculate_statistics()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Strategy name for title
        strategy_name = strategy.replace('_', ' ').title()
        
        # Title symbol
        ticker_b = getattr(backtest_results, 'ticker_b', None)
        if ticker_b:
            title_symbol = f"{ticker}/{ticker_b}"
        else:
            title_symbol = ticker
        
        # Get data
        strategy_df = backtest_results.strategy_df
        aum = backtest_results.aum
        benchmark_aum = backtest_results.benchmark_aum
        
        # Plot AUM
        ax.plot(
            strategy_df.index,
            aum,
            label=f'{strategy_name} Strategy',
            linewidth=2,
            color='k'
        )
        ax.plot(
            strategy_df.index,
            benchmark_aum,
            label=f'{ticker} Buy & Hold',
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
        
        # Title with key metrics
        title = f'{title_symbol} {strategy_name} Strategy\n'
        title += f'Return: {stats.total_return*100:.1f}% | '
        title += f'Sharpe: {stats.sharpe_ratio:.2f} | '
        title += f'Max DD: {stats.max_drawdown*100:.1f}%'
        ax.set_title(title)
        
        plt.tight_layout()
        
        return fig, ax
    

    def save_benchmark_plot_from_df(
        self,
        strategy: str,
        ticker: str,
        strategy_df: pd.DataFrame,
        initial_aum: float = 100000.0,
        figsize: Tuple[int, int] = (12, 6),
        show: bool = False
    ) -> str:
        """
        Save benchmark comparison plot from a DataFrame directly.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            strategy_df: DataFrame with AUM and ret_ticker columns
            initial_aum: Initial AUM value
            figsize: Figure size
            show: Whether to display the plot
            
        Returns:
            Path to saved file
        """
        output_dir = self._get_output_dir(strategy, self.PLOTS_DIR, ticker)
        filename = self._generate_filename(ticker, strategy, 'plot', 'png')
        filepath = output_dir / filename
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Strategy name for title
        strategy_name = strategy.replace('_', ' ').title()
        
        # Calculate benchmark AUM if not present
        if 'AUM_ticker' not in strategy_df.columns:
            strategy_df = strategy_df.copy()
            strategy_df['AUM_ticker'] = initial_aum * (1 + strategy_df['ret_ticker']).cumprod(skipna=True)
        
        # Plot
        ax.plot(
            strategy_df.index,
            strategy_df['AUM'],
            label=f'{strategy_name} Strategy',
            linewidth=2,
            color='k'
        )
        ax.plot(
            strategy_df.index,
            strategy_df['AUM_ticker'],
            label=f'{ticker} Buy & Hold',
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
        ax.set_title(f'{ticker} {strategy_name} Strategy Backtest')
        
        plt.tight_layout()
        
        # Save
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved benchmark plot to: {output_dir}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return str(filepath)
    

    def list_strategies(self) -> List[str]:
        """
        List all strategies with saved outputs.
        
        Returns:
            List of strategy names
        """
        if not self.base_dir.exists():
            return []
        
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    

    def list_tickers(self, strategy: str, output_type: str = SIMULATIONS_DIR) -> List[str]:
        """
        List all tickers with saved outputs for a strategy.
        
        Args:
            strategy: Strategy name
            output_type: Output type to check
            
        Returns:
            List of ticker symbols
        """
        strategy_dir = self._get_strategy_dir(strategy)
        type_dir = strategy_dir / output_type
        
        if not type_dir.exists():
            return []
        
        return [d.name for d in type_dir.iterdir() if d.is_dir()]
    

    def list_versions(
        self,
        strategy: str,
        ticker: str,
        output_type: str = SIMULATIONS_DIR
    ) -> List[Dict[str, Any]]:
        """
        List all saved versions for a strategy/ticker.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            output_type: Output type
            
        Returns:
            List of dicts with filename and timestamp info
        """
        extension = 'png' if output_type == self.PLOTS_DIR else 'csv'
        files = self._get_all_files(strategy, output_type, ticker, extension)
        
        versions = []
        for f in files:
            ts = self._parse_filename_timestamp(f.name)
            versions.append({
                'filename': f.name,
                'path': str(f),
                'timestamp': ts,
                'size_mb': f.stat().st_size / (1024 * 1024)
            })
        
        return versions
    

    def get_output_info(
        self,
        strategy: str,
        ticker: str,
        output_type: str = SIMULATIONS_DIR
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about the latest saved output.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            output_type: Output type
            
        Returns:
            Dictionary with output info or None if not found
        """
        extension = 'png' if output_type == self.PLOTS_DIR else 'csv'
        filepath = self._get_latest_file(strategy, output_type, ticker, extension)
        
        if filepath is None:
            return None
        
        info = {
            'strategy': strategy,
            'ticker': ticker,
            'output_type': output_type,
            'filepath': str(filepath),
            'filename': filepath.name,
            'timestamp': self._parse_filename_timestamp(filepath.name),
            'size_mb': filepath.stat().st_size / (1024 * 1024)
        }
        
        # Add row count for CSV files
        if extension == 'csv':
            with open(filepath, 'r') as f:
                info['row_count'] = sum(1 for _ in f) - 1
        
        return info
    

    def cleanup_old_versions(
        self,
        strategy: str,
        ticker: str,
        output_type: str = SIMULATIONS_DIR,
        keep_n: int = 3
    ) -> int:
        """
        Remove old versions, keeping only the N most recent.
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            output_type: Output type to clean
            keep_n: Number of versions to keep
            
        Returns:
            Number of files deleted
        """
        extension = 'png' if output_type == self.PLOTS_DIR else 'csv'
        files = self._get_all_files(strategy, output_type, ticker, extension)
        
        if len(files) <= keep_n:
            return 0
        
        files_to_delete = files[keep_n:]
        deleted = 0
        
        for f in files_to_delete:
            try:
                f.unlink()
                deleted += 1
                
                # Also delete associated JSON metadata if exists
                json_path = f.with_suffix('.json')
                if json_path.exists():
                    json_path.unlink()
                    deleted += 1
                
                logger.info(f"Deleted old version: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {f.name}: {e}")
        
        return deleted
    

    def save_all_best_strategy_outputs(
        self,
        strategy: str,
        ticker: str,
        backtest_results: Any,
        signals_df: pd.DataFrame,
        portfolio_df: pd.DataFrame,
        config: Dict[str, Any],
        show_plot: bool = False
    ) -> Dict[str, str]:
        """
        Save all outputs for the best performing strategy after simulation.
        
        This is a convenience method that saves:
            - Strategy signals
            - Portfolio results (trade-by-trade)
            - Benchmark comparison plot
        
        Args:
            strategy: Strategy name
            ticker: Stock ticker symbol
            backtest_results: BacktestResults object from backtest engine
            signals_df: DataFrame with signals
            portfolio_df: DataFrame with portfolio updates
            config: Strategy configuration
            show_plot: Whether to display the plot
            
        Returns:
            Dictionary mapping output type to file path
        """
        paths = {}
        
        # Calculate statistics for metadata
        stats = backtest_results.calculate_statistics()
        stats_dict = {
            'total_return': stats.total_return,
            'sharpe_ratio': stats.sharpe_ratio,
            'sortino_ratio': stats.sortino_ratio,
            'calmar_ratio': stats.calmar_ratio,
            'max_drawdown': stats.max_drawdown,
            'win_rate': stats.win_rate,
            'total_trades': stats.total_trades,
            'annualized_volatility': stats.annualized_volatility,
        }
        
        # Save signals
        paths['signals'] = self.save_signals(strategy, ticker, signals_df, config)
        
        # Save portfolio results
        paths['results'] = self.save_portfolio_results(
            strategy, ticker, portfolio_df, config, stats_dict
        )
        
        # Save benchmark plot
        paths['plot'] = self.save_benchmark_plot(
            strategy, ticker, backtest_results, show=show_plot
        )
        
        logger.info(f"Saved all outputs for best {strategy} strategy on {ticker}")
        
        return paths
