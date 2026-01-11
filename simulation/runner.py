"""
Simulation runner for parallel backtest execution.

This module provides the main simulation runner that:
    - Loads data for all tickers
    - Generates task combinations
    - Executes backtests in parallel
    - Collects and analyzes results
"""

import os
import time
import logging
import multiprocessing
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple

from .config import SimulationConfig, StrategyType
from .tasks import TaskGenerator, build_backtest_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _prewarm_numba_jit_cache() -> None:
    """
    Pre-warm Numba JIT compilation cache before parallel execution.
    
    This compiles all numba functions in the main process, creating
    cache files that worker processes can read. Without this, each
    worker would compile functions independently, causing significant
    slowdown (3-4 tasks/sec vs 10+ tasks/sec).
    
    Call this once before spawning parallel workers.
    """
    logger.info("Pre-warming Numba JIT cache...")
    
    # Create small dummy arrays for compilation
    n = 100
    dummy_prices = np.random.randn(n).cumsum() + 100
    dummy_prices = dummy_prices.astype(np.float64)
    dummy_high = (dummy_prices + np.abs(np.random.randn(n))).astype(np.float64)
    dummy_low = (dummy_prices - np.abs(np.random.randn(n))).astype(np.float64)
    dummy_vwap = dummy_prices.copy()
    dummy_signals = np.zeros(n, dtype=np.int32)
    dummy_signals[10] = 1
    dummy_signals[50] = -1
    dummy_mask = np.ones(n, dtype=np.bool_)
    dummy_returns = np.random.randn(100) * 0.01
    
    # Import and compile core portfolio functions
    try:
        from core.portfolio import (
            simulate_positions_numba,
            calculate_portfolio_values_numba,
        )
        
        # Trigger compilation
        simulate_positions_numba(
            dummy_prices, dummy_signals, dummy_mask,
            0.02, 0.04, 0, 0.0
        )
        
        positions = np.zeros(n, dtype=np.int32)
        positions[20:60] = 1
        entry_prices = np.zeros(n, dtype=np.float64)
        entry_prices[20:60] = 100.0
        
        calculate_portfolio_values_numba(
            dummy_prices, positions, entry_prices,
            100000.0, 0, 0.0035, 0.1
        )
        
        logger.debug("  - core.portfolio functions compiled")
    except ImportError as e:
        logger.warning(f"  - Could not compile core.portfolio: {e}")
    
    # Try to compile Kelly functions if available
    try:
        from core.portfolio import calculate_kelly_position_size_numba
        calculate_kelly_position_size_numba(
            dummy_returns, 100, 60, 0.5, 1.0, 30
        )
        logger.debug("  - core.portfolio Kelly functions compiled")
    except ImportError:
        pass  # Kelly functions may not be available
    
    # Import and compile momentum ATR functions
    try:
        from strats.momentum_atr import (
            calculate_true_range_numba,
            calculate_atr_numba,
            calculate_atr_bands_numba,
            calculate_trailing_atr_bands_numba,
            MomentumATR
        )
        
        # Trigger compilation
        calculate_true_range_numba(dummy_high, dummy_low, dummy_prices)
        atr = calculate_atr_numba(dummy_high, dummy_low, dummy_prices, 14)
        calculate_atr_bands_numba(dummy_prices, atr, 100.0, 2.0)
        calculate_trailing_atr_bands_numba(dummy_prices, dummy_high, dummy_low, atr, 2.0, 20)
        
        # Also compile through the class methods
        MomentumATR.generate_signals(
            dummy_prices, dummy_high, dummy_low, dummy_vwap,
            100.0, 99.0, 2.0, 14
        )
        MomentumATR.generate_signals_trailing(
            dummy_prices, dummy_high, dummy_low, dummy_vwap,
            2.0, 14
        )
        
        logger.debug("  - strats.momentum_atr functions compiled")
    except ImportError as e:
        logger.warning(f"  - Could not compile strats.momentum_atr: {e}")
    
    # Import and compile standard momentum functions
    try:
        from strats.momentum import Momentum
        dummy_sigma = np.ones(n, dtype=np.float64) * 0.02
        Momentum.generate_signals(
            dummy_prices, dummy_vwap, dummy_sigma,
            100.0, 99.0, 1.0
        )
        logger.debug("  - strats.momentum functions compiled")
    except ImportError as e:
        logger.warning(f"  - Could not compile strats.momentum: {e}")
    
    # Import and compile mean reversion functions
    try:
        from strats.mean_reversion import MeanReversion
        MeanReversion.generate_signals(dummy_prices, 20, 2.0, 2.0, 0.0)
        logger.debug("  - strats.mean_reversion functions compiled")
    except ImportError as e:
        logger.warning(f"  - Could not compile strats.mean_reversion: {e}")
    
    # Import and compile RSI mean reversion functions
    try:
        from strats.mean_reversion_rsi import MeanReversionRSI
        MeanReversionRSI.generate_signals(dummy_prices, 14, 30.0, 70.0, 50, True)
        logger.debug("  - strats.mean_reversion_rsi functions compiled")
    except ImportError as e:
        logger.warning(f"  - Could not compile strats.mean_reversion_rsi: {e}")
    
    # Import and compile stat arb functions
    try:
        from strats.stat_arb import StatArb
        dummy_prices_b = dummy_prices + np.random.randn(n) * 2
        StatArb.generate_signals(
            dummy_prices, dummy_prices_b.astype(np.float64),
            30, 2.0, 0.0, None, True
        )
        logger.debug("  - strats.stat_arb functions compiled")
    except ImportError as e:
        logger.warning(f"  - Could not compile strats.stat_arb: {e}")
    
    logger.info("Numba JIT cache pre-warming complete")



def _run_single_backtest(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Run a single backtest task.
    
    This function must be at module level for multiprocessing compatibility.
    
    Args:
        task: Task dictionary with all backtest parameters
        
    Returns:
        Dictionary with results or None if failed
    """
    try:
        # Import here to avoid circular imports
        from backtesting.engine import BacktestEngine
        from core.metrics import MetricsCalculator
        from core.benchmarks import BenchmarkMetrics
        
        ticker = task['ticker']
        df = task['df'].copy()
        df_daily = task['df_daily']
        strategy_type = task['strategy_type']
        risk_free_rate_series = task.get('risk_free_rate_series')
        market_calendar = task.get('market_calendar', {})
        
        # Prepare data
        df['day'] = pd.to_datetime(df['day']).dt.normalize()
        all_days = df['day'].unique()
        
        # Build backtest config
        config = build_backtest_config(task)
        
        # Add precomputed probabilities to config if available (FAST PATH for ML)
        if 'precomputed_probabilities' in task and task['precomputed_probabilities'] is not None:
            config['precomputed_probabilities'] = task['precomputed_probabilities']
        
        # Handle stat arb
        df_b = None
        df_b_daily = None
        ticker_b = None
        
        if strategy_type == StrategyType.STAT_ARB:
            ticker_b = task.get('ticker_b')
            df_b = task.get('df_b')
            if df_b is not None:
                df_b = df_b.copy()
                df_b['day'] = pd.to_datetime(df_b['day']).dt.normalize()
            df_b_daily = task.get('df_b_daily')
        
        # Run backtest
        engine = BacktestEngine(ticker=ticker, ticker_b=ticker_b)
        
        results = engine.run(
            df=df,
            ticker_daily_data=df_daily,
            all_days=all_days,
            config=config,
            market_calendar=market_calendar,
            risk_free_rate_series=risk_free_rate_series,
            verbose=False,
            df_b=df_b,
            ticker_b_daily_data=df_b_daily
        )
        
        # Calculate statistics
        stats = results.calculate_statistics()
        
        # Build result dictionary
        result = {
            'Ticker': ticker,
            'Strategy Type': strategy_type.value,
            'Trading Frequency': task['trade_freq'],
            'Target Volatility': task['target_vol'],
            'Stop Loss': task['stop_loss_pct'],
            'Take Profit': task['take_profit_pct'],
            'Max Drawdown Limit': task['max_drawdown_pct'],
            'Total Trades': stats.total_trades,
            'Buy Trades': stats.buy_trades,
            'Sell Trades': stats.sell_trades,
            'Stop Losses Hit': stats.stop_losses_hit,
            'Take Profits Hit': stats.take_profits_hit,
            'Win Rate (%)': round(stats.win_rate * 100, 2),
            'Max Drawdown (%)': round(stats.max_drawdown * 100, 2),
            'Annualized Volatility (%)': round(stats.annualized_volatility * 100, 2),
            'Total % Return (Strategy)': round(stats.total_return * 100, 2),
            'Sharpe Ratio (Strategy)': round(stats.sharpe_ratio, 4) if not np.isnan(stats.sharpe_ratio) else np.nan,
            'Calmar Ratio (Strategy)': round(stats.calmar_ratio, 4) if not np.isnan(stats.calmar_ratio) else np.nan,
            'Sortino Ratio (Strategy)': round(stats.sortino_ratio, 4) if not np.isnan(stats.sortino_ratio) else np.nan,
            'Total % Return (Buy & Hold)': round(stats.benchmark_return * 100, 2),
            'Sharpe Ratio (Buy & Hold)': round(stats.benchmark_sharpe, 4) if not np.isnan(stats.benchmark_sharpe) else np.nan,
            'Calmar Ratio (Buy & Hold)': round(stats.benchmark_calmar, 4) if not np.isnan(stats.benchmark_calmar) else np.nan,
            'Sortino Ratio (Buy & Hold)': round(stats.benchmark_sortino, 4) if not np.isnan(stats.benchmark_sortino) else np.nan,
            'Beta': round(stats.beta, 4) if not np.isnan(stats.beta) else np.nan,
            'Alpha (annualized %)': round(stats.alpha * 100, 4) if not np.isnan(stats.alpha) else np.nan,
            'Correlation': round(stats.correlation, 4) if not np.isnan(stats.correlation) else np.nan,
            'Information Ratio': round(stats.information_ratio, 4) if not np.isnan(stats.information_ratio) else np.nan,
            'R-Squared': round(stats.r_squared, 4) if not np.isnan(stats.r_squared) else np.nan,
        }
        
        # Add strategy-specific parameters
        if strategy_type == StrategyType.MOMENTUM:
            result['Band Multiplier'] = task['band_mult']
        
        elif strategy_type == StrategyType.MOMENTUM_ATR:
            result['ATR Multiplier'] = task['atr_mult']
            result['ATR Period'] = task['atr_period']
        
        elif strategy_type == StrategyType.MEAN_REVERSION:
            result['Z-Score Lookback'] = task['zscore_lookback']
            result['N Std Upper'] = task['n_std_upper']
            result['N Std Lower'] = task['n_std_lower']
            result['Exit Threshold'] = task.get('exit_threshold', 0.0)
        
        elif strategy_type == StrategyType.MEAN_REVERSION_RSI:
            result['RSI Period'] = task['rsi_period']
            result['RSI Oversold'] = task['rsi_oversold']
            result['RSI Overbought'] = task['rsi_overbought']
            result['SMA Period'] = task['sma_period']
        
        elif strategy_type == StrategyType.STAT_ARB:
            result['Ticker B'] = ticker_b
            result['Z-Score Lookback'] = task['zscore_lookback']
            result['Entry Threshold'] = task['entry_threshold']
            result['Exit Threshold'] = task.get('exit_threshold', 0.0)
            result['Use Dynamic Hedge'] = task.get('use_dynamic_hedge', True)
        
        elif strategy_type == StrategyType.SUPERVISED:
            result['Model Path'] = task['model_path']
            result['Scaler Path'] = task.get('scaler_path', '')
            result['Features Path'] = task.get('features_path', '')
            result['Buy Threshold'] = task.get('buy_threshold', 0.55)
            result['Sell Threshold'] = task.get('sell_threshold', 0.45)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in backtest for {task.get('ticker', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


class SimulationRunner:
    """
    Simulation runner for parallel backtest execution.
    
    This class manages:
        - Data loading for all tickers
        - Task generation
        - Parallel backtest execution
        - Results collection and analysis
    
    Example:
        config = SimulationConfig(
            strategy_type=StrategyType.MOMENTUM,
            tickers=['NVDA', 'AAPL']
        )
        
        runner = SimulationRunner(config)
        results = runner.run()
        
        # Get best configurations
        best = runner.get_best_by_ticker(results)
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        data_dir: str = 'data',
        output_dir: str = 'outputs/strategies'
    ):
        """
        Initialize the SimulationRunner.
        
        Args:
            config: Simulation configuration
            data_dir: Directory for data files
            output_dir: Directory for strategy outputs
        """
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Determine number of workers
        if config.n_workers is None:
            self.n_workers = max(1, multiprocessing.cpu_count() - 2)
        else:
            self.n_workers = config.n_workers
        
        # Data cache
        self._data_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        
        # Risk-free rate series
        self.risk_free_rate_series: Optional[pd.Series] = None
        
        # Market calendar (will be set from constants)
        self.market_calendar: Dict = {}
        
        # Results
        self.results: List[Dict[str, Any]] = []
        
        # Output manager
        self._output_manager = None
    
    def load_risk_free_rate(self, path: str = 'data/dtb3.csv') -> None:
        """
        Load risk-free rate data.
        
        Args:
            path: Path to risk-free rate CSV file
        """
        try:
            rf = pd.read_csv(path)
            rf['date'] = pd.to_datetime(rf['date'])
            rf = rf.set_index('date')
            rf.index = rf.index.normalize()
            self.risk_free_rate_series = rf['rate'] / 100.0
            logger.info(f"Loaded risk-free rate: {self.risk_free_rate_series.mean()*100:.2f}% avg")
        except Exception as e:
            logger.warning(f"Could not load risk-free rate: {e}")
            self.risk_free_rate_series = None
    
    def set_market_calendar(self, calendar: Dict) -> None:
        """
        Set market calendar.
        
        Args:
            calendar: Dictionary with 'holidays' and 'early_closes'
        """
        self.market_calendar = calendar
    
    def load_data(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load or fetch data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (intraday_df, daily_df)
        """
        if ticker in self._data_cache:
            return self._data_cache[ticker]
        
        ticker_dir = os.path.join(self.data_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Find existing data files
        if os.path.exists(ticker_dir):
            intra_files = [f for f in os.listdir(ticker_dir)
                          if f.startswith(f'{ticker}_intra_feed') and f.endswith('.csv')]
            daily_files = [f for f in os.listdir(ticker_dir)
                          if f.startswith(f'{ticker}_daily_feed') and f.endswith('.csv')]
            
            if intra_files and daily_files:
                intra_files.sort(key=lambda x: os.path.getmtime(os.path.join(ticker_dir, x)), reverse=True)
                daily_files.sort(key=lambda x: os.path.getmtime(os.path.join(ticker_dir, x)), reverse=True)
                
                try:
                    df = pd.read_csv(os.path.join(ticker_dir, intra_files[0]), index_col=0)
                    df_daily = pd.read_csv(os.path.join(ticker_dir, daily_files[0]), index_col=0)
                    
                    if len(df) > 0 and len(df_daily) > 0:
                        self._data_cache[ticker] = (df, df_daily)
                        logger.info(f"Loaded data for {ticker}: {len(df)} intraday records")
                        return df, df_daily
                except Exception as e:
                    logger.warning(f"Error loading data for {ticker}: {e}")
        
        # Fetch new data
        logger.info(f"Fetching data for {ticker}...")
        try:
            from data import DataFetcher
            fetcher = DataFetcher(ticker)
            df, df_daily = fetcher.process_data()
            
            if not df.empty:
                version = datetime.now().strftime("%Y%m%d%H%M%S")
                df.to_csv(os.path.join(ticker_dir, f'{ticker}_intra_feed_{version}.csv'))
                df_daily.to_csv(os.path.join(ticker_dir, f'{ticker}_daily_feed_{version}.csv'))
                self._data_cache[ticker] = (df, df_daily)
            
            return df, df_daily
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def run(self, parallel: bool = True) -> pd.DataFrame:
        """
        Run the simulation.
        
        Args:
            parallel: Whether to use parallel execution
            
        Returns:
            DataFrame with all results
        """
        logger.info(f"Starting Simulation ({self.config.strategy_type.value.upper()})")
        logger.info(f"Estimated combinations: {self.config.count_combinations()}")
        
        # Load risk-free rate if not already loaded
        if self.risk_free_rate_series is None:
            self.load_risk_free_rate()
        
        # Generate tasks
        tasks = []
        task_generator = TaskGenerator(self.config)
        
        if self.config.strategy_type == StrategyType.STAT_ARB:
            # Load data for pairs
            for ticker_a, ticker_b in self.config.pairs:
                df_a, df_daily_a = self.load_data(ticker_a)
                df_b, df_daily_b = self.load_data(ticker_b)
                
                if df_a.empty or df_b.empty:
                    logger.warning(f"Skipping pair {ticker_a}/{ticker_b}: missing data")
                    continue
                
                pair_tasks = task_generator.generate_tasks(
                    ticker=ticker_a,
                    df=df_a,
                    df_daily=df_daily_a,
                    market_calendar=self.market_calendar,
                    risk_free_rate_series=self.risk_free_rate_series,
                    df_b=df_b,
                    df_b_daily=df_daily_b,
                    ticker_b=ticker_b
                )
                tasks.extend(pair_tasks)
        else:
            # Load data for tickers
            for ticker in self.config.tickers:
                df, df_daily = self.load_data(ticker)
                
                if df.empty:
                    logger.warning(f"Skipping {ticker}: no data")
                    continue
                
                ticker_tasks = task_generator.generate_tasks(
                    ticker=ticker,
                    df=df,
                    df_daily=df_daily,
                    market_calendar=self.market_calendar,
                    risk_free_rate_series=self.risk_free_rate_series
                )
                tasks.extend(ticker_tasks)
        
        logger.info(f"Generated {len(tasks)} tasks")
        
        if not tasks:
            logger.warning("No tasks to run")
            return pd.DataFrame()
        
        # Pre-warm Numba JIT cache before parallel execution
        # This compiles functions in the main process so workers can use cached versions
        if parallel and len(tasks) > 1:
            _prewarm_numba_jit_cache()
        
        # Run backtests
        start_time = time.time()
        self.results = []
        total_tasks = len(tasks)
        
        if parallel and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(_run_single_backtest, task): i for i, task in enumerate(tasks)}
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    if result is not None:
                        self.results.append(result)
                    
                    completed = i + 1
                    if completed % 50 == 0 or completed == total_tasks:
                        elapsed = time.time() - start_time
                        tasks_per_sec = completed / elapsed if elapsed > 0 else 0
                        remaining = total_tasks - completed
                        eta = remaining / tasks_per_sec if tasks_per_sec > 0 else 0
                        logger.info(
                            f"[{completed}/{total_tasks}] completed | "
                            f"Elapsed: {elapsed:.1f}s | "
                            f"{tasks_per_sec:.1f} tasks/s | "
                            f"ETA: {eta:.1f}s"
                        )
        else:
            for i, task in enumerate(tasks):
                result = _run_single_backtest(task)
                if result is not None:
                    self.results.append(result)
                
                completed = i + 1
                if completed % 10 == 0 or completed == total_tasks:
                    elapsed = time.time() - start_time
                    tasks_per_sec = completed / elapsed if elapsed > 0 else 0
                    remaining = total_tasks - completed
                    eta = remaining / tasks_per_sec if tasks_per_sec > 0 else 0
                    logger.info(
                        f"[{completed}/{total_tasks}] completed | "
                        f"Elapsed: {elapsed:.1f}s | "
                        f"{tasks_per_sec:.1f} tasks/s | "
                        f"ETA: {eta:.1f}s"
                    )
        
        elapsed = time.time() - start_time
        logger.info(f"Completed {len(self.results)} backtests in {elapsed:.1f}s")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save results
        if not results_df.empty:
            self._save_results(results_df)
        
        return results_df
    
    def _save_results(self, results_df: pd.DataFrame) -> str:
        """
        Save results to CSV using the StrategyOutputManager.
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Path to saved file
        """
        # Lazy import to avoid circular imports
        from outputs import StrategyOutputManager
        
        if self._output_manager is None:
            self._output_manager = StrategyOutputManager(base_dir=self.output_dir)
        
        strategy_name = self.config.strategy_type.value
        
        # Save results for each ticker
        paths = []
        tickers = results_df['Ticker'].unique()
        
        for ticker in tickers:
            ticker_results = results_df[results_df['Ticker'] == ticker]
            path = self._output_manager.save_simulation_results(
                strategy=strategy_name,
                ticker=ticker,
                results_df=ticker_results,
                metadata={'n_combinations': len(ticker_results)}
            )
            paths.append(path)
        
        logger.info(f"Results saved for {len(tickers)} tickers")
        return paths[0] if paths else ''
    
    def get_best_by_ticker(
        self,
        results_df: pd.DataFrame,
        metric: str = 'Sharpe Ratio (Strategy)'
    ) -> pd.DataFrame:
        """
        Get best configuration for each ticker.
        
        Args:
            results_df: Results DataFrame
            metric: Metric to optimize
            
        Returns:
            DataFrame with best configuration per ticker
        """
        if results_df.empty or metric not in results_df.columns:
            return pd.DataFrame()
        
        idx = results_df.groupby('Ticker')[metric].idxmax()
        return results_df.loc[idx]
    
    def get_best_by_pair(
        self,
        results_df: pd.DataFrame,
        metric: str = 'Sharpe Ratio (Strategy)'
    ) -> pd.DataFrame:
        """
        Get best configuration for each pair (stat arb only).
        
        Args:
            results_df: Results DataFrame
            metric: Metric to optimize
            
        Returns:
            DataFrame with best configuration per pair
        """
        if results_df.empty or metric not in results_df.columns:
            return pd.DataFrame()
        
        if 'Ticker B' not in results_df.columns:
            return self.get_best_by_ticker(results_df, metric)
        
        idx = results_df.groupby(['Ticker', 'Ticker B'])[metric].idxmax()
        return results_df.loc[idx]
    
    def print_best_results(self, results_df: pd.DataFrame) -> None:
        """
        Print best results summary.
        
        Args:
            results_df: Results DataFrame
        """
        if results_df.empty:
            print("No results to display")
            return
        
        if self.config.strategy_type == StrategyType.STAT_ARB:
            best = self.get_best_by_pair(results_df)
        else:
            best = self.get_best_by_ticker(results_df)
        
        print("\n" + "=" * 80)
        print(f"BEST {self.config.strategy_type.value.upper()} CONFIGURATIONS")
        print("=" * 80)
        
        for _, row in best.iterrows():
            print(f"\n{'-' * 40}")
            print(f"Ticker: {row['Ticker']}")
            if 'Ticker B' in row and pd.notna(row['Ticker B']):
                print(f"Ticker B: {row['Ticker B']}")
            print(f"{'-' * 40}")
            print(f"Total Return: {row['Total % Return (Strategy)']:.2f}%")
            print(f"Sharpe Ratio: {row['Sharpe Ratio (Strategy)']:.4f}")
            print(f"Max Drawdown: {row['Max Drawdown (%)']:.2f}%")
            print(f"Win Rate: {row['Win Rate (%)']:.2f}%")
            print(f"Total Trades: {row['Total Trades']}")
    

    def run_best_strategy(
        self,
        results_df: pd.DataFrame,
        ticker: str,
        metric: str = 'Sharpe Ratio (Strategy)',
        save_outputs: bool = True,
        show_plot: bool = False,
        verbose: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Run the best performing strategy for a ticker and save all outputs.
        
        This method:
            1. Finds the best configuration from simulation results
            2. Runs a fresh backtest with that configuration
            3. Saves signals, portfolio results, and benchmark plot
        
        Args:
            results_df: Simulation results DataFrame
            ticker: Stock ticker symbol
            metric: Metric to optimize (default: Sharpe Ratio)
            save_outputs: Whether to save outputs to files
            show_plot: Whether to display the benchmark plot
            verbose: Print progress information
            
        Returns:
            Dictionary with backtest results and file paths, or None if failed
        """
        from backtesting.engine import BacktestEngine
        from outputs import StrategyOutputManager
        
        # Get best configuration
        ticker_results = results_df[results_df['Ticker'] == ticker.upper()]
        if ticker_results.empty:
            logger.warning(f"No results found for {ticker}")
            return None
        
        best_idx = ticker_results[metric].idxmax()
        best_row = results_df.loc[best_idx]
        
        if verbose:
            logger.info(f"Best {self.config.strategy_type.value} config for {ticker}:")
            logger.info(f"  - Sharpe: {best_row.get('Sharpe Ratio (Strategy)', 'N/A')}")
            logger.info(f"  - Return: {best_row.get('Total % Return (Strategy)', 'N/A')}%")
        
        # Load data
        df, df_daily = self.load_data(ticker)
        if df.empty:
            logger.error(f"Failed to load data for {ticker}")
            return None
        
        # Prepare data
        df['day'] = pd.to_datetime(df['day']).dt.normalize()
        all_days = df['day'].unique()
        
        # Build config from best row
        config = self._build_config_from_row(best_row)
        
        # For SUPERVISED strategy, precompute probabilities (same as simulation)
        if self.config.strategy_type == StrategyType.SUPERVISED:
            model_path = config.get('model_path')
            scaler_path = config.get('scaler_path')
            features_path = config.get('features_path')
            
            if all([model_path, scaler_path, features_path]):
                try:
                    from strats.classification import SupervisedStrategy
                    
                    strategy = SupervisedStrategy.from_artifacts(
                        model_path=model_path,
                        scaler_path=scaler_path,
                        features_path=features_path,
                        buy_threshold=config.get('buy_threshold', 0.55),
                        sell_threshold=config.get('sell_threshold', 0.45)
                    )
                    
                    probabilities = strategy.get_probabilities(df)
                    if probabilities is not None and len(probabilities) > 0:
                        config['precomputed_probabilities'] = probabilities
                        if verbose:
                            logger.info(f"  Pre-computed {len(probabilities)} probabilities for best strategy")
                except Exception as e:
                    if verbose:
                        logger.warning(f"  Could not precompute probabilities: {e}")
        
        # Handle stat arb
        df_b = None
        df_b_daily = None
        ticker_b = None
        
        if self.config.strategy_type == StrategyType.STAT_ARB:
            ticker_b = best_row.get('Ticker B')
            if ticker_b:
                df_b, df_b_daily = self.load_data(ticker_b)
                if df_b is not None and not df_b.empty:
                    df_b['day'] = pd.to_datetime(df_b['day']).dt.normalize()
        
        # Run backtest
        engine = BacktestEngine(ticker=ticker, ticker_b=ticker_b)
        
        results = engine.run(
            df=df,
            ticker_daily_data=df_daily,
            all_days=all_days,
            config=config,
            market_calendar=self.market_calendar,
            risk_free_rate_series=self.risk_free_rate_series,
            verbose=verbose,
            df_b=df_b,
            ticker_b_daily_data=df_b_daily
        )
        
        # Prepare output data
        output = {
            'ticker': ticker,
            'strategy': self.config.strategy_type.value,
            'config': config,
            'results': results,
            'statistics': results.calculate_statistics(),
            'paths': {}
        }
        
        if save_outputs:
            # Initialize output manager
            if self._output_manager is None:
                self._output_manager = StrategyOutputManager(base_dir=self.output_dir)
            
            strategy_name = self.config.strategy_type.value
            
            # Prepare signals DataFrame
            signals_df = self._extract_signals_df(df, results, config)
            
            # Prepare portfolio DataFrame (trade-by-trade)
            portfolio_df = self._extract_portfolio_df(results)
            
            # Save all outputs
            output['paths'] = self._output_manager.save_all_best_strategy_outputs(
                strategy=strategy_name,
                ticker=ticker,
                backtest_results=results,
                signals_df=signals_df,
                portfolio_df=portfolio_df,
                config=config,
                show_plot=show_plot
            )
            
            if verbose:
                logger.info(f"Saved outputs for best {strategy_name} strategy on {ticker}")
                for output_type, path in output['paths'].items():
                    logger.info(f"  - {output_type}: {path}")
        
        return output
    

    def _build_config_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Build backtest config from a results row.
        
        Args:
            row: Row from simulation results DataFrame
            
        Returns:
            Configuration dictionary
        """
        config = {
            'strategy_type': self.config.strategy_type.value,
            'AUM': self.config.common.initial_aum,
            'trade_freq': int(row.get('Trading Frequency', 30)),
            'target_vol': float(row.get('Target Volatility', 0.02)),
            'stop_loss_pct': float(row.get('Stop Loss', 0.02)),
            'take_profit_pct': float(row.get('Take Profit', 0.04)),
            'max_drawdown_pct': float(row.get('Max Drawdown Limit', 0.15)),
        }
        
        # Strategy-specific parameters
        strategy_type = self.config.strategy_type
        
        if strategy_type == StrategyType.MOMENTUM:
            config['band_mult'] = float(row.get('Band Multiplier', 1.0))
        
        elif strategy_type == StrategyType.MOMENTUM_ATR:
            config['atr_mult'] = float(row.get('ATR Multiplier', 2.0))
            config['atr_period'] = int(row.get('ATR Period', 14))
        
        elif strategy_type == StrategyType.MEAN_REVERSION:
            config['zscore_lookback'] = int(row.get('Z-Score Lookback', 20))
            config['n_std_upper'] = float(row.get('N Std Upper', 2.0))
            config['n_std_lower'] = float(row.get('N Std Lower', 2.0))
            config['exit_threshold'] = float(row.get('Exit Threshold', 0.0))
        
        elif strategy_type == StrategyType.MEAN_REVERSION_RSI:
            config['rsi_period'] = int(row.get('RSI Period', 10))
            config['rsi_oversold'] = float(row.get('RSI Oversold', 30.0))
            config['rsi_overbought'] = float(row.get('RSI Overbought', 70.0))
            config['sma_period'] = int(row.get('SMA Period', 200))
        
        elif strategy_type == StrategyType.STAT_ARB:
            config['zscore_lookback'] = int(row.get('Z-Score Lookback', 60))
            config['entry_threshold'] = float(row.get('Entry Threshold', 2.0))
            config['exit_threshold'] = float(row.get('Exit Threshold', 0.0))
            config['use_dynamic_hedge'] = row.get('Use Dynamic Hedge', True)
        
        elif strategy_type == StrategyType.SUPERVISED:
            config['model_path'] = row.get('Model Path')
            config['scaler_path'] = row.get('Scaler Path')
            config['features_path'] = row.get('Features Path')
            config['buy_threshold'] = float(row.get('Buy Threshold', 0.55))
            config['sell_threshold'] = float(row.get('Sell Threshold', 0.45))
            
            # If paths not in row, try to get from simulation config
            if not config['model_path'] and self.config.supervised.model_paths:
                config['model_path'] = self.config.supervised.model_paths[0]
            if not config['scaler_path'] and self.config.supervised.scaler_paths:
                config['scaler_path'] = self.config.supervised.scaler_paths[0]
            if not config['features_path'] and self.config.supervised.features_paths:
                config['features_path'] = self.config.supervised.features_paths[0]
        
        return config
    

    def _extract_signals_df(
        self,
        df: pd.DataFrame,
        results: Any,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Extract signals DataFrame from backtest data.
        
        Args:
            df: Original intraday DataFrame
            results: BacktestResults object
            config: Strategy configuration
            
        Returns:
            DataFrame with signals
        """
        # Get strategy DataFrame
        strategy_df = results.strategy_df.copy()
        
        # Create signals DataFrame with position changes
        signals_df = pd.DataFrame(index=strategy_df.index)
        signals_df['position'] = strategy_df['position']
        signals_df['position_change'] = strategy_df['position'].diff().fillna(0)
        
        # Derive signal from position changes
        signals_df['signal'] = 0
        signals_df.loc[signals_df['position_change'] > 0, 'signal'] = 1  # Buy
        signals_df.loc[signals_df['position_change'] < 0, 'signal'] = -1  # Sell
        
        # Add price info if available
        if 'ret' in strategy_df.columns:
            signals_df['daily_return'] = strategy_df['ret']
        
        if 'AUM' in strategy_df.columns:
            signals_df['aum'] = strategy_df['AUM']
        
        return signals_df
    

    def _extract_portfolio_df(self, results: Any) -> pd.DataFrame:
        """
        Extract portfolio DataFrame (trade-by-trade updates).
        
        Args:
            results: BacktestResults object
            
        Returns:
            DataFrame with portfolio updates
        """
        strategy_df = results.strategy_df.copy()
        
        # Select relevant columns for trade-by-trade view
        columns_to_keep = [
            'position', 'AUM', 'balance', 'shares_held', 'ret',
            'ret_ticker', 'slippage_cost', 'drawdown'
        ]
        
        available_cols = [c for c in columns_to_keep if c in strategy_df.columns]
        portfolio_df = strategy_df[available_cols].copy()
        
        # Add derived columns
        portfolio_df['direction'] = 'WAIT'
        portfolio_df.loc[strategy_df['position'].diff() > 0, 'direction'] = 'BUY'
        portfolio_df.loc[strategy_df['position'].diff() < 0, 'direction'] = 'SELL'
        
        # Calculate cumulative returns
        if 'ret' in portfolio_df.columns:
            portfolio_df['cumulative_returns'] = (1 + portfolio_df['ret']).cumprod() - 1
        
        # Calculate PnL
        if 'AUM' in portfolio_df.columns:
            initial_aum = results.initial_aum
            portfolio_df['PnL'] = portfolio_df['AUM'] - initial_aum
        
        return portfolio_df
    

    def run_all_best_strategies(
        self,
        results_df: pd.DataFrame,
        metric: str = 'Sharpe Ratio (Strategy)',
        save_outputs: bool = True,
        show_plots: bool = False,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run best strategy for all tickers and save outputs.
        
        Args:
            results_df: Simulation results DataFrame
            metric: Metric to optimize
            save_outputs: Whether to save outputs
            show_plots: Whether to display plots
            verbose: Print progress information
            
        Returns:
            Dictionary mapping ticker to output info
        """
        tickers = results_df['Ticker'].unique()
        outputs = {}
        
        for ticker in tickers:
            if verbose:
                logger.info(f"\nProcessing best strategy for {ticker}...")
            
            output = self.run_best_strategy(
                results_df=results_df,
                ticker=ticker,
                metric=metric,
                save_outputs=save_outputs,
                show_plot=show_plots,
                verbose=verbose
            )
            
            if output:
                outputs[ticker] = output
        
        return outputs


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simulation sweep')
    parser.add_argument(
        '--strategy',
        type=str,
        default='momentum',
        choices=['momentum', 'momentum_atr', 'mean_reversion', 'mean_reversion_rsi', 'stat_arb', 'supervised'],
        help='Strategy type'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        default=['NVDA'],
        help='Tickers to test'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = SimulationConfig(
        strategy_type=StrategyType(args.strategy),
        tickers=args.tickers,
        n_workers=args.workers
    )
    
    # Run simulation
    runner = SimulationRunner(config)
    results = runner.run()
    
    if not results.empty:
        runner.print_best_results(results)


if __name__ == '__main__':
    main()
