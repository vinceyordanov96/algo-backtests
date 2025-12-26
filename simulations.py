import os
import logging
import pandas as pd
import numpy as np
import time
import dotenv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Tuple, Any, Optional, List

import matplotlib
matplotlib.use('Agg')

from utils import Utils
from data import DataFetcher
from backtest import BackTest
from constants import Constants
from outputs import Outputs
from benchmarks import Benchmarks
from strats.momentum import Momentum
from strats.mean_reversion import MeanReversion
from strats.mean_reversion_rsi import MeanReversionRSI
from strats.stat_arb import StatArb
from strategies import (
    StrategyType,
    StrategyFactory
)

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize outputs formatter
outputs = Outputs()


def _run_single_backtest_task(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Standalone function for running a single backtest task.
    Must be at module level for multiprocessing pickle compatibility.
    
    Args:
        task: Dictionary containing all parameters needed for the backtest
        
    Returns:
        Dictionary with backtest results or None if failed
    """
    try:
        ticker = task['ticker']
        df = task['df']
        df_daily = task['df_daily']
        risk_free_rate_series = task['risk_free_rate_series']
        market_calendar = task['market_calendar']
        strategy_type = task['strategy_type']
        trade_freq = task['trade_freq']
        target_vol = task['target_vol']
        stop_loss_pct = task['stop_loss_pct']
        take_profit_pct = task['take_profit_pct']
        max_drawdown_pct = task['max_drawdown_pct']
        
        # Create config based on strategy type
        if strategy_type == StrategyType.MOMENTUM:
            band_mult = task['band_mult']
            
            config = {
                'strategy_type': strategy_type,
                'AUM': 100000.0,
                'commission': 0.0035,
                'min_comm_per_order': 0.35,
                'band_mult': band_mult,
                'trade_freq': trade_freq,
                'sizing_type': "vol_target",
                'target_vol': target_vol,
                'max_leverage': 1,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'lookback_period': 20,
                'slippage_factor': 0.1,
                'var_confidence': 0.95
            }
            
        elif strategy_type == StrategyType.MEAN_REVERSION:
            zscore_lookback = task['zscore_lookback']
            n_std_upper = task['n_std_upper']
            n_std_lower = task['n_std_lower']
            exit_threshold = task.get('exit_threshold', 0.0)
            
            config = {
                'strategy_type': strategy_type,
                'AUM': 100000.0,
                'commission': 0.0035,
                'min_comm_per_order': 0.35,
                'zscore_lookback': zscore_lookback,
                'n_std_upper': n_std_upper,
                'n_std_lower': n_std_lower,
                'exit_threshold': exit_threshold,
                'trade_freq': trade_freq,
                'sizing_type': "vol_target",
                'target_vol': target_vol,
                'max_leverage': 1,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'lookback_period': 20,
                'slippage_factor': 0.1,
                'var_confidence': 0.95
            }

        elif strategy_type == StrategyType.MEAN_REVERSION_RSI:
            rsi_period = task['rsi_period']
            rsi_oversold = task['rsi_oversold']
            rsi_overbought = task['rsi_overbought']
            sma_period = task['sma_period']
            exit_on_sma_cross = task.get('exit_on_sma_cross', True)
            
            config = {
                'strategy_type': strategy_type,
                'AUM': 100000.0,
                'commission': 0.0035,
                'min_comm_per_order': 0.35,
                'rsi_period': rsi_period,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought,
                'sma_period': sma_period,
                'exit_on_sma_cross': exit_on_sma_cross,
                'trade_freq': trade_freq,
                'sizing_type': "vol_target",
                'target_vol': target_vol,
                'max_leverage': 1,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'lookback_period': 20,
                'slippage_factor': 0.1,
                'var_confidence': 0.95
            }
                    
        elif strategy_type == StrategyType.STAT_ARB:
            ticker_b = task['ticker_b']
            df_b = task['df_b']
            df_b_daily = task['df_b_daily']
            zscore_lookback = task['zscore_lookback']
            entry_threshold = task['entry_threshold']
            exit_threshold = task.get('exit_threshold', 0.0)
            use_dynamic_hedge = task.get('use_dynamic_hedge', True)
            
            config = {
                'strategy_type': strategy_type,
                'AUM': 100000.0,
                'commission': 0.0035,
                'min_comm_per_order': 0.35,
                'zscore_lookback': zscore_lookback,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'use_dynamic_hedge': use_dynamic_hedge,
                'hedge_ratio': None,
                'trade_freq': trade_freq,
                'sizing_type': "vol_target",
                'target_vol': target_vol,
                'max_leverage': 1,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'lookback_period': 20,
                'slippage_factor': 0.1,
                'var_confidence': 0.95
            }
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Prepare data
        df_copy = df.copy()
        df_copy['day'] = pd.to_datetime(df_copy['day']).dt.normalize()
        all_days = df_copy['day'].unique()
        
        # For stat arb, prepare second asset data
        df_b_copy = None
        df_b_daily_copy = None
        if strategy_type == StrategyType.STAT_ARB:
            df_b_copy = df_b.copy()
            df_b_copy['day'] = pd.to_datetime(df_b_copy['day']).dt.normalize()
            df_b_daily_copy = df_b_daily
        
        # Run backtest
        if strategy_type == StrategyType.STAT_ARB:
            backtest = BackTest(ticker=ticker, ticker_b=ticker_b)
        else:
            backtest = BackTest(ticker=ticker)
        
        strat = backtest.backtest(
            df=df_copy,
            ticker_daily_data=df_daily,
            all_days=all_days,
            config=config,
            market_calendar=market_calendar,
            risk_free_rate_series=risk_free_rate_series,
            verbose=False,
            df_b=df_b_copy,
            ticker_b_daily_data=df_b_daily_copy
        )
        
        # Calculate statistics
        stats = _calculate_stats(strat, backtest, ticker, risk_free_rate_series)
                
        # Build result dictionary based on strategy type
        result = {
            'Ticker': ticker,
            'Strategy Type': strategy_type.value,
            'Trading Frequency': trade_freq,
            'Target Volatility': target_vol,
            'Stop Loss': stop_loss_pct,
            'Take Profit': take_profit_pct,
            'Max Drawdown': max_drawdown_pct,
            'Total Trades': backtest.total_trades,
            'Buy Trades': backtest.buy_trades,
            'Sell Trades': backtest.sell_trades,
            'Stop Losses Hit': backtest.stop_losses,
            'Take Profits Hit': backtest.profit_takes,
        }
        
        # Add strategy-specific parameters
        if strategy_type == StrategyType.MOMENTUM:
            result['Band Multiplier'] = band_mult
        elif strategy_type == StrategyType.MEAN_REVERSION:
            result['Z-Score Lookback'] = zscore_lookback
            result['N Std Upper'] = n_std_upper
            result['N Std Lower'] = n_std_lower
            result['Exit Threshold'] = exit_threshold
        elif strategy_type == StrategyType.MEAN_REVERSION_RSI:
            result['RSI Period'] = rsi_period
            result['RSI Oversold'] = rsi_oversold
            result['RSI Overbought'] = rsi_overbought
            result['SMA Period'] = sma_period
        elif strategy_type == StrategyType.STAT_ARB:
            result['Ticker B'] = ticker_b
            result['Z-Score Lookback'] = zscore_lookback
            result['Entry Threshold'] = entry_threshold
            result['Exit Threshold'] = exit_threshold
            result['Use Dynamic Hedge'] = use_dynamic_hedge
        
        # Add calculated stats
        result.update(stats)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in backtest for {task.get('ticker', 'unknown')} with strategy "
                    f"{task.get('strategy_type', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _calculate_stats(
    strat: pd.DataFrame, 
    backtest: BackTest, 
    ticker: str,
    risk_free_rate_series: pd.Series = None
) -> Dict[str, Any]:
    """
    Calculate performance statistics from backtest results.
    Standalone function for multiprocessing compatibility.
    
    Args:
        strat: Strategy results DataFrame
        backtest: BackTest instance with trade statistics
        ticker: Ticker symbol for benchmark
        risk_free_rate_series: Risk-free rate series for Sharpe calculation
        
    Returns:
        Dictionary with calculated statistics
    """
    utils = Utils()
    benchmarks = Benchmarks(ticker)
    AUM_0 = 100000.0
    
    # Calculate benchmark AUM
    strat['AUM_ticker'] = AUM_0 * (1 + strat['ret_ticker']).cumprod(skipna=True)
    
    # Get valid returns
    valid_strat_returns = strat['ret'].dropna()
    valid_ticker_returns = strat['ret_ticker'].dropna()
    
    # Strategy total return
    total_ret = (strat['AUM'].iloc[-1] - AUM_0) / AUM_0

    # Benchmark return from cumulative product of returns
    cumulative_ticker_ret = (1 + valid_ticker_returns).prod() - 1
    ticker_ret = cumulative_ticker_ret
    
    # Sharpe ratios
    sharpe = utils.calculate_sharpe_ratio(
        valid_strat_returns,
        lookback_period=20,
        risk_free_rate_series=risk_free_rate_series,
        annualize=True
    )
    
    ticker_sharpe = utils.calculate_sharpe_ratio(
        valid_ticker_returns,
        lookback_period=20,
        risk_free_rate_series=risk_free_rate_series,
        annualize=True
    )
    
    # Max drawdown
    strat['cummax'] = strat['AUM'].cummax()
    strat['drawdown_pct'] = (strat['AUM'] - strat['cummax']) / strat['cummax']
    max_dd = strat['drawdown_pct'].min()
    
    strat['ticker_cummax'] = strat['AUM_ticker'].cummax()
    strat['ticker_drawdown_pct'] = (strat['AUM_ticker'] - strat['ticker_cummax']) / strat['ticker_cummax']
    ticker_max_dd = strat['ticker_drawdown_pct'].min()
    
    # Volatility
    strategy_vol = valid_strat_returns.std() * np.sqrt(252)
    
    # Win rate
    winning_days = (valid_strat_returns > 0).sum()
    total_days = len(valid_strat_returns)
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Annualized returns for Calmar
    num_years = len(valid_strat_returns) / 252
    ticker_num_years = len(valid_ticker_returns) / 252
    
    annualized_ret = (1 + total_ret) ** (1 / num_years) - 1 if num_years > 0 else 0
    ticker_annualized_ret = (1 + ticker_ret) ** (1 / ticker_num_years) - 1 if ticker_num_years > 0 else 0
    
    # Calmar ratio
    calmar = annualized_ret / abs(max_dd) if (max_dd != 0 and not np.isnan(max_dd)) else np.nan
    ticker_calmar = ticker_annualized_ret / abs(ticker_max_dd) if (ticker_max_dd != 0 and not np.isnan(ticker_max_dd)) else np.nan
    
    # Sortino ratio
    excess_returns = utils.subtract_risk_free_rate(
        valid_strat_returns,
        risk_free_rate_series=risk_free_rate_series
    )
    ticker_excess_returns = utils.subtract_risk_free_rate(
        valid_ticker_returns,
        risk_free_rate_series=risk_free_rate_series
    )
    
    negative_returns = excess_returns[excess_returns < 0]
    downside_dev = np.sqrt((negative_returns ** 2).mean()) * np.sqrt(252) if len(negative_returns) > 0 else np.nan
    
    ticker_negative_returns = ticker_excess_returns[ticker_excess_returns < 0]
    ticker_downside_dev = np.sqrt((ticker_negative_returns ** 2).mean()) * np.sqrt(252) if len(ticker_negative_returns) > 0 else np.nan
    
    avg_excess_return_annualized = excess_returns.mean() * 252
    sortino = avg_excess_return_annualized / downside_dev if downside_dev > 0 else np.nan
    
    ticker_avg_excess_return_annualized = ticker_excess_returns.mean() * 252
    ticker_sortino = ticker_avg_excess_return_annualized / ticker_downside_dev if ticker_downside_dev > 0 else np.nan

    benchmark_metrics = benchmarks.calculate_all_metrics(
        valid_strat_returns,
        valid_ticker_returns,
        risk_free_rate=risk_free_rate_series.mean() if risk_free_rate_series is not None else 0.0
    )
    
    return {
        'Win Rate (%)': round(win_rate * 100, 2),
        'Max Drawdown (%)': round(max_dd * 100, 2),
        'Annualized Volatility (%)': round(strategy_vol * 100, 2),
        'Total % Return (Strategy)': round(total_ret * 100, 2),
        'Sharpe Ratio (Strategy)': round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
        'Calmar Ratio (Strategy)': round(calmar, 4) if not np.isnan(calmar) else np.nan,
        'Sortino Ratio (Strategy)': round(sortino, 4) if not np.isnan(sortino) else np.nan,
        'Total % Return (Buy & Hold)': round(ticker_ret * 100, 2),
        'Sharpe Ratio (Buy & Hold)': round(ticker_sharpe, 4) if not np.isnan(ticker_sharpe) else np.nan,
        'Calmar Ratio (Buy & Hold)': round(ticker_calmar, 4) if not np.isnan(ticker_calmar) else np.nan,
        'Sortino Ratio (Buy & Hold)': round(ticker_sortino, 4) if not np.isnan(ticker_sortino) else np.nan,
        'Beta': round(benchmark_metrics['Beta'], 4) if not np.isnan(benchmark_metrics['Beta']) else np.nan,
        'Alpha (annualized %)': round(benchmark_metrics['Alpha (annualized)'] * 100, 4) if not np.isnan(benchmark_metrics['Alpha (annualized)']) else np.nan,
        'Correlation': round(benchmark_metrics['Correlation'], 4) if not np.isnan(benchmark_metrics['Correlation']) else np.nan,
        'Information Ratio': round(benchmark_metrics['Information Ratio'], 4) if not np.isnan(benchmark_metrics['Information Ratio']) else np.nan,
        'Treynor Ratio': round(benchmark_metrics['Treynor Ratio'], 4) if not np.isnan(benchmark_metrics['Treynor Ratio']) else np.nan,
        'R-Squared': round(benchmark_metrics['R-Squared'], 4) if not np.isnan(benchmark_metrics['R-Squared']) else np.nan
    }


class Simulation:
    """
    Optimized Simulation class for running backtests across multiple parameter combinations.
    Uses parallel processing and vectorized operations for improved performance.
    
    Supports multiple strategy types:
    - Momentum (band breakout)
    - Mean Reversion (z-score based)
    - Statistical Arbitrage (pairs trading)
    """
    
    def __init__(
        self,
        strategy_type: StrategyType = StrategyType.MOMENTUM,
        n_workers: int = None,
        **kwargs
    ):
        """
        Initialize the Simulation class.
        
        Parameters are loaded in the following priority (highest to lowest):
        1. Explicit kwargs passed to __init__
        2. Strategy-specific defaults from the strategy class's get_parameter_grid()
        3. Common defaults from StrategyFactory.get_common_parameter_grid()
        
        Args:
            strategy_type: The type of strategy to simulate (default: MOMENTUM)
            n_workers: Number of parallel workers. Defaults to CPU count - 2.
            **kwargs: Override any default parameters.
        """
        self.constants = Constants()
        self.outputs = Outputs()
        self.utils = Utils()
        self.strategy_type = strategy_type
        
        # Results storage
        self.results = []
        
        # Risk-free rate series
        self.risk_free_rate_series = None
        
        # Cached data for pairs
        self.ticker_data_cache = {}
        
        # Parallel processing configuration
        if n_workers is None:
            self.n_workers = max(1, multiprocessing.cpu_count() - 2)
        else:
            self.n_workers = n_workers
        
        # Load parameters from strategy classes and factory
        self._initialize_parameters(kwargs)
    
    
    def _get_strategy_class(self, strategy_type: StrategyType):
        """
        Get the strategy class for a given strategy type.
        
        Args:
            strategy_type: The type of strategy to get the class for
            
        Returns:
            The strategy class
        """
        strategy_classes = {
            StrategyType.MOMENTUM: Momentum,
            StrategyType.MEAN_REVERSION: MeanReversion,
            StrategyType.MEAN_REVERSION_RSI: MeanReversionRSI,
            StrategyType.STAT_ARB: StatArb,
        }
        return strategy_classes.get(strategy_type)
    
    
    def _initialize_parameters(self, overrides: Dict[str, Any]) -> None:
        """
        Initialize all parameters with proper priority handling.
        
        Priority (highest to lowest):
        1. Explicit overrides from kwargs
        2. Strategy-specific defaults from strategy class
        3. Common defaults from StrategyFactory
        """

        # Step 1: Load common defaults from StrategyFactory
        common_defaults = StrategyFactory.get_common_parameter_grid()
        for param_name, values in common_defaults.items():
            setattr(self, param_name, values)
        
        # Step 2: Set common defaults not in factory
        if not hasattr(self, 'tickers'):
            self.tickers = ['NVDA']
        if not hasattr(self, 'pairs'):
            self.pairs = [('NVDA', 'AMD')]
        
        # Step 3: Load strategy-specific defaults from strategy class
        strategy_class = self._get_strategy_class(self.strategy_type)
        if strategy_class:
            instance = strategy_class()
            strategy_defaults = instance.get_parameter_grid()
            for param_name, values in strategy_defaults.items():
                setattr(self, param_name, values)
        
        # Step 4: Initialize params for other strategies (prevents AttributeError)
        self._ensure_all_params_exist()
        
        # Step 5: Apply overrides from kwargs
        for attr_name, value in overrides.items():
            setattr(self, attr_name, value)
    
    
    def _ensure_all_params_exist(self) -> None:
        """
        Ensure all strategy parameters exist with sensible defaults.
        """
        for strategy_type in StrategyType:
            strategy_class = self._get_strategy_class(strategy_type)
            if strategy_class:
                instance = strategy_class()
                strategy_defaults = instance.get_parameter_grid()
                for param_name, values in strategy_defaults.items():
                    if not hasattr(self, param_name):
                        if isinstance(values, list) and len(values) > 0:
                            setattr(self, param_name, [values[0]])
                        else:
                            setattr(self, param_name, values)
    

    def set_strategy_type(self, strategy_type: StrategyType) -> None:
        """
        Set the strategy type for simulation.
        
        Args:
            strategy_type: The type of strategy to simulate
        """
        self.strategy_type = strategy_type
        logger.info(f"Strategy type set to: {strategy_type.value}")


    def set_tickers(self, tickers: List[str]) -> None:
        """
        Set the list of tickers to simulate.
        
        Args:
            tickers: List of ticker symbols
        """
        self.tickers = tickers
        logger.info(f"Tickers set to: {tickers}")


    def set_pairs(self, pairs: List[Tuple[str, str]]) -> None:
        """
        Set the list of pairs for statistical arbitrage.
        
        Args:
            pairs: List of tuples (ticker_a, ticker_b)
        """
        self.pairs = pairs
        logger.info(f"Pairs set to: {pairs}")


    def set_momentum_parameters(
        self,
        band_multipliers: List[float] = None,
        trade_frequencies: List[int] = None,
        target_volatilities: List[float] = None
    ) -> None:
        """
        Set parameter grids for momentum strategy.
        
        Args:
            band_multipliers: List of band multiplier values
            trade_frequencies: List of trade frequency values (minutes)
            target_volatilities: List of target volatility values
        """
        if band_multipliers is not None:
            self.band_multipliers = band_multipliers
        if trade_frequencies is not None:
            self.trade_frequencies = trade_frequencies
        if target_volatilities is not None:
            self.target_volatilities = target_volatilities


    def set_mean_reversion_parameters(
        self,
        zscore_lookbacks: List[int] = None,
        n_std_uppers: List[float] = None,
        n_std_lowers: List[float] = None,
        exit_thresholds: List[float] = None,
        trade_frequencies: List[int] = None,
        target_volatilities: List[float] = None
    ) -> None:
        """
        Set parameter grids for mean reversion strategy.
        
        Args:
            zscore_lookbacks: List of lookback periods for z-score calculation
            n_std_uppers: List of upper band standard deviation multipliers
            n_std_lowers: List of lower band standard deviation multipliers
            exit_thresholds: List of z-score exit threshold values
            trade_frequencies: List of trade frequency values (minutes)
            target_volatilities: List of target volatility values
        """
        if zscore_lookbacks is not None:
            self.zscore_lookbacks = zscore_lookbacks
        if n_std_uppers is not None:
            self.n_std_uppers = n_std_uppers
        if n_std_lowers is not None:
            self.n_std_lowers = n_std_lowers
        if exit_thresholds is not None:
            self.exit_thresholds = exit_thresholds
        if trade_frequencies is not None:
            self.trade_frequencies = trade_frequencies
        if target_volatilities is not None:
            self.target_volatilities = target_volatilities

    
    def set_mean_reversion_rsi_parameters(
        self,
        rsi_periods: List[int] = None,
        rsi_oversold_levels: List[float] = None,
        rsi_overbought_levels: List[float] = None,
        sma_periods: List[int] = None,
        trade_frequencies: List[int] = None,
        target_volatilities: List[float] = None
    ) -> None:
        """
        Set parameter grids for RSI mean reversion strategy.
        
        Args:
            rsi_periods: List of lookback periods for RSI calculation
            rsi_oversold_levels: List of oversold levels for RSI calculation
            rsi_overbought_levels: List of overbought levels for RSI calculation
            sma_periods: List of lookback periods for SMA calculation
            trade_frequencies: List of trade frequency values (minutes)
            target_volatilities: List of target volatility values
        """
        if rsi_periods is not None:
            self.rsi_periods = rsi_periods
        if rsi_oversold_levels is not None:
            self.rsi_oversold_levels = rsi_oversold_levels
        if rsi_overbought_levels is not None:
            self.rsi_overbought_levels = rsi_overbought_levels
        if sma_periods is not None:
            self.sma_periods = sma_periods
        if trade_frequencies is not None:
            self.trade_frequencies = trade_frequencies
        if target_volatilities is not None:
            self.target_volatilities = target_volatilities


    def set_stat_arb_parameters(
        self,
        pairs: List[Tuple[str, str]] = None,
        zscore_lookbacks: List[int] = None,
        entry_thresholds: List[float] = None,
        exit_thresholds: List[float] = None,
        use_dynamic_hedge_options: List[bool] = None,
        trade_frequencies: List[int] = None,
        target_volatilities: List[float] = None
    ) -> None:
        """
        Set parameter grids for statistical arbitrage strategy.
        
        Args:
            pairs: List of tuples (ticker_a, ticker_b) for pairs trading
            zscore_lookbacks: List of lookback periods for spread z-score
            entry_thresholds: List of z-score entry thresholds
            exit_thresholds: List of z-score exit thresholds
            use_dynamic_hedge_options: List of boolean values for dynamic hedge
            trade_frequencies: List of trade frequency values (minutes)
            target_volatilities: List of target volatility values
        """
        if pairs is not None:
            self.pairs = pairs
        if zscore_lookbacks is not None:
            self.stat_arb_zscore_lookbacks = zscore_lookbacks
        if entry_thresholds is not None:
            self.entry_thresholds = entry_thresholds
        if exit_thresholds is not None:
            self.stat_arb_exit_thresholds = exit_thresholds
        if use_dynamic_hedge_options is not None:
            self.use_dynamic_hedge_options = use_dynamic_hedge_options
        if trade_frequencies is not None:
            self.trade_frequencies = trade_frequencies
        if target_volatilities is not None:
            self.target_volatilities = target_volatilities


    def load_risk_free_rate(self) -> None:
        """
        Load the risk-free rate data.
        """
        try:
            risk_free_rate = pd.read_csv('data/dtb3.csv')
            risk_free_rate['date'] = pd.to_datetime(risk_free_rate['date'])
            risk_free_rate = risk_free_rate[risk_free_rate['date'] >= self.constants.start_date]
            risk_free_rate = risk_free_rate.set_index('date')
            risk_free_rate.index = risk_free_rate.index.normalize()
            
            # Convert from percentage to decimal
            self.risk_free_rate_series = risk_free_rate['rate'] / 100.0
            
            logger.info(f"Loaded risk-free rate data: {len(self.risk_free_rate_series)} observations")
            logger.info(f"Average risk-free rate: {self.risk_free_rate_series.mean()*100:.2f}%")
            
        except FileNotFoundError:
            logger.warning("Risk-free rate file 'dtb3.csv' not found. Using 0.")
            self.risk_free_rate_series = None
        except Exception as e:
            logger.warning(f"Error loading risk-free rate: {e}")
            self.risk_free_rate_series = None


    def get_market_calendar(self) -> Dict:
        """
        Get the market calendar from constants.
        """
        return {
            'holidays': self.constants.holidays,
            'early_closes': {
                date: time
                for year_data in self.constants.holidays.values()
                for date, time in year_data['early_closes'].items()
            }
        }


    def load_or_fetch_data(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load existing data or fetch new data for a ticker.
        Uses cache to avoid redundant loading for pairs trading.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Tuple of (intraday_df, daily_df)
        """
        # Check cache first
        if ticker in self.ticker_data_cache:
            logger.info(f"Using cached data for {ticker}")
            return self.ticker_data_cache[ticker]
        
        # Create directories
        os.makedirs(f'data/{ticker}', exist_ok=True)
        
        # Check for existing files
        existing_intra = [
            f for f in os.listdir(f'data/{ticker}') 
            if f.startswith(f'{ticker}_intra_feed') 
            and f.endswith('.csv')
        ]
        existing_daily = [
            f for f in os.listdir(f'data/{ticker}') 
            if f.startswith(f'{ticker}_daily_feed') 
            and f.endswith('.csv')
        ]
        
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if existing_intra and existing_daily:
            # Sort by modification time
            existing_intra.sort(key=lambda x: os.path.getmtime(f'data/{ticker}/{x}'), reverse=True)
            existing_daily.sort(key=lambda x: os.path.getmtime(f'data/{ticker}/{x}'), reverse=True)
            
            logger.info(f"Saved data found for {ticker}... Loading existing data...")
            
            try:
                df = pd.read_csv(f"data/{ticker}/{existing_intra[0]}", index_col=0)
                df_daily = pd.read_csv(f"data/{ticker}/{existing_daily[0]}", index_col=0)
                
                if len(df) > 0 and len(df_daily) > 0:
                    logger.info(f"Loaded {len(df)} intraday records, {len(df_daily)} daily records")
                    # Cache the data
                    self.ticker_data_cache[ticker] = (df, df_daily)
                    return df, df_daily
                else:
                    raise ValueError("Empty data")
                    
            except Exception as e:
                logger.warning(f"Error loading data for {ticker}: {e}")
        
        # Fetch new data
        logger.info(f"Fetching new data for {ticker}...")
        fetcher = DataFetcher(ticker)
        df, df_daily = fetcher.process_data()
        
        if not df.empty:
            df.to_csv(f"data/{ticker}/{ticker}_intra_feed_{version}.csv")
            df_daily.to_csv(f"data/{ticker}/{ticker}_daily_feed_{version}.csv")
            logger.info(f"  Saved {len(df)} intraday records, {len(df_daily)} daily records")
            # Cache the data
            self.ticker_data_cache[ticker] = (df, df_daily)
        
        return df, df_daily


    def precompute_ticker_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pre-compute data that's shared across all parameter combinations for a ticker.
        This reduces redundant calculations in the backtest loop.
        
        Args:
            df: Intraday DataFrame for a ticker
            
        Returns:
            Dictionary with pre-computed data structures
        """
        df_with_day = df.copy()
        df_with_day['day'] = pd.to_datetime(df_with_day['day']).dt.normalize()
        
        precomputed = {
            'days': df_with_day['day'].unique(),
            'first_opens': df_with_day.groupby('day')['open'].first().to_dict(),
            'last_closes': df_with_day.groupby('day')['close'].last().to_dict(),
            'dividends': df_with_day.groupby('day')['dividend'].first().to_dict() if 'dividend' in df_with_day.columns else {},
            'daily_volatility': df_with_day.groupby('day')['ticker_dvol'].first().to_dict(),
        }
        
        return precomputed


    def _calculate_total_combinations(self) -> int:
        """
        Calculate total number of parameter combinations based on strategy type.
        
        Returns:
            Total number of combinations
        """
        if self.strategy_type == StrategyType.MOMENTUM:
            return (len(self.tickers) * len(self.band_multipliers) * 
                    len(self.trade_frequencies) * len(self.target_volatilities) *
                    len(self.stop_loss_pcts) * len(self.take_profit_pcts) *
                    len(self.max_drawdown_pcts))
        elif self.strategy_type == StrategyType.MEAN_REVERSION:
            return (len(self.tickers) * len(self.zscore_lookbacks) * 
                    len(self.n_std_uppers) * len(self.n_std_lowers) *
                    len(self.exit_thresholds) * len(self.trade_frequencies) * 
                    len(self.target_volatilities) * len(self.stop_loss_pcts) *
                    len(self.take_profit_pcts) * len(self.max_drawdown_pcts))
        elif self.strategy_type == StrategyType.MEAN_REVERSION_RSI:
            return (len(self.tickers) * len(self.rsi_periods) *
                    len(self.rsi_oversold_levels) * len(self.rsi_overbought_levels) *
                    len(self.sma_periods) * len(self.trade_frequencies) *
                    len(self.target_volatilities) * len(self.stop_loss_pcts) *
                    len(self.take_profit_pcts) * len(self.max_drawdown_pcts))
        elif self.strategy_type == StrategyType.STAT_ARB:
            return (len(self.pairs) * len(self.stat_arb_zscore_lookbacks) *
                    len(self.entry_thresholds) * len(self.stat_arb_exit_thresholds) *
                    len(self.use_dynamic_hedge_options) * len(self.trade_frequencies) *
                    len(self.target_volatilities) * len(self.stop_loss_pcts) *
                    len(self.take_profit_pcts) * len(self.max_drawdown_pcts))
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")


    def _generate_momentum_tasks(
        self,
        ticker: str,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        market_calendar: Dict
    ) -> List[Dict[str, Any]]:
        """
        Generate task dictionaries for momentum strategy parameter combinations.
        
        Args:
            ticker: Ticker symbol
            df: Intraday DataFrame
            df_daily: Daily DataFrame
            market_calendar: Market calendar dictionary
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        for band_mult in self.band_multipliers:
            for trade_freq in self.trade_frequencies:
                for target_vol in self.target_volatilities:
                    for stop_loss_pct in self.stop_loss_pcts:
                        for take_profit_pct in self.take_profit_pcts:
                            for max_drawdown_pct in self.max_drawdown_pcts:
                                tasks.append({
                                    'ticker': ticker,
                                    'df': df,
                                    'df_daily': df_daily,
                                    'strategy_type': StrategyType.MOMENTUM,
                                    'band_mult': band_mult,
                                    'trade_freq': trade_freq,
                                    'target_vol': target_vol,
                                    'stop_loss_pct': stop_loss_pct,
                                    'take_profit_pct': take_profit_pct,
                                    'max_drawdown_pct': max_drawdown_pct,
                                    'risk_free_rate_series': self.risk_free_rate_series,
                                    'market_calendar': market_calendar
                                })

        return tasks


    def _generate_mean_reversion_tasks(
        self,
        ticker: str,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        market_calendar: Dict
    ) -> List[Dict[str, Any]]:
        """
        Generate task dictionaries for mean reversion strategy parameter combinations.
        
        Args:
            ticker: Ticker symbol
            df: Intraday DataFrame
            df_daily: Daily DataFrame
            market_calendar: Market calendar dictionary
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        for zscore_lookback in self.zscore_lookbacks:
            for n_std_upper in self.n_std_uppers:
                for n_std_lower in self.n_std_lowers:
                    for exit_threshold in self.exit_thresholds:
                        for trade_freq in self.trade_frequencies:
                            for target_vol in self.target_volatilities:
                                for stop_loss_pct in self.stop_loss_pcts:
                                    for take_profit_pct in self.take_profit_pcts:
                                        for max_drawdown_pct in self.max_drawdown_pcts:
                                            tasks.append({
                                                'ticker': ticker,
                                                'df': df,
                                                'df_daily': df_daily,
                                                'strategy_type': StrategyType.MEAN_REVERSION,
                                                'zscore_lookback': zscore_lookback,
                                                'n_std_upper': n_std_upper,
                                                'n_std_lower': n_std_lower,
                                                'exit_threshold': exit_threshold,
                                                'trade_freq': trade_freq,
                                                'target_vol': target_vol,
                                                'stop_loss_pct': stop_loss_pct,
                                                'take_profit_pct': take_profit_pct,
                                                'max_drawdown_pct': max_drawdown_pct,
                                                'risk_free_rate_series': self.risk_free_rate_series,
                                                'market_calendar': market_calendar
                                            })
        return tasks

    
    def _generate_mean_reversion_rsi_tasks(
        self,
        ticker: str,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        market_calendar: Dict
    ) -> List[Dict[str, Any]]:
        """
        Generate task dictionaries for RSI mean reversion strategy.
        
        Args:
            ticker: Ticker symbol
            df: Intraday DataFrame
            df_daily: Daily DataFrame
            market_calendar: Market calendar dictionary
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        for rsi_period in self.rsi_periods:
            for rsi_oversold in self.rsi_oversold_levels:
                for rsi_overbought in self.rsi_overbought_levels:
                    for sma_period in self.sma_periods:
                        for trade_freq in self.trade_frequencies:
                            for target_vol in self.target_volatilities:
                                for stop_loss_pct in self.stop_loss_pcts:
                                    for take_profit_pct in self.take_profit_pcts:
                                        for max_drawdown_pct in self.max_drawdown_pcts:
                                            tasks.append({
                                                'ticker': ticker,
                                                'df': df,
                                                'df_daily': df_daily,
                                                'strategy_type': StrategyType.MEAN_REVERSION_RSI,
                                                'rsi_period': rsi_period,
                                                'rsi_oversold': rsi_oversold,
                                                'rsi_overbought': rsi_overbought,
                                                'sma_period': sma_period,
                                                'exit_on_sma_cross': True,
                                                'trade_freq': trade_freq,
                                                'target_vol': target_vol,
                                                'stop_loss_pct': stop_loss_pct,
                                                'take_profit_pct': take_profit_pct,
                                                'max_drawdown_pct': max_drawdown_pct,
                                                'risk_free_rate_series': self.risk_free_rate_series,
                                                'market_calendar': market_calendar
                                            })
        return tasks


    def _generate_stat_arb_tasks(
        self,
        ticker_a: str,
        ticker_b: str,
        df_a: pd.DataFrame,
        df_daily_a: pd.DataFrame,
        df_b: pd.DataFrame,
        df_daily_b: pd.DataFrame,
        market_calendar: Dict
    ) -> List[Dict[str, Any]]:
        """
        Generate task dictionaries for statistical arbitrage parameter combinations.
        
        Args:
            ticker_a: Primary ticker symbol
            ticker_b: Secondary ticker symbol
            df_a: Intraday DataFrame for ticker A
            df_daily_a: Daily DataFrame for ticker A
            df_b: Intraday DataFrame for ticker B
            df_daily_b: Daily DataFrame for ticker B
            market_calendar: Market calendar dictionary
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        for zscore_lookback in self.stat_arb_zscore_lookbacks:
            for entry_threshold in self.entry_thresholds:
                for exit_threshold in self.stat_arb_exit_thresholds:
                    for use_dynamic_hedge in self.use_dynamic_hedge_options:
                        for trade_freq in self.trade_frequencies:
                            for target_vol in self.target_volatilities:
                                for stop_loss_pct in self.stop_loss_pcts:
                                    for take_profit_pct in self.take_profit_pcts:
                                        for max_drawdown_pct in self.max_drawdown_pcts:
                                            tasks.append({
                                                'ticker': ticker_a,
                                                'ticker_b': ticker_b,
                                                'df': df_a,
                                                'df_daily': df_daily_a,
                                                'df_b': df_b,
                                                'df_b_daily': df_daily_b,
                                                'strategy_type': StrategyType.STAT_ARB,
                                                'zscore_lookback': zscore_lookback,
                                                'entry_threshold': entry_threshold,
                                                'exit_threshold': exit_threshold,
                                                'use_dynamic_hedge': use_dynamic_hedge,
                                                'trade_freq': trade_freq,
                                                'target_vol': target_vol,
                                                'stop_loss_pct': stop_loss_pct,
                                                'take_profit_pct': take_profit_pct,
                                                'max_drawdown_pct': max_drawdown_pct,
                                                'risk_free_rate_series': self.risk_free_rate_series,
                                                'market_calendar': market_calendar
                                            })
        return tasks


    def run_sequential(self) -> pd.DataFrame:
        """
        Run the simulation sequentially (for debugging or when parallelism isn't needed).
        
        Returns:
            DataFrame with all simulation results
        """
        logger.info("="*70)
        logger.info(f"STARTING SEQUENTIAL SIMULATION ({self.strategy_type.value.upper()})")
        logger.info("="*70)
        
        # Load risk-free rate
        self.load_risk_free_rate()
        
        # Calculate total combinations
        total_combinations = self._calculate_total_combinations()
        logger.info(f"Total parameter combinations: {total_combinations}")
        
        market_calendar = self.get_market_calendar()
        
        # Generate tasks based on strategy type
        tasks = []
        
        if self.strategy_type == StrategyType.STAT_ARB:
            # Load data for all pairs
            for ticker_a, ticker_b in self.pairs:
                logger.info(f"Loading data for pair {ticker_a}/{ticker_b}...")
                df_a, df_daily_a = self.load_or_fetch_data(ticker_a)
                df_b, df_daily_b = self.load_or_fetch_data(ticker_b)
                
                if not df_a.empty and not df_b.empty:
                    tasks.extend(self._generate_stat_arb_tasks(
                        ticker_a, ticker_b, df_a, df_daily_a, df_b, df_daily_b, market_calendar
                    ))
                else:
                    logger.error(f"Failed to load data for pair {ticker_a}/{ticker_b}")
        else:
            # Load data for single tickers
            ticker_data = {}
            for ticker in self.tickers:
                logger.info(f"Loading data for {ticker}...")
                df, df_daily = self.load_or_fetch_data(ticker)
                if not df.empty:
                    ticker_data[ticker] = (df, df_daily)
                else:
                    logger.error(f"Failed to load data for {ticker}")
            
            for ticker in self.tickers:
                if ticker not in ticker_data:
                    continue
                    
                df, df_daily = ticker_data[ticker]
                
                if self.strategy_type == StrategyType.MOMENTUM:
                    tasks.extend(self._generate_momentum_tasks(ticker, df, df_daily, market_calendar))
                elif self.strategy_type == StrategyType.MEAN_REVERSION:
                    tasks.extend(self._generate_mean_reversion_tasks(ticker, df, df_daily, market_calendar))
                elif self.strategy_type == StrategyType.MEAN_REVERSION_RSI:
                    tasks.extend(self._generate_mean_reversion_rsi_tasks(ticker, df, df_daily, market_calendar))
        
        # Run simulations
        completed = 0
        for task in tasks:
            completed += 1
            
            if completed % 50 == 0:
                logger.info(f"[{completed}/{total_combinations}] Progress update...")
            
            result = _run_single_backtest_task(task)
            
            if result is not None:
                self.results.append(result)
        
        return pd.DataFrame(self.results)


    def run(self, parallel: bool = True) -> pd.DataFrame:
        """
        Run the full simulation across all parameter combinations.
        Uses parallel processing by default for improved performance.
        
        Args:
            parallel: Whether to use parallel processing (default True)
        
        Returns:
            DataFrame with all simulation results
        """
        if not parallel:
            return self.run_sequential()
        
        logger.info("-"*70)
        logger.info(f"Starting Parallel Simulation ({self.strategy_type.value.upper()})")
        logger.info(f"Using {self.n_workers} worker processes")
        logger.info("-"*70 + "\n")
        
        # Load risk-free rate
        self.load_risk_free_rate()
        
        # Calculate total combinations
        total_combinations = self._calculate_total_combinations()
        logger.info(f"Total parameter combinations: {total_combinations}")
        
        # Log strategy-specific parameters using Outputs class
        self.outputs.log_simulation_parameters(self.strategy_type, self, logger)
        
        logger.info("-" * 70 + "\n")
            
        market_calendar = self.get_market_calendar()
        
        # Prepare all tasks
        tasks = []
        
        if self.strategy_type == StrategyType.STAT_ARB:
            # Load data for all pairs
            for ticker_a, ticker_b in self.pairs:
                logger.info(f"Loading data for pair {ticker_a}/{ticker_b}...")
                df_a, df_daily_a = self.load_or_fetch_data(ticker_a)
                df_b, df_daily_b = self.load_or_fetch_data(ticker_b)
                
                if not df_a.empty and not df_b.empty:
                    tasks.extend(self._generate_stat_arb_tasks(
                        ticker_a, ticker_b, df_a, df_daily_a, df_b, df_daily_b, market_calendar
                    ))
                else:
                    logger.error(f"Failed to load data for pair {ticker_a}/{ticker_b}")
        else:
            # Load data for single tickers
            ticker_data = {}
            for ticker in self.tickers:
                logger.info(f"Loading data for {ticker}...")
                df, df_daily = self.load_or_fetch_data(ticker)
                if not df.empty:
                    ticker_data[ticker] = (df, df_daily)
                else:
                    logger.error(f"Failed to load data for {ticker}")
            
            for ticker in self.tickers:
                if ticker not in ticker_data:
                    continue
                    
                df, df_daily = ticker_data[ticker]
                
                if self.strategy_type == StrategyType.MOMENTUM:
                    tasks.extend(self._generate_momentum_tasks(ticker, df, df_daily, market_calendar))
                elif self.strategy_type == StrategyType.MEAN_REVERSION:
                    tasks.extend(self._generate_mean_reversion_tasks(ticker, df, df_daily, market_calendar))
                elif self.strategy_type == StrategyType.MEAN_REVERSION_RSI:
                    tasks.extend(self._generate_mean_reversion_rsi_tasks(ticker, df, df_daily, market_calendar))
        
        logger.info(f"Prepared {len(tasks)} tasks for parallel execution")
        
        # Run tasks in parallel
        start_time = time.time()
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(_run_single_backtest_task, task): task 
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                completed += 1
                result = future.result()
                
                if result is not None:
                    self.results.append(result)
                
                # Progress update every 50 completions
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = (len(tasks) - completed) / rate if rate > 0 else 0
                    logger.info(f"[{completed}/{len(tasks)}] "
                               f"Elapsed: {elapsed:.1f}s, "
                               f"Rate: {rate:.1f} tasks/s, "
                               f"ETA: {remaining:.1f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Completed {len(tasks)} backtests in {total_time:.1f} seconds")
        logger.info(f"Average time per backtest: {total_time/len(tasks)*1000:.1f}ms")
        
        return pd.DataFrame(self.results)


    def get_best_by_ticker(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the best strategy configuration for each ticker based on Sharpe Ratio.
        
        Args:
            results_df: Full results DataFrame
            
        Returns:
            DataFrame with best configuration per ticker
        """
        # Get index of max Sharpe Ratio for each ticker
        idx = results_df.groupby('Ticker')['Sharpe Ratio (Strategy)'].idxmax()
        return results_df.loc[idx]


    def get_best_by_pair(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the best strategy configuration for each pair based on Sharpe Ratio.
        For statistical arbitrage strategy.
        
        Args:
            results_df: Full results DataFrame
            
        Returns:
            DataFrame with best configuration per pair
        """
        if 'Ticker B' not in results_df.columns:
            return self.get_best_by_ticker(results_df)
        
        # Create pair identifier
        results_df['Pair'] = results_df['Ticker'] + '/' + results_df['Ticker B']
        idx = results_df.groupby('Pair')['Sharpe Ratio (Strategy)'].idxmax()
        return results_df.loc[idx]


    def get_best_by_metric(
        self, 
        results_df: pd.DataFrame, 
        metric: str = 'Sharpe Ratio (Strategy)'
    ) -> pd.DataFrame:
        """
        Get the best strategy configuration for each ticker based on a specified metric.
        
        Args:
            results_df: Full results DataFrame
            metric: Column name to optimize for
            
        Returns:
            DataFrame with best configuration per ticker
        """
        idx = results_df.groupby('Ticker')[metric].idxmax()
        return results_df.loc[idx]


def main():
    """
    Main entry point for the simulation.
    Runs momentum, mean reversion, or statistical arbitrage strategies.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run trading strategy simulations')
    parser.add_argument(
        '--strategy', 
        type=str, 
        choices=['momentum', 'mean_reversion', 'mean_reversion_rsi', 'stat_arb', 'all'],
        default='momentum',
        help='Strategy type to simulate (default: momentum)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Use parallel processing (default: True)'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        default=False,
        help='Use sequential processing (for debugging)'
    )
    
    args = parser.parse_args()
    
    # Start time
    start_time = time.time()
    
    # Create output directory
    os.makedirs('simulations', exist_ok=True)
    
    strategies_to_run = []
    if args.strategy == 'all':
        strategies_to_run = [
            StrategyType.MOMENTUM, 
            StrategyType.MEAN_REVERSION, 
            StrategyType.MEAN_REVERSION_RSI,
            StrategyType.STAT_ARB
        ]
    elif args.strategy == 'momentum':
        strategies_to_run = [StrategyType.MOMENTUM]
    elif args.strategy == 'mean_reversion':
        strategies_to_run = [StrategyType.MEAN_REVERSION]
    elif args.strategy == 'mean_reversion_rsi':
        strategies_to_run = [StrategyType.MEAN_REVERSION_RSI]
    elif args.strategy == 'stat_arb':
        strategies_to_run = [StrategyType.STAT_ARB]
    
    all_results = []
    
    for strategy_type in strategies_to_run:
        
        # Run simulation
        sim = Simulation(strategy_type=strategy_type)
        
        # Use parallel=True for speed, parallel=False for debugging
        use_parallel = not args.sequential
        results_df = sim.run(parallel=use_parallel)
        
        if results_df.empty:
            logger.error(f"No results generated for {strategy_type.value}!")
            continue
        
        # Save full results
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs(f'simulations/{strategy_type.value}', exist_ok=True)
        results_path = f"simulations/{strategy_type.value}/simulation_results_{version}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Full results saved to: {results_path}")

        # Print summary statistics
        if strategy_type == StrategyType.STAT_ARB:
            outputs.print_simulation_summary(
                strategy_type, 
                results_df, 
                pairs=sim.pairs
            )
        else:
            outputs.print_simulation_summary(
                strategy_type, 
                results_df, 
                tickers=sim.tickers
            )
        
        # Get best strategies
        if strategy_type == StrategyType.STAT_ARB:
            best_df = sim.get_best_by_pair(results_df)
        else:
            best_df = sim.get_best_by_ticker(results_df)
        
        # Print best results using Outputs class
        outputs.print_best_results(strategy_type, best_df)
            
                
        # Save best results
        best_path = f"simulations/{strategy_type.value}/best_strategy_{version}.csv"
        best_df.to_csv(best_path, index=False)
        logger.info(f"Best strategies saved to: {best_path}")
        
        all_results.append(results_df)
    
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
