


"""
┌───────────────────┬──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│ Parameter         │ Purpose                                      │ Impact                                       │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Band Multiplier   │ Controls how far price must move from open   │ Higher = fewer trades, only catches strong   │
│                   │ (in units of sigma) to trigger a signal      │ moves; Lower = more trades, more whipsaws    │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Target Volatility │ Scales position size to achieve a consistent │ Higher = larger positions, more risk/reward; │
│                   │ portfolio volatility                         │ Lower = smaller positions, smoother curve    │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Slippage Factor   │ Base cost (in bps) for market impact when    │ Higher = more conservative P&L estimates;    │
│                   │ executing trades                             │ affects profitability of frequent trading    │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ VaR Confidence    │ Confidence level for daily Value-at-Risk     │ Higher = more extreme tail risk measure;     │
│                   │ calculation (e.g., 95%)                      │ used for monitoring, not position sizing     │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Max Drawdown %    │ Circuit breaker - closes all positions if    │ Tighter = preserves capital but may exit     │
│                   │ portfolio drops this much from peak          │ early; Looser = more room but larger losses  │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Stop Loss %       │ Exits individual trades when loss exceeds    │ Tighter = limits losses but more whipsaws;   │
│                   │ this threshold                               │ Looser = gives trades room to breathe        │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Take Profit %     │ Exits individual trades when gain exceeds    │ Lower = locks in gains but caps upside;      │
│                   │ this threshold                               │ Higher = lets winners run, risks giveback    │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Max Leverage      │ Caps position size regardless of vol-target  │ Higher = amplifies returns and risk;         │
│                   │ calculation                                  │ Lower = limits exposure when signals strong  │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Z-Score Lookback  │ Number of periods for moving average and     │ Shorter = more responsive but noisier;       │
│ (Mean Reversion)  │ standard deviation calculation               │ Longer = smoother but slower to react        │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ N Std Upper       │ Number of std devs above mean for sell       │ Higher = fewer signals, more extreme moves;  │
│ (Mean Reversion)  │ signal (overbought threshold)                │ Lower = more signals, smaller deviations     │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ N Std Lower       │ Number of std devs below mean for buy        │ Higher = fewer signals, more extreme moves;  │
│ (Mean Reversion)  │ signal (oversold threshold)                  │ Lower = more signals, smaller deviations     │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Entry Threshold   │ Z-score threshold to enter spread trade      │ Higher = fewer trades, stronger signals;     │
│ (Stat Arb)        │                                              │ Lower = more trades, weaker signals          │
├───────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Exit Threshold    │ Z-score threshold to exit spread trade       │ Closer to 0 = exit at mean reversion;        │
│ (Stat Arb)        │                                              │ Further = hold longer for more profit        │
└───────────────────┴──────────────────────────────────────────────┴──────────────────────────────────────────────┘
"""

import pickle
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from typing import Dict, Tuple, Any
from numba import njit
from constants import Constants
from utils import Utils
from strats.momentum.momentum import Momentum
from strats.mean_reversion.mean_reversion import MeanReversion
from strats.mean_reversion.mean_reversion_rsi import MeanReversionRSI
from strats.stat_arb.stat_arb import StatArb
from strats.ml.supervised import SupervisedStrategy
from ml.preprocessing import PreProcessing
from strategies import (
    StrategyType,
    StrategyFactory,
    simulate_positions_numba,
    calculate_portfolio_values_numba,
)


# Ignore all warnings
warnings.filterwarnings('ignore')

constants = Constants()
utils = Utils()


class BackTest:
    """
    Backtest class for trading strategies.

    Key optimizations:

        1. Vectorized signal generation
        2. Numba JIT compilation for position simulation loops
        3. Pre-computed data structures
        4. Reduced DataFrame operations
        5. Optional verbose output

    Supported strategies:
    
        - Momentum (band breakout)
        - Mean Reversion (z-score based)
        - Mean Reversion (RSI + SMA based)
        - Statistical Arbitrage (pairs trading)
    """
    
    def __init__(self, ticker: str = None, ticker_b: str = None):
        """
        Initialize the BackTest class.
        
        Args:
            ticker: Optional ticker symbol for primary asset. If not provided, uses constants.ticker
            ticker_b: Optional ticker symbol for secondary asset (pairs trading)
        """
        self.start_date = constants.start_date
        self.end_date = constants.end_date
        self.ticker = ticker if ticker is not None else getattr(constants, 'ticker', 'SPY')
        self.ticker_b = ticker_b
        self.benchmark_ticker = constants.benchmark_ticker
        self.timezone = constants.timezone
        self.enforce_rate_limit = constants.enforce_rate_limit
        self.limit = constants.limit
        self.multiplier = constants.multiplier
        self.period = constants.period
        self.holidays = constants.holidays

        # ML parameters
        self.ml_model_type = 'random_forest'
        self.ml_model = None
        self.ml_scalers = None
        self.ml_feature_names = None
        self._ml_preprocessor = None
        
        # Store risk-free rate series (will be set externally)
        self.risk_free_rate_series = None
        
        # Store AUM_0 for use across methods (can be overridden by config)
        self.AUM_0 = constants.AUM_0
        
        # Trade statistics (populated after backtest runs)
        self.total_trades = 0
        self.buy_trades = 0
        self.sell_trades = 0
        self.stop_losses = 0
        self.profit_takes = 0
        self.max_drawdown_breaches = 0
        self._last_size_factor = 0.0
        
        # Strategy type (default to momentum for backward compatibility)
        self.strategy_type = StrategyType.MOMENTUM


    def set_risk_free_rate(self, risk_free_rate_series: pd.Series):
        """
        Set the risk-free rate series for Sharpe ratio calculations.
        
        Args:
            risk_free_rate_series: Series indexed by date with annualized risk-free rates
        """
        self.risk_free_rate_series = risk_free_rate_series


    def _precompute_daily_data(
        self, 
        df: pd.DataFrame, 
        all_days: np.ndarray
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Pre-compute all daily data structures to avoid repeated groupby operations.
        
        Args:
            df: Intraday DataFrame
            all_days: Array of unique trading days
            
        Returns:
            Dictionary mapping days to pre-computed data
        """
        daily_groups = df.groupby('day')
        precomputed = {}
        
        for day in all_days:
            if day in daily_groups.groups:
                day_data = daily_groups.get_group(day)
                precomputed[day] = {
                    'data': day_data,
                    'open': day_data['open'].iloc[0],
                    'close_prices': day_data['close'].values.astype(np.float64),
                    'volumes': day_data['volume'].values.astype(np.float64),
                    'vwap': day_data['vwap'].values.astype(np.float64),
                    'sigma_open': day_data['sigma_open'].values.astype(np.float64),
                    'ticker_dvol': day_data['ticker_dvol'].iloc[0],
                    'min_from_open': day_data['min_from_open'].values.astype(np.float64),
                    'dividend': day_data['dividend'].iloc[0] if 'dividend' in day_data.columns else 0.0
                }
        
        return precomputed


    def _generate_signals_momentum(
        self,
        close_prices: np.ndarray,
        vwap: np.ndarray,
        sigma_open: np.ndarray,
        reference_high: float,
        reference_low: float,
        config: dict
    ) -> np.ndarray:
        """
        Generate signals using the momentum (band breakout) strategy.
        
        Args:
            close_prices: Array of close prices
            vwap: Array of VWAP values
            sigma_open: Array of sigma values for band calculation
            reference_high: Reference price for upper band
            reference_low: Reference price for lower band
            config: Strategy configuration dictionary
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
        """
        band_mult = config['band_mult']
        return Momentum.generate_signals(
            close_prices, vwap, sigma_open,
            reference_high, reference_low, band_mult
        )


    def _generate_signals_mean_reversion(
        self,
        close_prices: np.ndarray,
        config: dict
    ) -> np.ndarray:
        """
        Generate signals using the mean reversion (z-score) strategy.
        
        Args:
            close_prices: Array of close prices
            config: Strategy configuration dictionary containing:
                - zscore_lookback: Lookback period for MA and StdDev
                - n_std_upper: Number of std devs for upper band
                - n_std_lower: Number of std devs for lower band
                - exit_threshold: Z-score threshold for exit (default 0.0)
            
        Returns:
            Array of signals: 1 (long/buy), -1 (exit/sell), 0 (hold)
        """
        zscore_lookback = config.get('zscore_lookback', 20)
        n_std_upper = config.get('n_std_upper', 2.0)
        n_std_lower = config.get('n_std_lower', 2.0)
        exit_threshold = config.get('exit_threshold', 0.0)
        
        return MeanReversion.generate_signals(
            close_prices,
            zscore_lookback,
            n_std_upper,
            n_std_lower,
            exit_threshold
        )

    
    def _generate_signals_mean_reversion_rsi(
        self,
        close_prices: np.ndarray,
        config: dict
    ) -> np.ndarray:
        """
        Generate signals using the RSI + SMA mean reversion strategy.
        
        Args:
            close_prices: Array of close prices
            config: Strategy configuration dictionary
            
        Returns:
            Array of signals: 1 (long/buy), -1 (exit/sell), 0 (hold)
        """
        rsi_period = config.get('rsi_period', 10)
        rsi_oversold = config.get('rsi_oversold', 30.0)
        rsi_overbought = config.get('rsi_overbought', 70.0)
        sma_period = config.get('sma_period', 200)
        exit_on_sma_cross = config.get('exit_on_sma_cross', True)
        
        return MeanReversionRSI.generate_signals(
            close_prices,
            rsi_period,
            rsi_oversold,
            rsi_overbought,
            sma_period,
            exit_on_sma_cross
        )


    def _generate_signals_stat_arb(
        self,
        close_prices_a: np.ndarray,
        close_prices_b: np.ndarray,
        config: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate signals using the statistical arbitrage (pairs trading) strategy.
        
        Args:
            close_prices_a: Array of close prices for asset A
            close_prices_b: Array of close prices for asset B
            config: Strategy configuration dictionary containing:
                - zscore_lookback: Lookback period for spread MA and StdDev
                - entry_threshold: Z-score threshold for entry
                - exit_threshold: Z-score threshold for exit (default 0.0)
                - use_dynamic_hedge: Whether to use dynamic hedge ratio
                - hedge_ratio: Fixed hedge ratio (if not using dynamic)
            
        Returns:
            Tuple of (signals, zscore, spread, hedge_ratios)
        """
        zscore_lookback = config.get('zscore_lookback', 60)
        entry_threshold = config.get('entry_threshold', 2.0)
        exit_threshold = config.get('exit_threshold', 0.0)
        use_dynamic_hedge = config.get('use_dynamic_hedge', True)
        hedge_ratio = config.get('hedge_ratio', None)
        
        return StatArb.generate_signals(
            close_prices_a,
            close_prices_b,
            zscore_lookback,
            entry_threshold,
            exit_threshold,
            hedge_ratio,
            use_dynamic_hedge
        )


    def _generate_ml_features(
        self,
        current_day_info: Dict[str, Any],
        precomputed_data: Dict[Any, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate ML features from precomputed intraday data for a single day.
        
        Delegates to PreProcessing.generate_features_from_arrays() to ensure
        feature consistency between training and inference.
        
        Args:
            current_day_info: Dictionary containing current day's precomputed data:
                - close_prices: np.ndarray of close prices
                - volumes: np.ndarray of volumes  
                - vwap: np.ndarray of VWAP values
                - sigma_open: np.ndarray of sigma values
                - open: float, opening price
                - ticker_dvol: float, daily volatility
            precomputed_data: Dictionary of all precomputed daily data (unused,
                             reserved for future multi-day feature generation)
            
        Returns:
            DataFrame with features matching the trained model's expected input
        """
        # Lazy initialization of preprocessor
        if not hasattr(self, '_ml_preprocessor') or self._ml_preprocessor is None:
            self._ml_preprocessor = PreProcessing(min_lookback=0)
        
        # Generate features using the shared preprocessing logic
        features_df = self._ml_preprocessor.generate_features_from_arrays(
            current_day_info=current_day_info,
            feature_names=self.ml_feature_names
        )
        
        return features_df

    
    def _generate_ml_signals(
        self,
        config: Dict[str, Any],
        current_day_info: Dict[str, Any],
        precomputed_data: Dict[Any, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Generate ML signals from precomputed intraday data for a single day.
        
        Args:
            config: Configuration dictionary
            current_day_info: Dictionary containing current day's precomputed data
            precomputed_data: Dictionary of all precomputed daily data
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
        """

        if self.ml_model is None:
            model_path = config.get('model_path')
            scaler_path = config.get('scaler_path')
            features_path = config.get('features_path')
            
            with open(model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            
            # Extract the scalers dict from the saved structure
            # The saved format is {'scaler_type': ..., 'scalers': {...}, 'feature_columns': [...]}
             # Fallback to entire object if format is different
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
                
                if isinstance(scaler_data, dict) and 'scalers' in scaler_data:
                    self.ml_scalers = scaler_data['scalers']
                else:
                    self.ml_scalers = scaler_data 
            
            with open(features_path, 'rb') as f:
                self.ml_feature_names = pickle.load(f)
        
        # Use current day data to generate features
        day_features = self._generate_ml_features(current_day_info, precomputed_data)
        
        signals = SupervisedStrategy.generate_signals_from_probabilities(
            day_features,
            self.ml_model,
            self.ml_scalers,
            self.ml_feature_names,
            config.get('window_size', 30),
            config.get('buy_threshold', 0.6),
            config.get('sell_threshold', 0.4)
        )

        return signals


    def backtest(
        self,
        df: pd.DataFrame, 
        ticker_daily_data: pd.DataFrame, 
        all_days: list,
        config: dict,
        market_calendar: dict = None,
        risk_free_rate_series: pd.Series = None,
        verbose: bool = False,
        df_b: pd.DataFrame = None,
        ticker_b_daily_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Optimized backtesting function for trading strategies.
        
        Uses vectorized operations and Numba JIT compilation for improved performance.
        Supports momentum, mean reversion, and statistical arbitrage strategies.
        
        Args:
            df: Processed intraday data with columns: day, open, close, vwap, ticker_dvol, sigma_open, dividend
            ticker_daily_data: Daily OHLCV data for the traded instrument
            all_days: List of unique trading days in the dataset
            config: Dictionary containing strategy parameters
            market_calendar: Dictionary containing holidays and early closes
            risk_free_rate_series: Series of risk-free rates indexed by date
            verbose: Whether to print progress information (default False)
            df_b: Intraday data for second asset (stat arb only)
            ticker_b_daily_data: Daily data for second asset (stat arb only)
            
        Returns:
            DataFrame with daily strategy performance metrics
        """
        # Store risk-free rate series
        if risk_free_rate_series is not None:
            self.risk_free_rate_series = risk_free_rate_series

        # Initialize base parameters from config
        config = config.copy()
        
        # Determine strategy type
        strategy_type = config.get('strategy_type', StrategyType.MOMENTUM)
        if isinstance(strategy_type, str):
            strategy_type = StrategyType(strategy_type)
        self.strategy_type = strategy_type
        
        # Validate stat arb has required data
        if strategy_type == StrategyType.STAT_ARB:
            if df_b is None:
                raise ValueError("Statistical arbitrage requires df_b (second asset data)")
        
        # Unpack configuration parameters
        AUM_0 = config['AUM']
        self.AUM_0 = AUM_0
        trade_freq = config['trade_freq']
        sizing_type = config['sizing_type']
        target_vol = config['target_vol']
        max_leverage = config['max_leverage']
        
        # Risk Management Parameters
        stop_loss_pct = config.get('stop_loss_pct', 0.02)
        take_profit_pct = config.get('take_profit_pct', 0.04)
        max_drawdown_pct = config.get('max_drawdown_pct', 0.15)
        lookback_period = config.get('lookback_period', 20)
        commission_rate = config.get('commission', 0.0035)
        slippage_factor = config.get('slippage_factor', 0.1)

        # Position Rebalancing Parameters
        enable_vol_rebalance = config.get('enable_vol_rebalance', False)
        vol_rebalance_threshold = config.get('vol_rebalance_threshold', 0.25)
        min_rebalance_size = config.get('min_rebalance_size', 0.10)
        
        # Initialize counters
        total_stop_losses = 0
        total_take_profits = 0
        max_drawdowns = 0
        buy_trades = 0
        sell_trades = 0

        current_position = 0
        current_shares = 0
        current_balance = AUM_0
        peak_aum = AUM_0
        entry_price = 0.0
        entry_spread = 0.0
        
        # Ensure consistent datetime format
        df['day'] = pd.to_datetime(df['day'])
        all_days = pd.DatetimeIndex([pd.Timestamp(d) for d in all_days])

        # Pre-compute benchmark returns
        df_daily = pd.DataFrame(ticker_daily_data)
        df_daily['caldt'] = pd.to_datetime(df_daily['caldt'])
        df_daily.set_index('caldt', inplace=True)
        df_daily['ret'] = df_daily['close'].pct_change()
        
        # Pre-compute all daily data (Optimization #3)
        precomputed_data = self._precompute_daily_data(df, all_days)
        
        # Pre-compute data for second asset (stat arb)
        precomputed_data_b = None
        if strategy_type == StrategyType.STAT_ARB and df_b is not None:
            df_b['day'] = pd.to_datetime(df_b['day'])
            precomputed_data_b = self._precompute_daily_data(df_b, all_days)
        
        # Get initial open price for benchmark
        first_valid_day_idx = 14
        if first_valid_day_idx < len(all_days) and all_days[first_valid_day_idx] in precomputed_data:
            open_price_init = precomputed_data[all_days[first_valid_day_idx]]['open']
        else:
            for day in all_days:
                if day in precomputed_data:
                    open_price_init = precomputed_data[day]['open']
                    break
            else:
                raise ValueError("No valid data found to initialize benchmark")

        # Flatten holidays for quick lookup
        all_holidays = set()
        if market_calendar:
            holidays_dict = market_calendar.get('holidays', {})
            if isinstance(holidays_dict, dict):
                for year_data in holidays_dict.values():
                    if isinstance(year_data, dict) and 'holidays' in year_data:
                        all_holidays.update(year_data['holidays'])
            elif isinstance(holidays_dict, list):
                all_holidays.update(holidays_dict)
        
        early_closes = market_calendar.get('early_closes', {}) if market_calendar else {}

        # Initialize strategy DataFrame
        strat = pd.DataFrame(index=all_days)
        strat['ret'] = np.nan
        strat['AUM'] = AUM_0
        strat['balance'] = AUM_0
        strat['ret_ticker'] = np.nan
        strat['position'] = 0
        strat['daily_var'] = np.nan
        strat['drawdown'] = np.nan
        strat['slippage_cost'] = np.nan
        strat['shares_held'] = 0
        
        # Additional columns for stat arb
        if strategy_type == StrategyType.STAT_ARB:
            strat['spread_zscore'] = np.nan
            strat['hedge_ratio'] = np.nan
        
        # Previous returns for VaR calculation
        previous_returns = []

        # Main backtest loop
        for d in range(1, len(all_days)):
            current_day = all_days[d]
            prev_day = all_days[d-1]
            current_day_date = pd.Timestamp(current_day).date()
            
            # Skip holidays
            if current_day_date in all_holidays:
                continue
                
            if prev_day not in precomputed_data or current_day not in precomputed_data:
                continue
            
            # For stat arb, also check second asset data
            if strategy_type == StrategyType.STAT_ARB:
                if current_day not in precomputed_data_b or prev_day not in precomputed_data_b:
                    continue
            
            # Get precomputed data (no copy needed for read-only access)
            current_day_info = precomputed_data[current_day]
            prev_day_info = precomputed_data[prev_day]
            
            # Handle early close
            close_prices = current_day_info['close_prices']
            volumes = current_day_info['volumes']
            vwap = current_day_info['vwap']
            sigma_open = current_day_info['sigma_open']
            min_from_open = current_day_info['min_from_open']
            
            # Get second asset data for stat arb
            if strategy_type == StrategyType.STAT_ARB:
                current_day_info_b = precomputed_data_b[current_day]
                close_prices_b = current_day_info_b['close_prices']
            
            if current_day_date in early_closes:
                early_close_time = early_closes[current_day_date]
                early_close_minute = early_close_time.hour * 60 + early_close_time.minute - (9 * 60 + 30)
                mask = min_from_open <= early_close_minute
                close_prices = close_prices[mask]
                volumes = volumes[mask]
                vwap = vwap[mask]
                sigma_open = sigma_open[mask]
                min_from_open = min_from_open[mask]
                
                if strategy_type == StrategyType.STAT_ARB:
                    close_prices_b = close_prices_b[mask]
            
            # Skip if no valid data
            if len(close_prices) == 0:
                continue
            
            # For stat arb, ensure both assets have same length
            if strategy_type == StrategyType.STAT_ARB:
                min_len = min(len(close_prices), len(close_prices_b))
                close_prices = close_prices[:min_len]
                close_prices_b = close_prices_b[:min_len]
                min_from_open = min_from_open[:min_len]
            
            # For momentum strategy, skip if no valid volatility data
            if strategy_type == StrategyType.MOMENTUM and np.all(np.isnan(sigma_open)):
                continue
            
            previous_aum = float(strat.at[prev_day, 'AUM'])
            
            # Calculate VaR
            if len(previous_returns) >= 2:
                strat.loc[current_day, 'daily_var'] = utils.calculate_VaR(previous_returns, 0.95)
            
            # Update peak and calculate drawdown
            if previous_aum > peak_aum:
                peak_aum = previous_aum
            
            current_drawdown = (previous_aum - peak_aum) / peak_aum if peak_aum != 0 else 0
            strat.at[current_day, 'drawdown'] = current_drawdown
            
            # Check max drawdown breach
            if abs(current_drawdown) > max_drawdown_pct:
                if verbose:
                    print(f"Max drawdown exceeded on {current_day}: {current_drawdown:.2%}")
                max_drawdowns += 1
                
                if current_shares > 0:
                    proceeds = current_shares * close_prices[-1]
                    current_balance += proceeds
                    current_shares = 0
                    current_position = 0
                    sell_trades += 1
                
                strat.at[current_day, 'AUM'] = current_balance
                strat.at[current_day, 'balance'] = current_balance
                strat.at[current_day, 'ret'] = (current_balance - previous_aum) / previous_aum
                strat.at[current_day, 'position'] = 0
                strat.at[current_day, 'shares_held'] = 0
                continue
            
            # Generate signals based on strategy type
            if strategy_type == StrategyType.MOMENTUM:
                prev_dividend = prev_day_info['dividend']
                prev_close_adjusted = prev_day_info['close_prices'][-1] - prev_dividend
                open_price = current_day_info['open']
                
                reference_high = max(open_price, prev_close_adjusted)
                reference_low = min(open_price, prev_close_adjusted)
                
                signals = self._generate_signals_momentum(
                    close_prices, vwap, sigma_open,
                    reference_high, reference_low, config
                )
                
            elif strategy_type == StrategyType.MEAN_REVERSION:
                signals = self._generate_signals_mean_reversion(close_prices, config)

            elif strategy_type == StrategyType.MEAN_REVERSION_RSI:
                signals = self._generate_signals_mean_reversion_rsi(close_prices, config)
                            
            elif strategy_type == StrategyType.STAT_ARB:
                signals, zscore, spread, hedge_ratios = self._generate_signals_stat_arb(
                    close_prices, close_prices_b, config
                )
                
                # Store stat arb metrics
                if len(zscore) > 0 and not np.all(np.isnan(zscore)):
                    strat.at[current_day, 'spread_zscore'] = zscore[-1]
                if len(hedge_ratios) > 0 and not np.all(np.isnan(hedge_ratios)):
                    strat.at[current_day, 'hedge_ratio'] = hedge_ratios[-1]

            elif strategy_type == StrategyType.SUPERVISED:
                signals = self._generate_ml_signals(config, current_day_info, precomputed_data)
            
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            # Create trading frequency mask
            trade_freq_mask = (min_from_open % trade_freq == 0)
            
            # Run position simulation based on strategy type
            if strategy_type == StrategyType.STAT_ARB:
                positions, entry_spreads_arr, exit_reasons, day_stop_losses, day_take_profits = \
                    StatArb.simulate_positions(
                        close_prices, close_prices_b, signals, hedge_ratios,
                        trade_freq_mask, stop_loss_pct, take_profit_pct,
                        current_position, entry_spread
                    )
            else:
                positions, entry_prices_arr, exit_reasons, day_stop_losses, day_take_profits = \
                    simulate_positions_numba(
                        close_prices, signals, trade_freq_mask,
                        stop_loss_pct, take_profit_pct,
                        current_position, entry_price
                    )
            
            total_stop_losses += day_stop_losses
            total_take_profits += day_take_profits
            
            # Position sizing
            ticker_vol = current_day_info['ticker_dvol']
            if sizing_type == "vol_target" and not np.isnan(ticker_vol) and ticker_vol > 0:
                size_factor = min(target_vol / ticker_vol, max_leverage)
            else:
                size_factor = max_leverage
            
            # Check for volatility-based rebalancing
            rebalance_triggered = False
            target_shares = current_shares
            
            if enable_vol_rebalance and current_shares > 0 and current_position == 1:
                # Calculate target position given current volatility
                target_position_value = (current_balance + current_shares * close_prices[-1]) * size_factor
                new_target_shares = int(target_position_value / close_prices[-1])
                
                # Compare to previous size factor
                if hasattr(self, '_last_size_factor') and self._last_size_factor > 0:
                    vol_change_pct = abs(size_factor - self._last_size_factor) / self._last_size_factor
                    position_change_pct = abs(new_target_shares - current_shares) / current_shares if current_shares > 0 else 0
                    
                    if vol_change_pct >= vol_rebalance_threshold and position_change_pct >= min_rebalance_size:
                        rebalance_triggered = True
                        target_shares = new_target_shares
                        
                        if verbose:
                            print(f"Rebalance triggered on {current_day}: vol changed {vol_change_pct:.1%}, "
                                  f"adjusting from {current_shares} to {target_shares} shares")
            
            # Track size factor for next day comparison
            self._last_size_factor = size_factor if current_position == 1 else 0.0

            # Calculate portfolio values
            if current_shares == 0 and positions[0] == 1:
                base_shares = int(current_balance * size_factor / close_prices[0])
                init_shares = base_shares
            elif rebalance_triggered:
                init_shares = target_shares  # Use rebalanced target
            else:
                init_shares = current_shares

            
            
            if strategy_type == StrategyType.STAT_ARB:
                cash_arr, shares_a_arr, shares_b_arr, portfolio_values, commissions, slippages, day_buys, day_sells = \
                    StatArb.calculate_portfolio_values(
                        close_prices, close_prices_b, positions, hedge_ratios,
                        entry_spreads_arr, current_balance, init_shares, 0,
                        commission_rate, slippage_factor
                    )
                shares_arr = shares_a_arr
            else:
                cash_arr, shares_arr, portfolio_values, commissions, slippages, day_buys, day_sells = \
                    calculate_portfolio_values_numba(
                        close_prices, positions, entry_prices_arr,
                        current_balance, init_shares,
                        commission_rate, slippage_factor
                    )
            
            buy_trades += day_buys
            sell_trades += day_sells
            
            # Update state from end of day
            current_balance = cash_arr[-1]
            current_shares = shares_arr[-1]
            current_position = positions[-1]
            
            if strategy_type == StrategyType.STAT_ARB:
                entry_spread = entry_spreads_arr[-1]
            else:
                entry_price = entry_prices_arr[-1]
            
            new_aum = portfolio_values[-1]
            new_ret = (new_aum - previous_aum) / previous_aum if previous_aum != 0 else 0.0
            
            # Get benchmark return
            if current_day in df_daily.index:
                ticker_ret = float(df_daily.loc[current_day, 'ret'])
            else:
                prev_dates = df_daily.index[df_daily.index <= current_day]
                ticker_ret = float(df_daily.loc[prev_dates[-1], 'ret']) if len(prev_dates) > 0 else 0.0
            
            # Update strategy metrics
            strat.loc[current_day, 'AUM'] = new_aum
            strat.loc[current_day, 'ret'] = new_ret
            strat.loc[current_day, 'ret_ticker'] = ticker_ret
            strat.loc[current_day, 'position'] = float(positions[-1])
            strat.loc[current_day, 'slippage_cost'] = float(slippages.sum())
            strat.loc[current_day, 'balance'] = float(current_balance)
            strat.loc[current_day, 'shares_held'] = int(current_shares)
            
            # Update peak
            if new_aum > peak_aum:
                peak_aum = new_aum
            
            # Update returns history
            if not np.isnan(new_ret):
                previous_returns.append(new_ret)
                if len(previous_returns) > lookback_period:
                    previous_returns.pop(0)
        
        # Store trade statistics
        self.total_trades = buy_trades + sell_trades
        self.buy_trades = buy_trades
        self.sell_trades = sell_trades
        self.stop_losses = total_stop_losses
        self.profit_takes = total_take_profits
        self.max_drawdown_breaches = max_drawdowns
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"BACKTEST SUMMARY ({strategy_type.value.upper()} STRATEGY)")
            print(f"{'='*60}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Buy Trades: {buy_trades}")
            print(f"Sell Trades: {sell_trades}")
            print(f"Stop Losses Hit: {total_stop_losses}")
            print(f"Take Profits Hit: {total_take_profits}")
            print(f"Max Drawdown Breaches: {max_drawdowns}")
            print(f"{'='*60}\n")
        
        return strat


    def plot_results(
        self, 
        strat: pd.DataFrame,
        risk_free_rate_series: pd.Series = None
    ):
        """
        Plot the results of the backtested strategy and calculate various statistics.
        
        Args:
            strat: The DataFrame containing the backtested strategy results
            risk_free_rate_series: Optional series of risk-free rates for Sharpe calculation
            
        Returns:
            plt: The matplotlib.pyplot object for the plot
            stats: Dictionary containing the calculated statistics
        """
        AUM_0 = self.AUM_0
        symbol = self.ticker
        stats = {}

        if risk_free_rate_series is None:
            risk_free_rate_series = self.risk_free_rate_series

        # Calculate cumulative products for AUM calculations
        strat['AUM_ticker'] = AUM_0 * (1 + strat['ret_ticker']).cumprod(skipna=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Determine strategy name for title
        strategy_name = self.strategy_type.value.replace('_', ' ').title()
        
        # For stat arb, include both tickers in title
        if self.strategy_type == StrategyType.STAT_ARB and self.ticker_b:
            title_symbol = f"{symbol}/{self.ticker_b}"
        else:
            title_symbol = symbol

        # Plot AUM
        ax.plot(strat.index, strat['AUM'], label=f'{strategy_name} Strategy', linewidth=2, color='k')
        ax.plot(strat.index, strat['AUM_ticker'], label=f'{symbol} Buy & Hold', linewidth=1, color='r')

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

        # Calculate statistics
        valid_strat_returns = strat['ret'].dropna()
        valid_ticker_returns = strat['ret_ticker'].dropna()
        
        total_ret = (strat['AUM'].iloc[-1] - AUM_0) / AUM_0
        ticker_ret = (strat['AUM_ticker'].iloc[-1] - AUM_0) / AUM_0
        
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
        ticker_vol = valid_ticker_returns.std() * np.sqrt(252)
        
        # Win rate
        winning_days = (valid_strat_returns > 0).sum()
        total_days = len(valid_strat_returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Average risk-free rate
        if risk_free_rate_series is not None and not risk_free_rate_series.empty:
            avg_rf_rate = risk_free_rate_series.mean() * 100
        else:
            avg_rf_rate = 0.0
        
        # Annualized returns
        num_years = len(valid_strat_returns) / 252
        ticker_num_years = len(valid_ticker_returns) / 252
        
        annualized_ret = (1 + total_ret) ** (1 / num_years) - 1 if num_years > 0 else 0
        ticker_annualized_ret = (1 + ticker_ret) ** (1 / ticker_num_years) - 1 if ticker_num_years > 0 else 0
        
        # Calmar ratio
        calmar = annualized_ret / abs(max_dd) if max_dd != 0 else np.nan
        ticker_calmar = ticker_annualized_ret / abs(ticker_max_dd) if ticker_max_dd != 0 else np.nan
        
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
        
        # Populate stats dictionary
        stats['Strategy Type'] = strategy_name
        stats['Total Return (%)'] = f"{total_ret * 100:.2f}%"
        stats[f'{symbol} Return (%)'] = f"{ticker_ret * 100:.2f}%"
        stats['Sharpe Ratio'] = f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"
        stats[f'{symbol} Sharpe'] = f"{ticker_sharpe:.2f}" if not np.isnan(ticker_sharpe) else "N/A"
        stats['Sortino Ratio'] = f"{sortino:.2f}" if not np.isnan(sortino) else "N/A"
        stats[f'{symbol} Sortino'] = f"{ticker_sortino:.2f}" if not np.isnan(ticker_sortino) else "N/A"
        stats['Calmar Ratio'] = f"{calmar:.2f}" if not np.isnan(calmar) else "N/A"
        stats[f'{symbol} Calmar'] = f"{ticker_calmar:.2f}" if not np.isnan(ticker_calmar) else "N/A"
        stats['Max Drawdown (%)'] = f"{max_dd * 100:.2f}%"
        stats[f'{symbol} Max DD (%)'] = f"{ticker_max_dd * 100:.2f}%"
        stats['Annualized Vol (%)'] = f"{strategy_vol * 100:.2f}%"
        stats[f'{symbol} Vol (%)'] = f"{ticker_vol * 100:.2f}%"
        stats['Win Rate (%)'] = f"{win_rate * 100:.2f}%"
        stats['Total Trades'] = f"{self.total_trades}"
        stats['Buy Trades'] = f"{self.buy_trades}"
        stats['Sell Trades'] = f"{self.sell_trades}"
        stats['Stop Losses Hit'] = f"{self.stop_losses}"
        stats['Take Profits Hit'] = f"{self.profit_takes}"
        stats['Avg Risk-Free Rate (%)'] = f"{avg_rf_rate:.2f}%"
        
        plt.tight_layout()
        
        return plt, stats
