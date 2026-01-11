"""
Backtest execution engine.

This module provides the main backtest execution loop that:
    - Manages daily data iteration
    - Coordinates signal generation across strategy types
    - Handles position simulation and portfolio tracking
    - Applies risk management rules
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple

from core.portfolio import (
    simulate_positions_numba, 
    calculate_portfolio_values_numba,
    calculate_kelly_position_size_numba
)
from core.metrics import MetricsCalculator

from strats.momentum import Momentum
from strats.momentum_atr import MomentumATR
from strats.mean_reversion import MeanReversion
from strats.mean_reversion_rsi import MeanReversionRSI
from strats.stat_arb import StatArb

# Import ML strategy (handle case where it might not be available)
try:
    from strats.classification import SupervisedStrategy
    HAS_ML = True
except ImportError:
    HAS_ML = False
    SupervisedStrategy = None

from .results import BacktestResults

warnings.filterwarnings('ignore')


class StrategyType:
    """Strategy type constants."""
    MOMENTUM = "momentum"
    MOMENTUM_ATR = "momentum_atr"
    MEAN_REVERSION = "mean_reversion"
    MEAN_REVERSION_RSI = "mean_reversion_rsi"
    STAT_ARB = "stat_arb"
    SUPERVISED = "supervised"


class BacktestEngine:
    """
    Core backtest execution engine.
    
    Manages the backtest loop and coordinates:
        - Data preprocessing and daily iteration
        - Signal generation for different strategy types
        - Position simulation with risk management
        - Portfolio value tracking
    
    Example:
        engine = BacktestEngine(ticker='NVDA')
        
        config = {
            'strategy_type': 'momentum',
            'AUM': 100000.0,
            'trade_freq': 30,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'band_mult': 1.0,
            ...
        }
        
        results = engine.run(
            df=intraday_data,
            ticker_daily_data=daily_data,
            all_days=trading_days,
            config=config
        )
    """
    
    def __init__(
        self,
        ticker: str,
        ticker_b: Optional[str] = None,
        initial_aum: float = 100000.0
    ):
        """
        Initialize the BacktestEngine.
        
        Args:
            ticker: Primary ticker symbol
            ticker_b: Secondary ticker for pairs trading (optional)
            initial_aum: Starting portfolio value
        """
        self.ticker = ticker
        self.ticker_b = ticker_b
        self.initial_aum = initial_aum
        
        # ML strategy instance (lazy loaded)
        self._ml_strategy: Optional[SupervisedStrategy] = None
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Trade statistics (populated after backtest)
        self.total_trades = 0
        self.buy_trades = 0
        self.sell_trades = 0
        self.stop_losses = 0
        self.profit_takes = 0
        self.max_drawdown_breaches = 0
        
        # Internal state
        self._last_size_factor = 0.0
        self.strategy_type = StrategyType.MOMENTUM
        self.risk_free_rate_series = None
    
    def _precompute_daily_data(
        self, 
        df: pd.DataFrame, 
        all_days: np.ndarray
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Pre-compute all daily data structures.
        
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
                    'high_prices': day_data['high'].values.astype(np.float64),
                    'low_prices': day_data['low'].values.astype(np.float64),
                    'volumes': day_data['volume'].values.astype(np.float64),
                    'vwap': day_data['vwap'].values.astype(np.float64) if 'vwap' in day_data.columns else np.zeros(len(day_data)),
                    'sigma_open': day_data['sigma_open'].values.astype(np.float64) if 'sigma_open' in day_data.columns else np.zeros(len(day_data)),
                    'ticker_dvol': day_data['ticker_dvol'].iloc[0] if 'ticker_dvol' in day_data.columns else 0.0,
                    'min_from_open': day_data['min_from_open'].values.astype(np.float64) if 'min_from_open' in day_data.columns else np.arange(len(day_data)),
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
        """Generate momentum strategy signals."""
        band_mult = config['band_mult']
        return Momentum.generate_signals(
            close_prices, vwap, sigma_open,
            reference_high, reference_low, band_mult
        )
    
    def _generate_signals_momentum_atr(
        self,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        vwap: np.ndarray,
        reference_high: float,
        reference_low: float,
        config: dict,
        lookback: int = 14
    ) -> np.ndarray:
        """
        Generate ATR-based momentum strategy signals.
        
        Args:
            close_prices: Array of close prices
            high_prices: Array of high prices
            low_prices: Array of low prices
            vwap: Array of VWAP values
            reference_high: Reference price for upper band
            reference_low: Reference price for lower band
            config: Strategy configuration
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
        """
        atr_mult = config.get('atr_mult', 2.0)
        atr_period = config.get('atr_period', 14)
        
        return MomentumATR.generate_signals_trailing(
            close_prices, high_prices, low_prices, vwap,
            atr_mult, atr_period, lookback
        )
    
    def _generate_signals_mean_reversion(
        self,
        close_prices: np.ndarray,
        config: dict
    ) -> np.ndarray:
        """Generate mean reversion (z-score) strategy signals."""
        return MeanReversion.generate_signals(
            close_prices,
            config.get('zscore_lookback', 20),
            config.get('n_std_upper', 2.0),
            config.get('n_std_lower', 2.0),
            config.get('exit_threshold', 0.0)
        )
    
    def _generate_signals_mean_reversion_rsi(
        self,
        close_prices: np.ndarray,
        config: dict
    ) -> np.ndarray:
        """Generate RSI + SMA mean reversion strategy signals."""
        return MeanReversionRSI.generate_signals(
            close_prices,
            config.get('rsi_period', 10),
            config.get('rsi_oversold', 30.0),
            config.get('rsi_overbought', 70.0),
            config.get('sma_period', 200),
            config.get('exit_on_sma_cross', True)
        )
    
    def _generate_signals_stat_arb(
        self,
        close_prices_a: np.ndarray,
        close_prices_b: np.ndarray,
        config: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate statistical arbitrage signals."""
        return StatArb.generate_signals(
            close_prices_a,
            close_prices_b,
            config.get('zscore_lookback', 60),
            config.get('entry_threshold', 2.0),
            config.get('exit_threshold', 0.0),
            config.get('hedge_ratio', None),
            config.get('use_dynamic_hedge', True)
        )
    
    def _load_ml_strategy(self, config: Dict[str, Any]) -> 'SupervisedStrategy':
        """Load or retrieve the ML strategy instance."""
        if not HAS_ML:
            raise ImportError("ML module not available. Install ml package dependencies.")
        
        if self._ml_strategy is None:
            model_path = config.get('model_path')
            scaler_path = config.get('scaler_path')
            features_path = config.get('features_path')
            
            if not all([model_path, scaler_path, features_path]):
                raise ValueError(
                    "Supervised strategy requires model_path, scaler_path, "
                    "and features_path in config"
                )
            
            self._ml_strategy = SupervisedStrategy.from_artifacts(
                model_path=model_path,
                scaler_path=scaler_path,
                features_path=features_path,
                buy_threshold=config.get('buy_threshold', 0.55),
                sell_threshold=config.get('sell_threshold', 0.45)
            )
        
        return self._ml_strategy
    
    def _generate_signals_supervised(
        self,
        current_day_info: Dict[str, Any],
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Generate ML strategy signals."""
        strategy = self._load_ml_strategy(config)
        return strategy.generate_signals_for_backtest(current_day_info)
    
    def _generate_signals_from_precomputed(
        self,
        day_probs: np.ndarray,
        buy_threshold: float,
        sell_threshold: float
    ) -> np.ndarray:
        """
        Generate ML signals from precomputed probabilities (FAST PATH).
        
        This avoids loading the model and generating features for each day,
        providing ~10-20x speedup for ML strategy backtests.
        
        Args:
            day_probs: Probability array for this day's bars
            buy_threshold: Threshold for buy signals
            sell_threshold: Threshold for sell signals
            
        Returns:
            Array of signals for the day
        """
        n = len(day_probs)
        signals = np.zeros(n, dtype=np.int32)
        
        in_position = False
        
        for i in range(n):
            prob = day_probs[i]
            
            if np.isnan(prob):
                continue
            
            if prob >= buy_threshold and not in_position:
                signals[i] = 1  # LONG
                in_position = True
            elif prob <= sell_threshold and in_position:
                signals[i] = -1  # EXIT
                in_position = False
        
        return signals
    
    def _calculate_position_size_factor(
        self,
        sizing_type: str,
        ticker_vol: float,
        target_vol: float,
        max_leverage: float,
        previous_returns: list,
        kelly_fraction: float,
        kelly_lookback: int,
        kelly_min_trades: int,
        kelly_vol_blend_weight: float
    ) -> float:
        """
        Calculate position size factor based on sizing type.
        
        Args:
            sizing_type: 'vol_target', 'kelly', or 'kelly_vol_blend'
            ticker_vol: Current ticker volatility
            target_vol: Target volatility for vol_target sizing
            max_leverage: Maximum leverage allowed
            previous_returns: List of previous daily returns
            kelly_fraction: Fraction of Kelly to use
            kelly_lookback: Lookback period for Kelly calculation
            kelly_min_trades: Minimum trades before using Kelly
            kelly_vol_blend_weight: Weight for Kelly in blended approach
            
        Returns:
            Position size factor (0.0 to max_leverage)
        """
        vol_size = max_leverage
        if not np.isnan(ticker_vol) and ticker_vol > 0:
            vol_size = min(target_vol / ticker_vol, max_leverage)
        
        if sizing_type == "vol_target":
            return vol_size
        
        elif sizing_type == "kelly":
            if len(previous_returns) >= kelly_min_trades:
                returns_arr = np.array(previous_returns, dtype=np.float64)
                _, kelly_raw = calculate_kelly_position_size_numba(
                    returns=returns_arr,
                    base_size=1,
                    lookback=kelly_lookback,
                    kelly_fraction=kelly_fraction,
                    max_leverage=max_leverage,
                    min_trades=kelly_min_trades
                )
                return max(0.0, min(kelly_raw * kelly_fraction, max_leverage))
            else:
                return 0.5 * max_leverage
        
        elif sizing_type == "kelly_vol_blend":
            if len(previous_returns) >= kelly_min_trades:
                returns_arr = np.array(previous_returns, dtype=np.float64)
                _, kelly_raw = calculate_kelly_position_size_numba(
                    returns=returns_arr,
                    base_size=1,
                    lookback=kelly_lookback,
                    kelly_fraction=kelly_fraction,
                    max_leverage=max_leverage,
                    min_trades=kelly_min_trades
                )
                kelly_size = max(0.0, min(kelly_raw * kelly_fraction, max_leverage))
                return kelly_vol_blend_weight * kelly_size + (1 - kelly_vol_blend_weight) * vol_size
            else:
                return vol_size
        
        else:
            return vol_size
    
    def run(
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
    ) -> BacktestResults:
        """
        Run the backtest.
        
        Args:
            df: Intraday data with OHLCV columns
            ticker_daily_data: Daily OHLCV data
            all_days: List of trading days
            config: Strategy configuration dictionary
            market_calendar: Holidays and early closes
            risk_free_rate_series: Risk-free rate series
            verbose: Print progress information
            df_b: Second asset intraday data (stat arb only)
            ticker_b_daily_data: Second asset daily data (stat arb only)
            
        Returns:
            BacktestResults object with all results
        """
        self.risk_free_rate_series = risk_free_rate_series
        config = config.copy()
        
        # Determine strategy type
        strategy_type = config.get('strategy_type', StrategyType.MOMENTUM)
        if hasattr(strategy_type, 'value'):
            strategy_type = strategy_type.value
        self.strategy_type = strategy_type
        
        # Validate stat arb data
        if strategy_type == StrategyType.STAT_ARB and df_b is None:
            raise ValueError("Statistical arbitrage requires df_b")
        
        # Extract config parameters
        AUM_0 = config['AUM']
        self.initial_aum = AUM_0
        trade_freq = config['trade_freq']
        sizing_type = config.get('sizing_type', 'vol_target')
        target_vol = config.get('target_vol', 0.02)
        max_leverage = config.get('max_leverage', 1)
        
        # Kelly sizing parameters
        kelly_fraction = config.get('kelly_fraction', 0.5)
        kelly_lookback = config.get('kelly_lookback', 60)
        kelly_min_trades = config.get('kelly_min_trades', 30)
        kelly_vol_blend_weight = config.get('kelly_vol_blend_weight', 0.5)
        
        # Risk parameters
        stop_loss_pct = config.get('stop_loss_pct', 0.02)
        take_profit_pct = config.get('take_profit_pct', 0.04)
        max_drawdown_pct = config.get('max_drawdown_pct', 0.15)
        lookback_period = config.get('lookback_period', 20)
        commission_rate = config.get('commission', 0.0035)
        slippage_factor = config.get('slippage_factor', 0.1)
        
        # Rebalancing parameters
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
        
        # Prepare data
        df['day'] = pd.to_datetime(df['day'])
        all_days = pd.DatetimeIndex([pd.Timestamp(d) for d in all_days])
        
        # Precompute benchmark returns
        df_daily = pd.DataFrame(ticker_daily_data)
        df_daily['caldt'] = pd.to_datetime(df_daily['caldt'])
        df_daily.set_index('caldt', inplace=True)
        df_daily['ret'] = df_daily['close'].pct_change()
        
        # Precompute daily data
        precomputed_data = self._precompute_daily_data(df, all_days)
        
        # Precompute day-to-index mapping for fast probability lookups (ML optimization)
        day_to_indices = {}
        if strategy_type == StrategyType.SUPERVISED and 'precomputed_probabilities' in config:
            df_reset = df.reset_index(drop=True)
            if verbose:
                print(f"DEBUG: Building day_to_indices mapping...")
                print(f"DEBUG: df_reset.index type: {type(df_reset.index)}")
                print(f"DEBUG: df_reset.index[:5]: {df_reset.index[:5].tolist()}")
            for day in all_days:
                mask = df_reset['day'] == day
                day_to_indices[day] = df_reset.index[mask].tolist()
            if verbose:
                print(f"DEBUG: Built {len(day_to_indices)} day mappings")
                if len(day_to_indices) > 0:
                    first_key = list(day_to_indices.keys())[0]
                    first_vals = day_to_indices[first_key][:5]
                    print(f"DEBUG: First key: {first_key} -> indices[:5]: {first_vals}")
        
        # Precompute second asset data
        precomputed_data_b = None
        if strategy_type == StrategyType.STAT_ARB and df_b is not None:
            df_b['day'] = pd.to_datetime(df_b['day'])
            precomputed_data_b = self._precompute_daily_data(df_b, all_days)
        
        # Process market calendar
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
        
        # Initialize results DataFrame
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
        strat['size_factor'] = np.nan
        strat['kelly_raw'] = np.nan
        
        # Pre-populate ret_ticker for ALL days from daily data
        # This ensures benchmark return is consistent regardless of which days strategy processes
        for day in all_days:
            if day in df_daily.index:
                strat.loc[day, 'ret_ticker'] = df_daily.loc[day, 'ret']
            else:
                # Find most recent available date
                prev_dates = df_daily.index[df_daily.index <= day]
                if len(prev_dates) > 0:
                    strat.loc[day, 'ret_ticker'] = df_daily.loc[prev_dates[-1], 'ret']
        
        if strategy_type == StrategyType.STAT_ARB:
            strat['spread_zscore'] = np.nan
            strat['hedge_ratio'] = np.nan
        
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
            
            if strategy_type == StrategyType.STAT_ARB:
                if current_day not in precomputed_data_b or prev_day not in precomputed_data_b:
                    continue
            
            # Get precomputed data
            current_day_info = precomputed_data[current_day]
            prev_day_info = precomputed_data[prev_day]
            
            close_prices = current_day_info['close_prices']
            high_prices = current_day_info['high_prices']
            low_prices = current_day_info['low_prices']
            volumes = current_day_info['volumes']
            vwap = current_day_info['vwap']
            sigma_open = current_day_info['sigma_open']
            min_from_open = current_day_info['min_from_open']
            
            if strategy_type == StrategyType.STAT_ARB:
                current_day_info_b = precomputed_data_b[current_day]
                close_prices_b = current_day_info_b['close_prices']
            
            # Handle early close
            if current_day_date in early_closes:
                early_close_time = early_closes[current_day_date]
                early_close_minute = early_close_time.hour * 60 + early_close_time.minute - (9 * 60 + 30)
                mask = min_from_open <= early_close_minute
                close_prices = close_prices[mask]
                high_prices = high_prices[mask]
                low_prices = low_prices[mask]
                volumes = volumes[mask]
                vwap = vwap[mask]
                sigma_open = sigma_open[mask]
                min_from_open = min_from_open[mask]
                
                if strategy_type == StrategyType.STAT_ARB:
                    close_prices_b = close_prices_b[mask]
            
            if len(close_prices) == 0:
                continue
            
            if strategy_type == StrategyType.STAT_ARB:
                min_len = min(len(close_prices), len(close_prices_b))
                close_prices = close_prices[:min_len]
                close_prices_b = close_prices_b[:min_len]
                min_from_open = min_from_open[:min_len]
            
            if strategy_type == StrategyType.MOMENTUM and np.all(np.isnan(sigma_open)):
                continue
            
            previous_aum = float(strat.at[prev_day, 'AUM'])
            
            # Calculate VaR
            if len(previous_returns) >= 2:
                strat.loc[current_day, 'daily_var'] = self.metrics_calculator.calculate_var(
                    np.array(previous_returns), 0.95
                )
            
            # Update peak and drawdown
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
            
            elif strategy_type == StrategyType.MOMENTUM_ATR:
                prev_dividend = prev_day_info['dividend']
                prev_close_adjusted = prev_day_info['close_prices'][-1] - prev_dividend
                open_price = current_day_info['open']
                
                reference_high = max(open_price, prev_close_adjusted)
                reference_low = min(open_price, prev_close_adjusted)
                
                signals = self._generate_signals_momentum_atr(
                    close_prices, high_prices, low_prices, vwap,
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
                
                if len(zscore) > 0 and not np.all(np.isnan(zscore)):
                    strat.at[current_day, 'spread_zscore'] = zscore[-1]
                if len(hedge_ratios) > 0 and not np.all(np.isnan(hedge_ratios)):
                    strat.at[current_day, 'hedge_ratio'] = hedge_ratios[-1]
            
            elif strategy_type == StrategyType.SUPERVISED:
                # Check for precomputed probabilities (FAST PATH)
                precomputed_probs = config.get('precomputed_probabilities')
                
                # Debug: Log first iteration info
                if d == 1 and verbose:
                    print(f"DEBUG: day_to_indices has {len(day_to_indices)} days")
                    print(f"DEBUG: precomputed_probs is {'set' if precomputed_probs is not None else 'None'}")
                    if precomputed_probs is not None:
                        print(f"DEBUG: precomputed_probs length: {len(precomputed_probs)}")
                    print(f"DEBUG: current_day = {current_day}, type = {type(current_day)}")
                    if len(day_to_indices) > 0:
                        first_key = list(day_to_indices.keys())[0]
                        print(f"DEBUG: first key in day_to_indices = {first_key}, type = {type(first_key)}")
                        print(f"DEBUG: current_day in day_to_indices = {current_day in day_to_indices}")
                
                if precomputed_probs is not None and current_day in day_to_indices:
                    # Get row indices for this day
                    day_indices = day_to_indices[current_day]
                    
                    # Handle potential index mismatch
                    if len(day_indices) > 0 and max(day_indices) < len(precomputed_probs):
                        day_probs = precomputed_probs[day_indices]
                        
                        # Match length to close_prices (handle early close truncation)
                        n_prices = len(close_prices)
                        if len(day_probs) > n_prices:
                            day_probs = day_probs[:n_prices]
                        elif len(day_probs) < n_prices:
                            # Shouldn't happen, but pad with NaN if needed
                            padded = np.full(n_prices, np.nan)
                            padded[:len(day_probs)] = day_probs
                            day_probs = padded
                        
                        signals = self._generate_signals_from_precomputed(
                            day_probs,
                            config.get('buy_threshold', 0.55),
                            config.get('sell_threshold', 0.45)
                        )
                    else:
                        # Fallback to slow path
                        ml_day_info = {
                            'close_prices': close_prices,
                            'high_prices': high_prices,
                            'low_prices': low_prices,
                            'volumes': volumes,
                            'vwap': vwap,
                            'open': current_day_info['open'],
                            'min_from_open': min_from_open,
                            'ticker_dvol': current_day_info['ticker_dvol']
                        }
                        signals = self._generate_signals_supervised(ml_day_info, config)
                else:
                    # Original slow path - load model and generate features per day
                    ml_day_info = {
                        'close_prices': close_prices,
                        'high_prices': high_prices,
                        'low_prices': low_prices,
                        'volumes': volumes,
                        'vwap': vwap,
                        'open': current_day_info['open'],
                        'min_from_open': min_from_open,
                        'ticker_dvol': current_day_info['ticker_dvol']
                    }
                    signals = self._generate_signals_supervised(ml_day_info, config)
            
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            # Trade frequency mask - use bar count instead of minutes
            if len(min_from_open) > 1:
                bar_size_minutes = int(min_from_open[1] - min_from_open[0])
            else:
                bar_size_minutes = 1

            bars_per_trade = max(1, trade_freq // bar_size_minutes)
            trade_freq_mask = (np.arange(len(close_prices)) % bars_per_trade == 0)
                        
            # Debug: Show signals for first few days
            if verbose and d <= 5 and strategy_type == StrategyType.SUPERVISED:
                n_buy = (signals == 1).sum()
                n_sell = (signals == -1).sum()
                n_hold = (signals == 0).sum()
                print(f"DEBUG Day {d} ({current_day.date()}): signals -> BUY:{n_buy}, SELL:{n_sell}, HOLD:{n_hold}")
                print(f"DEBUG Day {d}: trade_freq={trade_freq}, trade_freq_mask sum={trade_freq_mask.sum()}")
                print(f"DEBUG Day {d}: min_from_open[:10]={min_from_open[:10]}")
            
            # Position simulation
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
            
            # Debug: Show positions for first few days
            if verbose and d <= 5 and strategy_type == StrategyType.SUPERVISED:
                pos_changes = np.diff(positions, prepend=current_position)
                n_entries = (pos_changes == 1).sum()
                n_exits = (pos_changes == -1).sum()
                print(f"DEBUG Day {d}: positions -> entries:{n_entries}, exits:{n_exits}, final_pos:{positions[-1]}")
            
            total_stop_losses += day_stop_losses
            total_take_profits += day_take_profits
            
            # Position sizing
            ticker_vol = current_day_info['ticker_dvol']
            size_factor = self._calculate_position_size_factor(
                sizing_type=sizing_type,
                ticker_vol=ticker_vol,
                target_vol=target_vol,
                max_leverage=max_leverage,
                previous_returns=previous_returns,
                kelly_fraction=kelly_fraction,
                kelly_lookback=kelly_lookback,
                kelly_min_trades=kelly_min_trades,
                kelly_vol_blend_weight=kelly_vol_blend_weight
            )
            
            strat.at[current_day, 'size_factor'] = size_factor
            
            if sizing_type in ('kelly', 'kelly_vol_blend') and len(previous_returns) >= kelly_min_trades:
                returns_arr = np.array(previous_returns, dtype=np.float64)
                _, kelly_raw = calculate_kelly_position_size_numba(
                    returns=returns_arr,
                    base_size=1,
                    lookback=kelly_lookback,
                    kelly_fraction=kelly_fraction,
                    max_leverage=max_leverage,
                    min_trades=kelly_min_trades
                )
                strat.at[current_day, 'kelly_raw'] = kelly_raw
            
            # Volatility-based rebalancing
            rebalance_triggered = False
            target_shares = current_shares
            
            if enable_vol_rebalance and current_shares > 0 and current_position == 1:
                target_position_value = (current_balance + current_shares * close_prices[-1]) * size_factor
                new_target_shares = int(target_position_value / close_prices[-1])
                
                if self._last_size_factor > 0:
                    vol_change_pct = abs(size_factor - self._last_size_factor) / self._last_size_factor
                    position_change_pct = abs(new_target_shares - current_shares) / current_shares if current_shares > 0 else 0
                    
                    if vol_change_pct >= vol_rebalance_threshold and position_change_pct >= min_rebalance_size:
                        rebalance_triggered = True
                        target_shares = new_target_shares
                        
                        if verbose:
                            print(f"Rebalance triggered on {current_day}: vol changed {vol_change_pct:.1%}, "
                                  f"adjusting from {current_shares} to {target_shares} shares")
            
            self._last_size_factor = size_factor if current_position == 1 else 0.0
            
            # Calculate portfolio values
            if current_shares == 0 and positions[0] == 1:
                init_shares = int(current_balance * size_factor / close_prices[0])
            elif rebalance_triggered:
                init_shares = target_shares
            else:
                init_shares = current_shares
            
            if strategy_type == StrategyType.STAT_ARB:
                cash_arr, shares_a_arr, shares_b_arr, portfolio_values, commissions, slippages, day_buys, day_sells = \
                    StatArb.calculate_portfolio_values(
                        close_prices, close_prices_b, positions, hedge_ratios,
                        entry_spreads_arr, current_balance, init_shares, 0,
                        commission_rate, slippage_factor,
                        size_factor
                    )
                shares_arr = shares_a_arr
            else:
                cash_arr, shares_arr, portfolio_values, commissions, slippages, day_buys, day_sells = \
                    calculate_portfolio_values_numba(
                        close_prices, positions, entry_prices_arr,
                        current_balance, init_shares,
                        commission_rate, slippage_factor,
                        size_factor
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
            
            # Update strategy metrics
            strat.loc[current_day, 'AUM'] = new_aum
            strat.loc[current_day, 'ret'] = new_ret
            # Note: ret_ticker is pre-populated for all days at initialization
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
            print(f"BACKTEST SUMMARY ({strategy_type.upper()} STRATEGY)")
            print(f"{'='*60}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Buy Trades: {buy_trades}")
            print(f"Sell Trades: {sell_trades}")
            print(f"Stop Losses Hit: {total_stop_losses}")
            print(f"Take Profits Hit: {total_take_profits}")
            print(f"Max Drawdown Breaches: {max_drawdowns}")
            print(f"Sizing Type: {sizing_type}")
            if sizing_type in ('kelly', 'kelly_vol_blend'):
                print(f"Kelly Fraction: {kelly_fraction}")
                print(f"Kelly Lookback: {kelly_lookback}")
            print(f"{'='*60}\n")
        
        # Build results object
        trade_stats = {
            'total_trades': self.total_trades,
            'buy_trades': self.buy_trades,
            'sell_trades': self.sell_trades,
            'stop_losses': self.stop_losses,
            'take_profits': self.profit_takes,
            'max_drawdown_breaches': self.max_drawdown_breaches
        }
        
        return BacktestResults(
            strategy_df=strat,
            ticker=self.ticker,
            ticker_b=self.ticker_b,
            strategy_type=strategy_type,
            initial_aum=AUM_0,
            trade_stats=trade_stats,
            risk_free_rate_series=risk_free_rate_series
        )
    
    # Backwards compatibility method
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
        Run backtest and return strategy DataFrame (backwards compatible).
        
        This method provides backwards compatibility with the original BackTest class.
        For new code, use the `run()` method which returns a BacktestResults object.
        """
        results = self.run(
            df=df,
            ticker_daily_data=ticker_daily_data,
            all_days=all_days,
            config=config,
            market_calendar=market_calendar,
            risk_free_rate_series=risk_free_rate_series,
            verbose=verbose,
            df_b=df_b,
            ticker_b_daily_data=ticker_b_daily_data
        )
        return results.strategy_df
