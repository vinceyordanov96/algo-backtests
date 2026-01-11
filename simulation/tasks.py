"""
Task generation for simulation sweeps.

This module provides task generators that create parameter combinations
for each strategy type.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from itertools import product

from .config import SimulationConfig, StrategyType


class TaskGenerator:
    """
    Generates task dictionaries for simulation sweeps.
    
    Each task dictionary contains all parameters needed to run
    a single backtest configuration.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the TaskGenerator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
    
    def generate_tasks(
        self,
        ticker: str,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        market_calendar: Dict,
        risk_free_rate_series: Optional[pd.Series] = None,
        df_b: Optional[pd.DataFrame] = None,
        df_b_daily: Optional[pd.DataFrame] = None,
        ticker_b: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate task dictionaries for the configured strategy type.
        
        Args:
            ticker: Primary ticker symbol
            df: Intraday data
            df_daily: Daily data
            market_calendar: Market calendar with holidays and early closes
            risk_free_rate_series: Risk-free rate series (optional)
            df_b: Second asset intraday data (stat arb only)
            df_b_daily: Second asset daily data (stat arb only)
            ticker_b: Second ticker symbol (stat arb only)
            
        Returns:
            List of task dictionaries
        """
        if self.config.strategy_type == StrategyType.MOMENTUM:
            return self._generate_momentum_tasks(
                ticker, df, df_daily, market_calendar, risk_free_rate_series
            )
        elif self.config.strategy_type == StrategyType.MOMENTUM_ATR:
            return self._generate_momentum_atr_tasks(
                ticker, df, df_daily, market_calendar, risk_free_rate_series
            )
        elif self.config.strategy_type == StrategyType.MEAN_REVERSION:
            return self._generate_mean_reversion_tasks(
                ticker, df, df_daily, market_calendar, risk_free_rate_series
            )
        elif self.config.strategy_type == StrategyType.MEAN_REVERSION_RSI:
            return self._generate_mean_reversion_rsi_tasks(
                ticker, df, df_daily, market_calendar, risk_free_rate_series
            )
        elif self.config.strategy_type == StrategyType.STAT_ARB:
            return self._generate_stat_arb_tasks(
                ticker, ticker_b, df, df_daily, df_b, df_b_daily,
                market_calendar, risk_free_rate_series
            )
        elif self.config.strategy_type == StrategyType.SUPERVISED:
            return self._generate_supervised_tasks(
                ticker, df, df_daily, market_calendar, risk_free_rate_series
            )
        else:
            raise ValueError(f"Unknown strategy type: {self.config.strategy_type}")
    
    def _get_common_params(self) -> Dict[str, List]:
        """Get common parameter lists."""
        c = self.config.common
        return {
            'trade_freq': c.trade_frequencies,
            'target_vol': c.target_volatilities,
            'stop_loss_pct': c.stop_loss_pcts,
            'take_profit_pct': c.take_profit_pcts,
            'max_drawdown_pct': c.max_drawdown_pcts,
            'sizing_type': c.sizing_types,
            'kelly_fraction': c.kelly_fractions,
            'kelly_lookback': c.kelly_lookbacks,
        }
    
    def _base_task(
        self,
        ticker: str,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        market_calendar: Dict,
        risk_free_rate_series: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Create base task with common fields."""
        c = self.config.common
        return {
            'ticker': ticker,
            'df': df,
            'df_daily': df_daily,
            'market_calendar': market_calendar,
            'risk_free_rate_series': risk_free_rate_series,
            'commission': c.commission,
            'slippage_factor': c.slippage_factor,
            'max_leverage': c.max_leverage,
            'initial_aum': c.initial_aum,
            'lookback_period': c.lookback_period,
            'kelly_min_trades': c.kelly_min_trades,
            'kelly_vol_blend_weight': c.kelly_vol_blend_weight,
        }
    
    def _generate_momentum_tasks(
        self,
        ticker: str,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        market_calendar: Dict,
        risk_free_rate_series: Optional[pd.Series]
    ) -> List[Dict[str, Any]]:
        """Generate momentum strategy tasks."""
        tasks = []
        common = self._get_common_params()
        base = self._base_task(ticker, df, df_daily, market_calendar, risk_free_rate_series)
        
        param_lists = [
            self.config.momentum.band_multipliers,
            common['trade_freq'],
            common['target_vol'],
            common['stop_loss_pct'],
            common['take_profit_pct'],
            common['max_drawdown_pct'],
            common['sizing_type'],
            common['kelly_fraction'],
            common['kelly_lookback'],
        ]
        
        for (band_mult, trade_freq, target_vol, stop_loss, take_profit, 
             max_dd, sizing_type, kelly_fraction, kelly_lookback) in product(*param_lists):
            task = base.copy()
            task.update({
                'strategy_type': StrategyType.MOMENTUM,
                'band_mult': band_mult,
                'trade_freq': trade_freq,
                'target_vol': target_vol,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
                'max_drawdown_pct': max_dd,
                'sizing_type': sizing_type,
                'kelly_fraction': kelly_fraction,
                'kelly_lookback': kelly_lookback,
            })
            tasks.append(task)
        
        return tasks

    def _generate_momentum_atr_tasks(
        self,
        ticker: str,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        market_calendar: Dict,
        risk_free_rate_series: Optional[pd.Series]
    ) -> List[Dict[str, Any]]:
        """Generate ATR-based momentum strategy tasks."""
        tasks = []
        common = self._get_common_params()
        base = self._base_task(ticker, df, df_daily, market_calendar, risk_free_rate_series)
        
        # Generate all parameter combinations
        param_lists = [
            self.config.momentum_atr.atr_multipliers,
            self.config.momentum_atr.atr_periods,
            common['trade_freq'],
            common['target_vol'],
            common['stop_loss_pct'],
            common['take_profit_pct'],
            common['max_drawdown_pct'],
            common['sizing_type'],
            common['kelly_fraction'],
            common['kelly_lookback'],
        ]
        
        for (atr_mult, atr_period, trade_freq, target_vol, 
             stop_loss, take_profit, max_dd, sizing_type, kelly_fraction, kelly_lookback) in product(*param_lists):
            task = base.copy()
            task.update({
                'strategy_type': StrategyType.MOMENTUM_ATR,
                'atr_mult': atr_mult,
                'atr_period': atr_period,
                'trade_freq': trade_freq,
                'target_vol': target_vol,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
                'max_drawdown_pct': max_dd,
                'sizing_type': sizing_type,
                'kelly_fraction': kelly_fraction,
                'kelly_lookback': kelly_lookback,
            })
            tasks.append(task)
        
        return tasks
    
    def _generate_mean_reversion_tasks(
        self,
        ticker: str,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        market_calendar: Dict,
        risk_free_rate_series: Optional[pd.Series]
    ) -> List[Dict[str, Any]]:
        """Generate mean reversion (z-score) strategy tasks."""
        tasks = []
        common = self._get_common_params()
        base = self._base_task(ticker, df, df_daily, market_calendar, risk_free_rate_series)
        mr = self.config.mean_reversion
        
        param_lists = [
            mr.zscore_lookbacks,
            mr.n_std_uppers,
            mr.n_std_lowers,
            mr.exit_thresholds,
            common['trade_freq'],
            common['target_vol'],
            common['stop_loss_pct'],
            common['take_profit_pct'],
            common['max_drawdown_pct'],
            common['sizing_type'],
            common['kelly_fraction'],
            common['kelly_lookback'],
        ]
        
        for (zscore_lookback, n_std_upper, n_std_lower, exit_threshold,
             trade_freq, target_vol, stop_loss, take_profit, max_dd,
             sizing_type, kelly_fraction, kelly_lookback) in product(*param_lists):
            task = base.copy()
            task.update({
                'strategy_type': StrategyType.MEAN_REVERSION,
                'zscore_lookback': zscore_lookback,
                'n_std_upper': n_std_upper,
                'n_std_lower': n_std_lower,
                'exit_threshold': exit_threshold,
                'trade_freq': trade_freq,
                'target_vol': target_vol,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
                'max_drawdown_pct': max_dd,
                'sizing_type': sizing_type,
                'kelly_fraction': kelly_fraction,
                'kelly_lookback': kelly_lookback,
            })
            tasks.append(task)
        
        return tasks
    
    def _generate_mean_reversion_rsi_tasks(
        self,
        ticker: str,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        market_calendar: Dict,
        risk_free_rate_series: Optional[pd.Series]
    ) -> List[Dict[str, Any]]:
        """Generate RSI mean reversion strategy tasks."""
        tasks = []
        common = self._get_common_params()
        base = self._base_task(ticker, df, df_daily, market_calendar, risk_free_rate_series)
        rsi = self.config.mean_reversion_rsi
        
        param_lists = [
            rsi.rsi_periods,
            rsi.rsi_oversold_levels,
            rsi.rsi_overbought_levels,
            rsi.sma_periods,
            common['trade_freq'],
            common['target_vol'],
            common['stop_loss_pct'],
            common['take_profit_pct'],
            common['max_drawdown_pct'],
            common['sizing_type'],
            common['kelly_fraction'],
            common['kelly_lookback'],
        ]
        
        for (rsi_period, rsi_oversold, rsi_overbought, sma_period,
             trade_freq, target_vol, stop_loss, take_profit, max_dd,
             sizing_type, kelly_fraction, kelly_lookback) in product(*param_lists):
            task = base.copy()
            task.update({
                'strategy_type': StrategyType.MEAN_REVERSION_RSI,
                'rsi_period': rsi_period,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought,
                'sma_period': sma_period,
                'exit_on_sma_cross': True,
                'trade_freq': trade_freq,
                'target_vol': target_vol,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
                'max_drawdown_pct': max_dd,
                'sizing_type': sizing_type,
                'kelly_fraction': kelly_fraction,
                'kelly_lookback': kelly_lookback,
            })
            tasks.append(task)
        
        return tasks
    
    def _generate_stat_arb_tasks(
        self,
        ticker_a: str,
        ticker_b: str,
        df_a: pd.DataFrame,
        df_daily_a: pd.DataFrame,
        df_b: pd.DataFrame,
        df_daily_b: pd.DataFrame,
        market_calendar: Dict,
        risk_free_rate_series: Optional[pd.Series]
    ) -> List[Dict[str, Any]]:
        """Generate statistical arbitrage strategy tasks."""
        tasks = []
        common = self._get_common_params()
        sa = self.config.stat_arb
        c = self.config.common
        
        param_lists = [
            sa.zscore_lookbacks,
            sa.entry_thresholds,
            sa.exit_thresholds,
            sa.use_dynamic_hedge_options,
            common['trade_freq'],
            common['target_vol'],
            common['stop_loss_pct'],
            common['take_profit_pct'],
            common['max_drawdown_pct'],
            common['sizing_type'],
            common['kelly_fraction'],
            common['kelly_lookback'],
        ]
        
        for (zscore_lookback, entry_threshold, exit_threshold, use_dynamic_hedge,
             trade_freq, target_vol, stop_loss, take_profit, max_dd,
             sizing_type, kelly_fraction, kelly_lookback) in product(*param_lists):
            task = {
                'ticker': ticker_a,
                'ticker_b': ticker_b,
                'df': df_a,
                'df_daily': df_daily_a,
                'df_b': df_b,
                'df_b_daily': df_daily_b,
                'market_calendar': market_calendar,
                'risk_free_rate_series': risk_free_rate_series,
                'commission': c.commission,
                'slippage_factor': c.slippage_factor,
                'max_leverage': c.max_leverage,
                'initial_aum': c.initial_aum,
                'lookback_period': c.lookback_period,
                'kelly_min_trades': c.kelly_min_trades,
                'kelly_vol_blend_weight': c.kelly_vol_blend_weight,
                'strategy_type': StrategyType.STAT_ARB,
                'zscore_lookback': zscore_lookback,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'use_dynamic_hedge': use_dynamic_hedge,
                'trade_freq': trade_freq,
                'target_vol': target_vol,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
                'max_drawdown_pct': max_dd,
                'sizing_type': sizing_type,
                'kelly_fraction': kelly_fraction,
                'kelly_lookback': kelly_lookback,
            }
            tasks.append(task)
        
        return tasks
    
    def _generate_supervised_tasks(
        self,
        ticker: str,
        df: pd.DataFrame,
        df_daily: pd.DataFrame,
        market_calendar: Dict,
        risk_free_rate_series: Optional[pd.Series]
    ) -> List[Dict[str, Any]]:
        """
        Generate supervised ML strategy tasks.
        
        OPTIMIZATION: Pre-compute model probabilities ONCE, then generate signals
        for each threshold combination without reloading the model or re-running
        feature generation.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        tasks = []
        common = self._get_common_params()
        base = self._base_task(ticker, df, df_daily, market_calendar, risk_free_rate_series)
        sup = self.config.supervised
        
        # Validate model paths
        if not sup.model_paths:
            return tasks
        
        if len(sup.scaler_paths) != len(sup.model_paths):
            raise ValueError("scaler_paths must have same length as model_paths")
        if len(sup.features_paths) != len(sup.model_paths):
            raise ValueError("features_paths must have same length as model_paths")
        
        # Create model path tuples
        model_configs = list(zip(sup.model_paths, sup.scaler_paths, sup.features_paths))
        
        # PRE-COMPUTE: Load model and generate probabilities ONCE per model
        # This avoids loading the model 162+ times
        precomputed_probabilities = {}
        
        for model_path, scaler_path, features_path in model_configs:
            logger.info(f"Pre-computing ML probabilities for {ticker}...")
            
            try:
                from strats.classification import SupervisedStrategy
                
                # Load model once
                strategy = SupervisedStrategy.from_artifacts(
                    model_path=model_path,
                    scaler_path=scaler_path,
                    features_path=features_path,
                    buy_threshold=0.5,
                    sell_threshold=0.5
                )
                
                # Generate features and probabilities for entire dataset
                probabilities = strategy.get_probabilities(df)
                
                if probabilities is not None and len(probabilities) > 0:
                    precomputed_probabilities[model_path] = probabilities
                    logger.info(f"Pre-computed {len(probabilities)} probabilities")
                else:
                    logger.warning(f"Failed to pre-compute probabilities, falling back to per-task computation")
                    precomputed_probabilities[model_path] = None
                    
            except Exception as e:
                logger.warning(f"Pre-computation failed: {e}, falling back to per-task computation")
                precomputed_probabilities[model_path] = None
        
        param_lists = [
            model_configs,
            sup.buy_thresholds,
            sup.sell_thresholds,
            common['trade_freq'],
            common['target_vol'],
            common['stop_loss_pct'],
            common['take_profit_pct'],
            common['max_drawdown_pct'],
            common['sizing_type'],
            common['kelly_fraction'],
            common['kelly_lookback'],
        ]
        
        for ((model_path, scaler_path, features_path), buy_threshold, sell_threshold,
             trade_freq, target_vol, stop_loss, take_profit, max_dd,
             sizing_type, kelly_fraction, kelly_lookback) in product(*param_lists):
            task = base.copy()
            task.update({
                'strategy_type': StrategyType.SUPERVISED,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'features_path': features_path,
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'trade_freq': trade_freq,
                'target_vol': target_vol,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
                'max_drawdown_pct': max_dd,
                'sizing_type': sizing_type,
                'kelly_fraction': kelly_fraction,
                'kelly_lookback': kelly_lookback,
            })
            
            if model_path in precomputed_probabilities:
                task['precomputed_probabilities'] = precomputed_probabilities[model_path]
            
            tasks.append(task)
        
        return tasks


def build_backtest_config(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a backtest configuration dictionary from a task.
    
    This converts a task dictionary into the format expected by
    BacktestEngine.run().
    
    Args:
        task: Task dictionary from TaskGenerator
        
    Returns:
        Configuration dictionary for backtest
    """
    strategy_type = task['strategy_type']
    
    config = {
        'strategy_type': strategy_type,
        'AUM': task.get('initial_aum', 100000.0),
        'commission': task.get('commission', 0.0035),
        'min_comm_per_order': task.get('min_comm_per_order', 0.35),
        'trade_freq': task['trade_freq'],
        'sizing_type': task.get('sizing_type', 'vol_target'),
        'target_vol': task['target_vol'],
        'kelly_fraction': task.get('kelly_fraction', 0.5),
        'kelly_lookback': task.get('kelly_lookback', 60),
        'kelly_min_trades': task.get('kelly_min_trades', 30),
        'kelly_vol_blend_weight': task.get('kelly_vol_blend_weight', 0.5),
        'max_leverage': task.get('max_leverage', 1),
        'stop_loss_pct': task['stop_loss_pct'],
        'take_profit_pct': task['take_profit_pct'],
        'max_drawdown_pct': task['max_drawdown_pct'],
        'lookback_period': task.get('lookback_period', 20),
        'slippage_factor': task.get('slippage_factor', 0.1),
        'var_confidence': task.get('var_confidence', 0.95),
    }
    
    # Add strategy-specific parameters
    if strategy_type == StrategyType.MOMENTUM:
        config['band_mult'] = task['band_mult']

    elif strategy_type == StrategyType.MOMENTUM_ATR:
        config['atr_mult'] = task['atr_mult']
        config['atr_period'] = task['atr_period']
    
    elif strategy_type == StrategyType.MEAN_REVERSION:
        config['zscore_lookback'] = task['zscore_lookback']
        config['n_std_upper'] = task['n_std_upper']
        config['n_std_lower'] = task['n_std_lower']
        config['exit_threshold'] = task.get('exit_threshold', 0.0)
    
    elif strategy_type == StrategyType.MEAN_REVERSION_RSI:
        config['rsi_period'] = task['rsi_period']
        config['rsi_oversold'] = task['rsi_oversold']
        config['rsi_overbought'] = task['rsi_overbought']
        config['sma_period'] = task['sma_period']
        config['exit_on_sma_cross'] = task.get('exit_on_sma_cross', True)
    
    elif strategy_type == StrategyType.STAT_ARB:
        config['zscore_lookback'] = task['zscore_lookback']
        config['entry_threshold'] = task['entry_threshold']
        config['exit_threshold'] = task.get('exit_threshold', 0.0)
        config['use_dynamic_hedge'] = task.get('use_dynamic_hedge', True)
        config['hedge_ratio'] = None
    
    elif strategy_type == StrategyType.SUPERVISED:
        config['model_path'] = task['model_path']
        config['scaler_path'] = task['scaler_path']
        config['features_path'] = task['features_path']
        config['buy_threshold'] = task.get('buy_threshold', 0.55)
        config['sell_threshold'] = task.get('sell_threshold', 0.45)
        if 'precomputed_probabilities' in task:
            config['precomputed_probabilities'] = task['precomputed_probabilities']
    
    return config
