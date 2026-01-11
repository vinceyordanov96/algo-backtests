#!/usr/bin/env python3
# algo-backtests/examples/momentum.py
"""
Momentum Strategy Example - Complete Backtesting Workflow.

This script demonstrates:
    1. Loading market data for a ticker
    2. Running a parameter sweep simulation
    3. Finding the best performing configuration
    4. Running the best strategy and saving all outputs

Usage:
    python examples/momentum.py
    
Outputs:
    - outputs/strategies/momentum/simulations/{ticker}/  - Simulation results
    - outputs/strategies/momentum/signals/{ticker}/  - Best strategy signals
    - outputs/strategies/momentum/results/{ticker}/  - Portfolio results
    - outputs/strategies/momentum/plots/{ticker}/  - Performance plots
"""

import os
import sys
import logging
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def main():
    """Main entry point for the momentum strategy example."""
    
    print("\n\n" + "=" * 80)
    print("MOMENTUM STRATEGY - BACKTESTING")
    print("=" * 80)
    print()
    
    from connectors import DataFetcher, DataManager
    from simulation import SimulationRunner, SimulationConfig, StrategyType
    from simulation.config import CommonParameters, MomentumParameters
    from outputs import StrategyOutputManager
    
    # =========================================================================
    # Step 1: Configuration
    # =========================================================================
    print("\n" + "-" * 80)
    print("[Step 1] Setting up configuration...")
    print("-" * 80)
    
    TICKER = 'NVDA'
    DATA_SOURCE = 'polygon'
    STRATEGY = StrategyType.MOMENTUM
    
    # Common parameters (shared across strategies)
    common_params = CommonParameters(
        trade_frequencies=[30, 60],
        target_volatilities=[0.015, 0.02, 0.025, 0.03],
        stop_loss_pcts=[0.015, 0.02, 0.025],
        take_profit_pcts=[0.03, 0.035, 0.04],
        max_drawdown_pcts=[0.15, 0.20, 0.25],
        initial_aum=100000.0,
        
        # Kelly position sizing
        sizing_types=['kelly'],              # or ['kelly_vol_blend'] or ['vol_target', 'kelly', 'kelly_vol_blend'] to compare all
        kelly_fractions=[0.5],               # Half-Kelly
        kelly_lookbacks=[60],                # 60 and 120 day lookbacks
        kelly_min_trades=30,                 # Minimum observations before Kelly activates
        kelly_vol_blend_weight=0.5,          # 50/50 blend if using kelly_vol_blend_
    )
    
    # Momentum-specific parameters
    momentum_params = MomentumParameters(
        band_multipliers=[0.9, 1.0, 1.1],
    )
    
    # Create simulation config
    simulation_config = SimulationConfig(
        strategy_type=STRATEGY,
        tickers=[TICKER],
        common=common_params,
        momentum=momentum_params,
        n_workers=None  # Auto-detect
    )
    
    print(f"  ✓ Ticker: {TICKER}")
    print(f"  ✓ Strategy: {STRATEGY.value}")
    print(f"  ✓ Band multipliers: {momentum_params.band_multipliers}")
    print(f"  ✓ Trade frequencies: {common_params.trade_frequencies}")
    print(f"  ✓ Target volatilities: {common_params.target_volatilities}")
    print(f"  ✓ Stop loss range: {common_params.stop_loss_pcts}")
    print(f"  ✓ Take profit range: {common_params.take_profit_pcts}")
    print(f"  ✓ Expected combinations: {simulation_config.count_combinations()}")
    print("-" * 80)
    
    # =========================================================================
    # Step 2: Load market data
    # =========================================================================
    print("\n\n" + "-" * 80)
    print("[Step 2] Loading market data...")
    print("-" * 80)
    
    data_manager = DataManager()
    
    # Try to load existing data FIRST
    df_intraday = data_manager.load_intraday(TICKER)
    df_daily = data_manager.load_daily(TICKER)
    
    if df_intraday is not None and df_daily is not None:
        print(f"✓ Loaded existing data:")
        print(f"  - {len(df_intraday)} intraday records")
        print(f"  - {len(df_daily)} daily records")
        print(f"  - Date range: {df_intraday.index.min()} to {df_intraday.index.max()}")
    else:
        print("  No existing data found, fetching from API...")
        
        try:
            fetcher = DataFetcher(TICKER, source=DATA_SOURCE)
            df_intraday, df_daily = fetcher.process_data()
            
            if df_intraday.empty:
                print(f"  ✗ Failed to fetch data for {TICKER}")
                print("  Note: yfinance only provides ~60 days of intraday data")
                return
            
            print(f"✓ Fetched {len(df_intraday)} intraday records")
            print(f"✓ Fetched {len(df_daily)} daily records")
            print(f"✓ Date range: {df_intraday.index.min()} to {df_intraday.index.max()}")
            
            # Save data using DataManager
            paths = data_manager.save_all(TICKER, df_intraday, df_daily)
            
            print(f"✓ Data saved to:")
            for data_type, path in paths.items():
                print(f"  - {data_type}: {path}")
        
        except Exception as e:
            print(f"  ✗ Error fetching data: {e}")
            print(f"  ✗ No data available for {TICKER}")
            return
    
    print("-" * 80)
    
    # =========================================================================
    # Step 3: Configure simulation
    # =========================================================================
    print("\n\n" + "-" * 80)
    print("[Step 3] Configuring simulation...")
    print("-" * 80)
    
    n_backtests = simulation_config.count_combinations()
    
    print(f"  ✓ Strategy: MOMENTUM")
    print(f"  ✓ Band multipliers: {momentum_params.band_multipliers}")
    print(f"  ✓ Trade frequencies: {common_params.trade_frequencies} min")
    print(f"  ✓ Target volatilities: {common_params.target_volatilities}")
    print(f"  ✓ Stop loss range: {common_params.stop_loss_pcts}")
    print(f"  ✓ Take profit range: {common_params.take_profit_pcts}")
    print(f"  ✓ Expected combinations: {n_backtests}")
    print("-" * 80)
    
    # =========================================================================
    # Step 4: Run simulation
    # =========================================================================
    print("\n\n" + "-" * 80)
    print("[Step 4] Running simulation parameter sweep...")
    print("-" * 80)
    
    runner = SimulationRunner(
        config=simulation_config,
        data_dir='outputs/data',
        output_dir='outputs/strategies'
    )
    
    # Load data into runner's cache
    runner._data_cache[TICKER] = (df_intraday, df_daily)
    
    start_time = datetime.now()
    results_df = runner.run(parallel=True)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    if results_df.empty:
        print("  ✗ No simulation results generated")
        return
    
    print("-" * 80)
    
    # =========================================================================
    # Step 5: Analyze results
    # =========================================================================
    print("\n\n" + "-" * 80)
    print("[Step 5] Analyzing results...")
    print("-" * 80)
    
    best_df = runner.get_best_by_ticker(results_df, metric='Sharpe Ratio (Strategy)')
    
    if best_df.empty:
        print("  ✗ Could not determine best configuration")
        return
    
    best_row = best_df.iloc[0]
    
    print("Best configuration found:")
    print(f"  - Band Multiplier: {best_row.get('Band Multiplier', 'N/A')}")
    print(f"  - Trade Frequency: {best_row.get('Trading Frequency', 'N/A')} min")
    print(f"  - Target Volatility: {best_row.get('Target Volatility', 'N/A')}")
    print(f"  - Stop Loss: {best_row.get('Stop Loss', 'N/A')}")
    print(f"  - Take Profit: {best_row.get('Take Profit', 'N/A')}")
    print(f"  - Sharpe Ratio: {best_row.get('Sharpe Ratio (Strategy)', 0):.4f}")
    print(f"  - Total Return: {best_row.get('Total % Return (Strategy)', 0):.2f}%")
    print(f"  - Max Drawdown: {best_row.get('Max Drawdown (%)', 0):.2f}%")
    print(f"  - Win Rate: {best_row.get('Win Rate (%)', 0):.2f}%")
    print(f"  - Total Trades: {best_row.get('Total Trades', 0)}")
    print("-" * 80)
    
    # =========================================================================
    # Step 6: Run best strategy and save outputs
    # =========================================================================
    print("\n\n" + "-" * 80)
    print("[Step 6] Running best strategy and saving outputs...")
    print("-" * 80)
    
    output = runner.run_best_strategy(
        results_df=results_df,
        ticker=TICKER,
        metric='Sharpe Ratio (Strategy)',
        save_outputs=True,
        show_plot=False,
        verbose=False
    )
    
    if output is None:
        print("  ✗ Failed to run best strategy")
        return
    
    print("-" * 80)
    
    # =========================================================================
    # Step 7: Display summary
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("[Step 7] Summary of best strategy performance")
    print("=" * 80)
    
    stats = output['statistics']
    
    print("\n\n" + "-" * 80)
    print(f"{TICKER} STRATEGY PERFORMANCE")
    print("-" * 80)
    print(f"Total Return:        {stats.total_return * 100:>10.2f}%")
    print(f"Benchmark Return:   {stats.benchmark_return * 100:>10.2f}%")
    print(f"Sharpe Ratio:        {stats.sharpe_ratio:>10.4f}")
    print(f"Sortino Ratio:       {stats.sortino_ratio:>10.4f}")
    print(f"Calmar Ratio:        {stats.calmar_ratio:>10.4f}")
    print(f"Max Drawdown:        {stats.max_drawdown * 100:>10.2f}%")
    print(f"Annualized Vol:     {stats.annualized_volatility * 100:>10.2f}%")
    print(f"Win Rate:           {stats.win_rate * 100:>10.2f}%")
    print(f"Total Trades:     {stats.total_trades:>10d}")
    print("-" * 80)
    
    print("\n\n" + "-" * 80)
    print("BENCHMARK COMPARISON")
    print("-" * 80)
    print(f"Alpha:              {stats.alpha * 100:>10.2f}%" if not np.isnan(stats.alpha) else "Alpha:               N/A")
    print(f"Beta:                {stats.beta:>10.4f}" if not np.isnan(stats.beta) else "Beta:                N/A")
    print(f"Correlation:         {stats.correlation:>10.4f}" if not np.isnan(stats.correlation) else "Correlation:         N/A")
    print(f"R-Squared:           {stats.r_squared:>10.4f}" if not np.isnan(stats.r_squared) else "R-Squared:           N/A")
    print(f"Information Ratio:   {stats.information_ratio:>10.4f}" if not np.isnan(stats.information_ratio) else "Information Ratio:   N/A")
    print(f"Treynor Ratio:       {stats.treynor_ratio:>10.4f}" if not np.isnan(stats.treynor_ratio) else "Treynor Ratio:       N/A")
    print("-" * 80)
    
    print("\n\n" + "-" * 80)
    print("STRATEGY CONFIGURATION")
    print("-" * 80)
    print(f"Strategy:            {STRATEGY.value}")
    print(f"Band Multiplier:     {best_row.get('Band Multiplier', 'N/A')}")
    print(f"Trade Frequency:     {best_row.get('Trading Frequency', 'N/A')} min")
    print(f"Target Volatility:   {best_row.get('Target Volatility', 'N/A')}")
    print(f"Stop Loss:           {best_row.get('Stop Loss', 'N/A')}")
    print(f"Take Profit:         {best_row.get('Take Profit', 'N/A')}")
    print("-" * 80)
    
    print("\n\n" + "-" * 80)
    print("OUTPUT FILES")
    print("-" * 80)
    
    output_manager = StrategyOutputManager()
    
    print("Data files:")
    data_info = data_manager.get_data_info(TICKER, 'intraday')
    if data_info:
        print(f"  - Intraday: {data_info['filepath']}")
    
    print("\nStrategy files:")
    for output_type in ['simulations', 'signals', 'results', 'plots']:
        info = output_manager.get_output_info('momentum', TICKER, output_type)
        if info:
            print(f"  - {output_type.capitalize()}: {info['filepath']}")
    
    print("\n\n" + "=" * 80)
    print("Momentum Strategy Example completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
