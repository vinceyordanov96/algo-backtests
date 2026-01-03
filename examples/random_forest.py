#!/usr/bin/env python3
# algo-backtests/examples/random_forest.py
"""
ML Strategy Example - Complete Training and Backtesting Workflow.

This script demonstrates:
    1. Training an ML model using TrainingPipeline (or loading existing)
    2. Saving model artifacts via ArtifactManager
    3. Running a backtest simulation with the SUPERVISED strategy

CRITICAL ALIGNMENT CONSIDERATIONS:
----------------------------------
The forecast_horizon, resample_freq, and trading_frequency must be aligned:

    forecast_horizon: How many bars ahead the model predicts
    resample_freq: The bar size used for features and predictions  
    trading_frequency: How often the backtest evaluates trading decisions

For consistency:
    - If forecast_horizon = 12 bars and resample_freq = 5min,
      then the model predicts 12 * 5 = 60 minutes ahead
    - The trading_frequency should match: 60 minutes

Usage:
    python ml_example.py
    
    # Force retraining even if model exists:
    python ml_example.py --retrain
    
Outputs:
    - outputs/models/{ticker}/  - Model artifacts
    - outputs/strategies/supervised/simulations/{ticker}/  - Simulation results
    - outputs/strategies/supervised/signals/{ticker}/  - Best strategy signals
    - outputs/strategies/supervised/results/{ticker}/  - Portfolio results
    - outputs/strategies/supervised/plots/{ticker}/  - Performance plots
"""

import os
import sys
import logging
import warnings
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


def find_latest_model(ticker: str, model_type: str, models_dir: str = 'outputs/models') -> dict:
    """
    Find the most recent model artifacts for a ticker.
    
    Args:
        ticker: Stock ticker
        model_type: Model type (e.g., 'xgboost', 'gradient_boosting')
        models_dir: Base directory for models
        
    Returns:
        Dictionary with artifact paths, or None if not found
    """
    ticker_dir = os.path.join(models_dir, ticker)
    
    if not os.path.exists(ticker_dir):
        return None
    
    # Find all model files matching pattern
    pattern = f'{ticker}_{model_type}_'
    model_files = [
        f for f in os.listdir(ticker_dir)
        if f.startswith(pattern) and f.endswith('_model.pkl')
    ]
    
    if not model_files:
        return None
    
    # Sort by timestamp in filename (newest first)
    model_files.sort(reverse=True)
    latest = model_files[0]
    
    # Extract base name
    base_name = latest.replace('_model.pkl', '')
    
    # Build paths dictionary
    paths = {
        'model_path': os.path.join(ticker_dir, f'{base_name}_model.pkl'),
        'scaler_path': os.path.join(ticker_dir, f'{base_name}_scalers.pkl'),
        'features_path': os.path.join(ticker_dir, f'{base_name}_features.pkl'),
        'selector_path': os.path.join(ticker_dir, f'{base_name}_selector.pkl'),
        'config_path': os.path.join(ticker_dir, f'{base_name}_config.pkl'),
    }
    
    # Verify all required files exist
    required = ['model_path', 'scaler_path', 'features_path']
    for key in required:
        if not os.path.exists(paths[key]):
            return None
    
    return paths


def main():
    """Main entry point for ML example."""
    
    # Check for --retrain flag
    force_retrain = '--retrain' in sys.argv
    
    print("=" * 70)
    print("ML STRATEGY - TRAINING AND BACKTESTING")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Step 1: Import required modules
    # =========================================================================
    print("[Step 1] Importing modules...")
    
    from connectors import DataManager
    from ml import TrainingPipeline, TrainingConfig
    from simulation import SimulationRunner, SimulationConfig, StrategyType
    from simulation.config import CommonParameters, SupervisedParameters
    from outputs import StrategyOutputManager
    
    print("  ✓ Modules imported successfully")
    print()
    
    # =========================================================================
    # Step 2: Configuration
    # =========================================================================
    print("[Step 2] Setting up configuration...")
    
    TICKER = 'TSLA'
    
    # ML Configuration
    # ----------------
    # CRITICAL: These parameters must be aligned!
    # 
    # resample_freq = 5min means we work with 5-minute bars
    # forecast_horizon = 12 means we predict 12 bars (= 60min) ahead
    # trading_frequency = 60 means we trade every 60 minutes
    #
    # This ensures: prediction horizon (12 * 5min = 60min) == trading frequency (60min)
    
    RESAMPLE_FREQ = '5min'     # Bar size for features
    FORECAST_HORIZON = 12      # Bars ahead (12 * 5min = 60min)
    TRADING_FREQUENCY = 60     # Minutes between trades (must match forecast horizon)
    MODEL_TYPE = 'random_forest'
    
    # Calculate prediction horizon in minutes
    resample_minutes = int(RESAMPLE_FREQ.replace('min', ''))
    prediction_horizon_minutes = FORECAST_HORIZON * resample_minutes
    
    print(f"  ✓ Ticker: {TICKER}")
    print(f"  ✓ Resample frequency: {RESAMPLE_FREQ}")
    print(f"  ✓ Forecast horizon: {FORECAST_HORIZON} bars ({prediction_horizon_minutes} minutes)")
    print(f"  ✓ Trading frequency: {TRADING_FREQUENCY} minutes")
    print(f"  ✓ Model type: {MODEL_TYPE}")
    print()
    
    # =========================================================================
    # Step 3: Load or Train ML model
    # =========================================================================
    print("[Step 3] Loading or training ML model...")
    
    artifact_paths = None
    result = None
    
    # Check if model already exists
    if not force_retrain:
        artifact_paths = find_latest_model(TICKER, MODEL_TYPE)
        
        if artifact_paths:
            print(f"  ✓ Found existing model artifacts:")
            for name, path in artifact_paths.items():
                if os.path.exists(path):
                    print(f"      - {name}: {path.split('/')[-1]}")
            print()
            print("  ℹ To retrain, run with --retrain flag")
    
    # Train if no existing model or force retrain
    if artifact_paths is None or force_retrain:
        if force_retrain:
            print("  ℹ Force retrain requested")
        else:
            print("  ℹ No existing model found, training new model...")
        
        # Create training configuration
        training_config = TrainingConfig(
            ticker=TICKER,
            model_type=MODEL_TYPE,
            forecast_horizon=FORECAST_HORIZON,
            resample_freq=RESAMPLE_FREQ,
            train_ratio=0.6,
            valid_ratio=0.2,
            test_ratio=0.2,
            output_dir=f'outputs/models/{TICKER}',
        )
        
        # Run training pipeline
        pipeline = TrainingPipeline(training_config)
        result = pipeline.run(data_dir='outputs/data')
        
        artifact_paths = result.paths
        
        print(f"  ✓ Model trained successfully")
        print(f"      - Features selected: {result.feature_count}")
        print(f"      - Validation Accuracy: {result.valid_metrics.accuracy:.4f}")
        print(f"      - Validation ROC-AUC: {result.valid_metrics.roc_auc:.4f}")
        print(f"      - Validation Precision: {result.valid_metrics.precision:.4f}")
        print(f"      - Validation Recall: {result.valid_metrics.recall:.4f}")
    
    print()
    
    # =========================================================================
    # Step 4: Display artifact paths
    # =========================================================================
    print("[Step 4] Model artifacts...")
    
    print(f"  ✓ Artifacts in outputs/models/{TICKER}/")
    for name, path in artifact_paths.items():
        if os.path.exists(path):
            print(f"      - {name}: {path.split('/')[-1]}")
    print()
    
    # =========================================================================
    # Step 5: Load data for simulation
    # =========================================================================
    print("[Step 5] Loading data for simulation...")
    
    data_manager = DataManager()
    
    # Load resampled data (matching training)
    df_resampled = data_manager.load_intraday_resampled(TICKER, interval=RESAMPLE_FREQ)
    df_daily = data_manager.load_daily(TICKER)
    
    if df_resampled is None or df_daily is None:
        print(f"  ✗ Failed to load data for {TICKER}")
        return
    
    print(f"  DEBUG: Intraday date range: {df_resampled['day'].min()} to {df_resampled['day'].max()}")
    print(f"  DEBUG: Daily data rows: {len(df_daily)}")
    if 'caldt' in df_daily.columns:
        print(f"  DEBUG: Daily date range: {df_daily['caldt'].min()} to {df_daily['caldt'].max()}")
    else:
        print(f"  DEBUG: Daily index range: {df_daily.index.min()} to {df_daily.index.max()}")
    print()
    
    print(f"  ✓ Loaded resampled data: {len(df_resampled)} bars ({RESAMPLE_FREQ})")
    print(f"  ✓ Loaded daily data: {len(df_daily)} records")
    print()
    
    # =========================================================================
    # Step 6: Configure simulation
    # =========================================================================
    print("[Step 6] Configuring simulation...")
    
    # For supervised strategy, trading frequency is FIXED to match forecast horizon
    # We can only sweep over:
    #   - buy_threshold / sell_threshold (model output interpretation)
    #   - stop_loss_pct / take_profit_pct / max_drawdown_pct (risk management)
    
    common_params = CommonParameters(
        trade_frequencies=[TRADING_FREQUENCY],  # FIXED - must match forecast horizon
        target_volatilities=[0.02],             # Fixed for simplicity
        stop_loss_pcts=[0.015, 0.02, 0.025],    # Can sweep
        take_profit_pcts=[0.03, 0.04, 0.05],    # Can sweep
        max_drawdown_pcts=[0.15, 0.20],         # Can sweep
        initial_aum=100000.0,
    )
    
    supervised_params = SupervisedParameters(
        model_paths=[artifact_paths['model_path']],
        scaler_paths=[artifact_paths['scaler_path']],
        features_paths=[artifact_paths['features_path']],
        buy_thresholds=[0.50, 0.55, 0.60],      # Can sweep
        sell_thresholds=[0.40, 0.45, 0.50],     # Can sweep
    )
    
    simulation_config = SimulationConfig(
        strategy_type=StrategyType.SUPERVISED,
        tickers=[TICKER],
        common=common_params,
        supervised=supervised_params,
        n_workers=None  # Auto-detect
    )
    
    n_combinations = simulation_config.count_combinations()
    
    print(f"  ✓ Strategy: SUPERVISED (ML)")
    print(f"  ✓ Trading frequency: {TRADING_FREQUENCY} min (fixed to match forecast horizon)")
    print(f"  ✓ Buy thresholds: {supervised_params.buy_thresholds}")
    print(f"  ✓ Sell thresholds: {supervised_params.sell_thresholds}")
    print(f"  ✓ Stop loss range: {common_params.stop_loss_pcts}")
    print(f"  ✓ Take profit range: {common_params.take_profit_pcts}")
    print(f"  ✓ Expected combinations: {n_combinations}")
    print()
    
    # =========================================================================
    # Step 7: Run simulation
    # =========================================================================
    print("[Step 7] Running simulation parameter sweep...")
    print(f"  ✓ Testing {n_combinations} parameter combinations...")
    
    runner = SimulationRunner(
        config=simulation_config,
        data_dir='outputs/data',
        output_dir='outputs/strategies'
    )
    
    # Load resampled data into runner's cache
    # IMPORTANT: Use the resampled data that matches training
    runner._data_cache[TICKER] = (df_resampled, df_daily)
    
    start_time = datetime.now()
    results_df = runner.run(parallel=True)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    if results_df.empty:
        print("  ✗ No simulation results generated")
        print("  ℹ Check the warnings above for feature generation issues")
        return
    
    print(f"  ✓ Completed {len(results_df)} backtests in {elapsed:.1f}s")
    print()
    
    # DEBUG: Check for NaN issues in results
    print(f"  DEBUG: Results rows: {len(results_df)}")
    print(f"  DEBUG: Unique benchmark returns: {results_df['Total % Return (Buy & Hold)'].unique()}")
    
    # =========================================================================
    # Step 8: Analyze results
    # =========================================================================
    print("[Step 8] Analyzing results...")
    
    best_df = runner.get_best_by_ticker(results_df, metric='Sharpe Ratio (Strategy)')
    
    if best_df.empty:
        print("  ✗ Could not determine best configuration")
        return
    
    best_row = best_df.iloc[0]
    
    print(f"  ✓ Best configuration found:")
    print(f"      - Buy Threshold: {best_row.get('Buy Threshold', 'N/A')}")
    print(f"      - Sell Threshold: {best_row.get('Sell Threshold', 'N/A')}")
    print(f"      - Stop Loss: {best_row.get('Stop Loss', 'N/A')}")
    print(f"      - Take Profit: {best_row.get('Take Profit', 'N/A')}")
    print(f"      - Sharpe Ratio: {best_row.get('Sharpe Ratio (Strategy)', 0):.4f}")
    print(f"      - Total Return: {best_row.get('Total % Return (Strategy)', 0):.2f}%")
    print(f"      - Max Drawdown: {best_row.get('Max Drawdown (%)', 0):.2f}%")
    print(f"      - Win Rate: {best_row.get('Win Rate (%)', 0):.2f}%")
    print(f"      - Total Trades: {best_row.get('Total Trades', 0)}")
    print()
    
    # =========================================================================
    # Step 9: Run best strategy and save outputs
    # =========================================================================
    print("[Step 9] Running best strategy and saving outputs...")
    
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
    
    print(f"  ✓ Best strategy backtest completed")
    print(f"  ✓ Outputs saved to:")
    for output_type, path in output['paths'].items():
        print(f"      - {output_type}: {path}")
    print()
    
    # =========================================================================
    # Step 10: Display summary
    # =========================================================================
    print("[Step 10] Summary")
    print("=" * 70)
    
    stats = output['statistics']
    
    print(f"\n{TICKER} SUPERVISED ML STRATEGY RESULTS")
    print("-" * 40)
    print(f"Total Return:        {stats.total_return * 100:>10.2f}%")
    print(f"Benchmark Return:    {stats.benchmark_return * 100:>10.2f}%")
    print(f"Sharpe Ratio:        {stats.sharpe_ratio:>10.4f}")
    print(f"Sortino Ratio:       {stats.sortino_ratio:>10.4f}")
    print(f"Calmar Ratio:        {stats.calmar_ratio:>10.4f}")
    print(f"Max Drawdown:        {stats.max_drawdown * 100:>10.2f}%")
    print(f"Annualized Vol:      {stats.annualized_volatility * 100:>10.2f}%")
    print(f"Win Rate:            {stats.win_rate * 100:>10.2f}%")
    print(f"Total Trades:        {stats.total_trades:>10d}")
    print("-" * 40)
    
    print("\nMODEL CONFIGURATION")
    print("-" * 40)
    print(f"Resampling:          {RESAMPLE_FREQ}")
    print(f"Forecast Horizon:    {FORECAST_HORIZON} bars ({prediction_horizon_minutes} min)")
    print(f"Trading Frequency:   {TRADING_FREQUENCY} min")
    print(f"Model Type:          {MODEL_TYPE}")
    if result:
        print(f"Features Used:       {result.feature_count}")
    print("-" * 40)
    
    print("\nOUTPUT FILES")
    print("-" * 40)
    
    print("Model artifacts:")
    for name, path in artifact_paths.items():
        if os.path.exists(path):
            print(f"  - {name}: {path}")
    
    output_manager = StrategyOutputManager()
    
    print("\nStrategy files:")
    for output_type in ['simulations', 'signals', 'results', 'plots']:
        info = output_manager.get_output_info('supervised', TICKER, output_type)
        if info:
            print(f"  - {output_type.capitalize()}: {info['filepath']}")
    
    print("\n" + "=" * 70)
    print("ML Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
