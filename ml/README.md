# ML Module

Supervised machine learning pipeline for classification-based trading strategies.

## Overview

This module provides a complete ML training pipeline that:

1. Generates technical features from OHLCV data
2. Creates binary classification labels (price up/down)
3. Performs feature selection and normalization
4. Trains tree-based models (RandomForest, GradientBoosting, XGBoost)
5. Saves artifacts for backtesting integration

**Critical Design Principle**: All fitting (feature selection, normalization) happens on training data only. Test and validation sets are transformed using fitted parameters to prevent data leakage.

---

## Components

### 1. TrainingPipeline

Main entry point that orchestrates the complete training workflow.

**Pipeline Steps:**
1. Load data
2. Generate features
3. Create labels
4. Temporal split (before any fitting)
5. Feature selection (fit on train only)
6. Normalization (fit on train only)
7. Train model
8. Evaluate on test/validation
9. Save artifacts

#

### 2. FeatureGeneration

Generates 60+ technical indicators from OHLCV data.

**Feature Categories:**
- Lagged values (prev_open, prev_close, etc.)
- Rolling volatility (5, 10, 20, 30, 50, 60 periods)
- Moving averages (SMA and EMA at multiple windows)
- Log returns
- RSI (5, 10, 14, 26 periods)
- MACD variants
- Z-scores
- Momentum
- Volume features
- Rolling risk metrics (Sharpe, Sortino, drawdown)
- Price position features
- Time-of-day features

#

### 3. FeatureSelection

Two-stage feature selection to reduce dimensionality.

**Stage 1**: Variance threshold (removes near-constant features)
**Stage 2**: Model-based selection using RandomForest cross-validation ROC-AUC

#

### 4. FeatureNormalization

Per-feature scaling with outlier handling.

**Supported Scalers:**
- `minmax` - Scales to [0, 1] (default)
- `standard` - Zero mean, unit variance
- `robust` - Median/IQR based (outlier resistant)

#

### 5. ModelArtifacts / ArtifactManager

Handles saving, loading, and versioning of trained model artifacts.


---

## Data Requirements

### 1. Input Data Format

The pipeline expects intraday OHLCV data with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `open` | Yes | Opening price |
| `high` | Yes | High price |
| `low` | Yes | Low price |
| `close` | Yes | Closing price |
| `volume` | Yes | Volume |
| `caldt` | No | Datetime (for time features) |
| `day` | No | Trading day |
| `vwap` | No | VWAP (enables vwap_deviation feature) |

#

### 2. Data Volume

- **Minimum**: ~1,000 bars (after lookback period)
- **Recommended**: 10,000+ bars for robust training
- **Feature lookback**: 60 bars consumed for rolling calculations

# 

### 3. Resampling

If training on different timeframes than 1-minute data:

```python
config = TrainingConfig(
    ticker='NVDA',
    resample_freq='5min'  # Resample to 5-minute bars
)
```

Common values: `'5min'`, `'15min'`, `'30min'`, `'1H'`

---

## Training Best Practices

### 1. Match Training and Backtest Timeframes

The `forecast_horizon` in training should match `trade_freq` in backtesting:

```python
# Training
config = TrainingConfig(
    ticker='TSLA',
    forecast_horizon=30,  # Predict 30 bars ahead
    resample_freq='5min'  # Using 5-min bars
)

# Backtesting
backtest_config = {
    'trade_freq': 30,  # Trade every 30 bars (matches forecast_horizon)
    ...
}
```

### 2. Temporal Split Ratios

Temporal (non-random) split of the data based on time. Default 80/10/10 split. For shorter datasets, consider 70/15/15.

```python
config = TrainingConfig(
    train_ratio=0.8,   # First 80% for training
    test_ratio=0.1,    # Next 10% for hyperparameter tuning
    valid_ratio=0.1    # Final 10% for unbiased evaluation
)
```

### 3. Class Balance

Check `result.class_distribution` after training. Highly imbalanced classes (>70/30) may need:
- Adjusted `return_threshold`
- Model with `class_weight='balanced'` (RandomForest)
- `scale_pos_weight` adjustment (XGBoost)

### 4. Overfitting Detection

Compare train vs validation metrics:
- Large gap (e.g., train ROC-AUC 0.95 vs valid 0.52) = overfitting
- Target: validation ROC-AUC > 0.52 with reasonable train/valid gap

### 5. Feature Selection Thresholds

Adjust based on dataset size:
- Small datasets: Lower `roc_auc_threshold` (0.51)
- Large datasets: Higher threshold acceptable (0.53-0.55)

---

## Usage

### 1. Basic Training

```python
from ml import TrainingPipeline, TrainingConfig

config = TrainingConfig(
    ticker='NVDA',
    model_type='xgboost',
    forecast_horizon=60,
    resample_freq='5min'
)

pipeline = TrainingPipeline(config)
result = pipeline.run(data_dir='outputs/data')

print(f"Validation ROC-AUC: {result.valid_metrics.roc_auc:.4f}")
print(f"Features selected: {result.feature_count}")
```

#

### 2. Loading Trained Models

```python
from ml import ArtifactManager, ModelArtifacts

# Find and load latest model
manager = ArtifactManager(base_dir='outputs/models')
artifacts = manager.load_latest(ticker='NVDA', model_type='xgboost')

# Use for predictions
features = ...  # Generate features
X_norm = artifacts.normalizer.transform(features)
probabilities = artifacts.model.predict_proba(X_norm[artifacts.feature_names])[:, 1]
```

#

### 3. Manual Artifact Loading

```python
from ml import ModelArtifacts

artifacts = ModelArtifacts.load(
    model_path='outputs/models/NVDA/NVDA_xgboost_20240115_model.pkl',
    scaler_path='outputs/models/NVDA/NVDA_xgboost_20240115_scalers.pkl',
    features_path='outputs/models/NVDA/NVDA_xgboost_20240115_features.pkl'
)
```

#

### 4. Feature Generation Only

```python
from ml import FeatureGeneration

generator = FeatureGeneration()
features_df = generator.generate_features(df_ohlcv)
print(f"Generated {len(generator.feature_names)} features")
```

#

### 5. Feature Generation for Inference

```python
# Ensure same features as training
features_df = generator.generate_features_for_inference(
    df=new_data,
    required_features=artifacts.feature_names
)
```

---

## Configuration Reference

### 1. TrainingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `start_date` | str | '2024-01-01' | Data start date |
| `resample_freq` | str | None | Resample frequency (e.g., '5min') |
| `model_type` | str | 'gradient_boosting' | Model type |
| `forecast_horizon` | int | 30 | Bars ahead to predict |
| `return_threshold` | float | None | Threshold for positive class (auto if None) |
| `train_ratio` | float | 0.6 | Training set fraction |
| `test_ratio` | float | 0.2 | Test set fraction |
| `valid_ratio` | float | 0.2 | Validation set fraction |
| `variance_threshold` | float | 0.01 | Min variance for features |
| `roc_auc_threshold` | float | 0.52 | Min ROC-AUC for feature selection |
| `scaler_type` | str | 'minmax' | Normalization type |
| `clip_outliers` | bool | True | Clip outliers before scaling |
| `output_dir` | str | None | Output directory (auto-generated if None) |

#

### 2. Supported Models

| Model | Config Value | Notes |
|-------|--------------|-------|
| Random Forest | `'random_forest'` | Good baseline, handles imbalance |
| Gradient Boosting | `'gradient_boosting'` | Sklearn implementation |
| XGBoost | `'xgboost'` | Often best performance, requires xgboost package |

---

## Output Artifacts

Training produces these files in `output_dir`:

| File | Description |
|------|-------------|
| `*_model.pkl` | Trained sklearn/xgboost model |
| `*_scalers.pkl` | Fitted FeatureNormalization instance |
| `*_features.pkl` | List of selected feature names |
| `*_selector.pkl` | Fitted FeatureSelection instance |
| `*_config.pkl` | TrainingConfig used |

---

## Dataclass Containers

### 1. TrainingResult

`TrainingResult` is a dataclass container that bundles all training outputs:

```python
@dataclass
class TrainingResult:
    train_metrics: TrainingMetrics
    test_metrics: TrainingMetrics
    valid_metrics: TrainingMetrics
    paths: Dict[str, str]
    feature_count: int
    class_distribution: Dict[str, float]
```

**Benefits:**

1. **Single Return Value**: Pipeline returns one object instead of multiple values
   ```python
   # Without container: messy unpacking
   train_m, test_m, valid_m, paths, n_feat, dist = pipeline.run()
   
   # With container: clean access
   result = pipeline.run()
   print(result.valid_metrics.roc_auc)
   ```

2. **Self-Documenting**: Field names describe what each value represents

3. **IDE Support**: Autocomplete and type hints work properly

4. **Extensibility**: Add new fields without breaking existing code

5. **Serialization**: Easy to save/load results
   ```python
   import pickle
   with open('result.pkl', 'wb') as f:
       pickle.dump(result, f)
   ```

6. **Comparison**: Compare results across experiments
   ```python
   if result_v2.valid_metrics.roc_auc > result_v1.valid_metrics.roc_auc:
       print("v2 is better")
   ```

#

### 2. TrainingMetrics

Similarly, `TrainingMetrics` bundles evaluation metrics:

```python
@dataclass
class TrainingMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: Optional[np.ndarray] = None
```

Provides `to_dict()` for logging and `__str__()` for printing.

---

## Integration with Backtesting

```python
from ml import ArtifactManager
from strats.classification import SupervisedStrategy
from backtesting import BacktestEngine

# Load trained model
manager = ArtifactManager()
paths = manager.find_latest('NVDA', 'xgboost')

# Create strategy
strategy = SupervisedStrategy.from_artifacts(
    model_path=paths['model_path'],
    scaler_path=paths['scaler_path'],
    features_path=paths['features_path'],
    buy_threshold=0.55,
    sell_threshold=0.45
)

# Run backtest with precomputed probabilities (fast)
probabilities = strategy.get_probabilities(df_resampled)

config = {
    'strategy_type': 'supervised',
    'precomputed_probabilities': probabilities,
    'buy_threshold': 0.55,
    'sell_threshold': 0.45,
    ...
}

engine = BacktestEngine(ticker='NVDA')
results = engine.run(df, df_daily, all_days, config)
```
