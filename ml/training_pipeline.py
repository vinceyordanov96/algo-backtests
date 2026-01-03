# algo-backtests/ml/training_pipeline.py
import os
import logging
import pickle
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any, Union

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score, 
    roc_auc_score,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .feature_generation import FeatureGeneration, FeatureConfig
from .feature_selection import FeatureSelection
from .feature_normalization import FeatureNormalization, create_temporal_split

from .training_config import TrainingConfig
from .training_metrics import TrainingMetrics
from .training_results import TrainingResult

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for supervised ML trading strategies.
    
    This pipeline ensures proper ordering of operations to prevent data leakage.
    
    1. Load data
    2. Generate features
    3. Create labels
    4. Temporal split of data before any fitting.
    5. Feature selection from training data.
    6. Normalization on training data.
    7. Train model to fit training data.
    8. Evaluate model on test data.
    9. Final evaluation on validation data.
    10. Save all artifacts for signal generation.
    
    Example:
        config = TrainingConfig(ticker='NVDA', model_type='xgboost')
        pipeline = TrainingPipeline(config)
        result = pipeline.run()
        
        print(f"Validation ROC-AUC: {result.valid_metrics.roc_auc:.4f}")
    """
    
    SUPPORTED_MODELS = ['random_forest', 'gradient_boosting', 'xgboost']
    
    def __init__(self, config: Union[TrainingConfig, Dict[str, Any]]):
        """
        Initialize the training pipeline.
        
        Args:
            config: TrainingConfig instance or dictionary with config values
        """
        if isinstance(config, dict):
            config = TrainingConfig(**config)
        
        config.validate()
        self.config = config
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize pipeline components
        self.preprocessor = FeatureGeneration(config=FeatureConfig())
        self.feature_selector = FeatureSelection(
            variance_threshold=config.variance_threshold,
            roc_auc_threshold=config.roc_auc_threshold
        )
        self.normalizer = FeatureNormalization(
            scaler_type=config.scaler_type,
            clip_outliers=config.clip_outliers
        )
        
        # Model (created later based on config)
        self.model = None
        
        # Data storage
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.X_valid: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.y_valid: Optional[pd.Series] = None
        
        # Feature names (set after feature selection)
        self.feature_names: List[str] = []
    
    
    def load_data(self, data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load intraday and daily data for the ticker.
        Fetches from DataFetcher if not found locally.
        
        Searches multiple directory structures:
        1. {data_dir}/intraday/{ticker}/ and {data_dir}/daily/{ticker}/
        2. {data_dir}/{ticker}/
        3. data/{ticker}/
        
        Args:
            data_dir: Base directory for data files
            
        Returns:
            Tuple of (intraday_df, daily_df)
        """
        start_time = time.time()
        ticker = self.config.ticker
        
        logger.info(f"Loading data for {ticker}...")
        
        # Try multiple directory structures
        search_patterns = [
            # Pattern 1: outputs/data/intraday/{ticker} + outputs/data/daily/{ticker}
            {
                'intra_dir': os.path.join(data_dir, 'intraday', ticker),
                'daily_dir': os.path.join(data_dir, 'daily', ticker),
            },
            # Pattern 2: {data_dir}/{ticker}/
            {
                'intra_dir': os.path.join(data_dir, ticker),
                'daily_dir': os.path.join(data_dir, ticker),
            },
            # Pattern 3: data/{ticker}/
            {
                'intra_dir': os.path.join('data', ticker),
                'daily_dir': os.path.join('data', ticker),
            },
        ]
        
        df_intraday = None
        df_daily = None
        
        for pattern in search_patterns:
            intra_dir = pattern['intra_dir']
            daily_dir = pattern['daily_dir']
            
            # Find intraday files
            intra_files = []
            if os.path.exists(intra_dir):
                intra_files = [
                    f for f in os.listdir(intra_dir)
                    if ('intra' in f.lower() or 'minute' in f.lower()) and f.endswith('.csv')
                ]
            
            # Find daily files
            daily_files = []
            if os.path.exists(daily_dir):
                daily_files = [
                    f for f in os.listdir(daily_dir)
                    if 'daily' in f.lower() and f.endswith('.csv')
                ]
            
            if intra_files and daily_files:
                # Sort by modification time (most recent first)
                intra_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(intra_dir, x)),
                    reverse=True
                )
                daily_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(daily_dir, x)),
                    reverse=True
                )
                
                # Load data
                df_intraday = pd.read_csv(os.path.join(intra_dir, intra_files[0]), index_col=0)
                df_daily = pd.read_csv(os.path.join(daily_dir, daily_files[0]), index_col=0)
                
                # Reset index if caldt is the index (common in saved data)
                if df_intraday.index.name == 'caldt' or 'caldt' not in df_intraday.columns:
                    df_intraday = df_intraday.reset_index()
                if df_daily.index.name in ('day', 'caldt') or ('day' not in df_daily.columns and 'caldt' not in df_daily.columns):
                    df_daily = df_daily.reset_index()
                
                elapsed = time.time() - start_time
                logger.info(f"  Found data in {intra_dir}")
                logger.info(f"  Loaded {len(df_intraday):,} intraday, {len(df_daily):,} daily records | {elapsed:.1f}s")
                break
        
        if df_intraday is None or df_daily is None:
            # Fetch new data
            logger.info(f"  No local data found. Fetching from API...")
            ticker_dir = os.path.join(data_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            try:
                from connectors.data import DataFetcher
                fetcher = DataFetcher(ticker)
                df_intraday, df_daily = fetcher.process_data()
                
                if df_intraday.empty or df_daily.empty:
                    raise ValueError(f"DataFetcher returned empty data for {ticker}")
                
                # Save fetched data
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                intra_path = os.path.join(ticker_dir, f'{ticker}_intra_feed_{version}.csv')
                daily_path = os.path.join(ticker_dir, f'{ticker}_daily_feed_{version}.csv')
                
                df_intraday.to_csv(intra_path, index=False)
                df_daily.to_csv(daily_path, index=False)
                
                elapsed = time.time() - start_time
                logger.info(f"  Fetched and saved data | {elapsed:.1f}s")
                
            except Exception as e:
                raise FileNotFoundError(
                    f"No data files found for {ticker} in searched directories "
                    f"and failed to fetch data: {e}"
                )
        
        # Clean up index columns
        for df in [df_intraday, df_daily]:
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)
        
        # Filter by start date
        if 'caldt' in df_intraday.columns:
            df_intraday = df_intraday[df_intraday['caldt'] >= self.config.start_date]
        
        if 'day' in df_daily.columns:
            df_daily = df_daily[df_daily['day'] >= self.config.start_date]
        elif 'caldt' in df_daily.columns:
            df_daily = df_daily[df_daily['caldt'] >= self.config.start_date]
        
        # Resample intraday data if needed
        if self.config.resample_freq and 'caldt' in df_intraday.columns:
            logger.info(f"  Resampling to {self.config.resample_freq}...")
            df_intraday = self._resample_intraday(df_intraday)
            logger.info(f"  Resampled to {len(df_intraday):,} bars")
        
        return df_intraday, df_daily
    
    
    def _resample_intraday(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample intraday data to specified frequency.
        
        Args:
            df: Intraday DataFrame with 'caldt' column
            
        Returns:
            Resampled DataFrame
        """
        df = df.copy()
        df['caldt'] = pd.to_datetime(df['caldt'])
        df = df.set_index('caldt')
        
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Add optional columns if present
        if 'vwap' in df.columns:
            agg_rules['vwap'] = 'mean'
        if 'day' in df.columns:
            agg_rules['day'] = 'first'
        
        df_resampled = df.resample(self.config.resample_freq).agg(agg_rules).dropna()
        
        return df_resampled.reset_index()
    
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from raw OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with generated features
        """
        logger.info(f"Generating features for {self.config.ticker}...")
        
        features_df = self.preprocessor.generate_features(df, drop_lookback_rows=True, verbose=True)
        
        return features_df
    
    
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary labels for classification.
        
        Label = 1 if price increases more than threshold over forecast_horizon
        Label = 0 otherwise
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series of binary labels
        """
        logger.info("Creating labels...")
        
        horizon = self.config.forecast_horizon
        
        # Calculate forward returns
        forward_returns = df['close'].pct_change(horizon).shift(-horizon)
        
        # Determine threshold
        if self.config.return_threshold is not None:
            threshold = self.config.return_threshold
        else:
            # Adaptive threshold: use a small positive value
            # This creates roughly balanced classes for typical market data
            threshold = 0.001
        
        labels = (forward_returns > threshold).astype(int)
        
        # Report class distribution
        valid_labels = labels.dropna()
        class_dist = valid_labels.value_counts(normalize=True)
        
        logger.info(f"  Threshold: {threshold:.4f}, Horizon: {horizon} bars")
        logger.info(f"  Class distribution: 0={class_dist.get(0, 0):.2%}, 1={class_dist.get(1, 0):.2%}")
        
        return labels
    
    
    def split_data(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series
    ) -> None:
        """
        Split data into train/test/validation sets.
        
        CRITICAL: This must be done BEFORE any fitting (feature selection,
        normalization) to prevent data leakage.
        
        Args:
            features_df: Feature DataFrame
            labels: Label Series
        """
        logger.info("Splitting data temporally...")
        
        # Remove rows with NaN labels (from forward-looking label creation)
        valid_mask = ~labels.isna()
        features_clean = features_df[valid_mask].reset_index(drop=True)
        labels_clean = labels[valid_mask].reset_index(drop=True)
        
        # Temporal split
        (
            self.X_train, self.X_test, self.X_valid,
            self.y_train, self.y_test, self.y_valid
        ) = create_temporal_split(
            features_clean,
            labels_clean,
            train_ratio=self.config.train_ratio,
            test_ratio=self.config.test_ratio,
            valid_ratio=self.config.valid_ratio
        )
        
        logger.info(
            f"  Train: {len(self.X_train):,} | "
            f"Test: {len(self.X_test):,} | "
            f"Valid: {len(self.X_valid):,}"
        )
    
    
    def fit_feature_selection(self, verbose: bool = True) -> None:
        """
        Fit feature selection on TRAINING data only.
        
        Args:
            verbose: Whether to print progress
        """
        if self.X_train is None:
            raise RuntimeError("Must call split_data() before fit_feature_selection()")
        
        logger.info("Fitting feature selection on training data...")
        
        # Fit on training data only
        self.feature_selector.fit(self.X_train, self.y_train, verbose=verbose)
        
        # Transform all sets
        self.X_train = self.feature_selector.transform(self.X_train)
        self.X_test = self.feature_selector.transform(self.X_test)
        self.X_valid = self.feature_selector.transform(self.X_valid)
        
        self.feature_names = self.feature_selector.selected_features
    
    
    def fit_normalization(self, verbose: bool = True) -> None:
        """
        Fit normalization on TRAINING data only.
        
        Args:
            verbose: Whether to print progress
        """
        if self.X_train is None:
            raise RuntimeError("Must call split_data() before fit_normalization()")
        
        logger.info("Fitting normalization on training data...")
        
        # Fit on training data only
        self.normalizer.fit(self.X_train, verbose=verbose)
        
        # Transform all sets
        self.X_train = self.normalizer.transform(self.X_train)
        self.X_test = self.normalizer.transform(self.X_test)
        self.X_valid = self.normalizer.transform(self.X_valid)
    
    
    def create_model(self) -> None:
        """
        Create the ML model based on configuration.
        """
        model_type = self.config.model_type
        
        logger.info(f"Creating {model_type} model...")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ImportError("xgboost not installed. Use 'pip install xgboost'")
            
            # Calculate scale_pos_weight for imbalanced classes
            pos_count = self.y_train.sum()
            neg_count = len(self.y_train) - pos_count
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    
    def _get_feature_columns(self) -> List[str]:
        """
        Get feature column names (excluding metadata columns).
        
        Returns:
            List of feature column names
        """
        exclude = {'date', 'day', 'caldt', 'timestamp', 'index'}
        return [c for c in self.X_train.columns if c.lower() not in exclude]
    
    
    def train(self) -> TrainingMetrics:
        """
        Train the model on training data and evaluate on test data.
        
        Returns:
            TrainingMetrics for test set evaluation
        """
        if self.model is None:
            raise RuntimeError("Must call create_model() before train()")
        
        feature_cols = self._get_feature_columns()
        
        X_train = self.X_train[feature_cols].values
        X_test = self.X_test[feature_cols].values
        
        logger.info(f"Training {self.config.model_type} model...")
        logger.info(f"  Training samples: {len(X_train):,}, Features: {len(feature_cols)}")
        
        start_time = time.time()
        self.model.fit(X_train, self.y_train)
        train_time = time.time() - start_time
        
        logger.info(f"  Training complete | {train_time:.1f}s")
        
        # Evaluate on test set
        metrics = self._evaluate(X_test, self.y_test, "Test")
        
        return metrics
    
    
    def validate(self) -> TrainingMetrics:
        """
        Evaluate on held-out validation set.
        
        Returns:
            TrainingMetrics for validation set
        """
        feature_cols = self._get_feature_columns()
        X_valid = self.X_valid[feature_cols].values
        
        metrics = self._evaluate(X_valid, self.y_valid, "Validation")
        
        return metrics
    
    
    def _evaluate(
        self,
        X: np.ndarray,
        y: pd.Series,
        set_name: str
    ) -> TrainingMetrics:
        """
        Evaluate model on a dataset.
        
        Args:
            X: Feature array
            y: Label Series
            set_name: Name for printing
            
        Returns:
            TrainingMetrics instance
        """
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = TrainingMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1=f1_score(y, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y, y_proba),
            confusion_matrix=confusion_matrix(y, y_pred)
        )
        
        logger.info(f"  {set_name} metrics: Acc={metrics.accuracy:.4f}, ROC-AUC={metrics.roc_auc:.4f}, "
                   f"Prec={metrics.precision:.4f}, Rec={metrics.recall:.4f}")
        
        return metrics
    
    
    def save_artifacts(self, suffix: str = '') -> Dict[str, str]:
        """
        Save all artifacts needed for inference.
        
        Saves:
        - Trained model
        - Fitted normalizer (with scalers)
        - Selected feature names
        - Feature selector (optional, for analysis)
        - Training configuration
        
        Args:
            suffix: Optional suffix for filenames
            
        Returns:
            Dictionary of artifact paths
        """
        logger.info("Saving model artifacts...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config.ticker}_{self.config.model_type}_{timestamp}{suffix}"
        
        paths = {}
        
        # Save model
        model_path = os.path.join(self.config.output_dir, f"{base_name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        paths['model_path'] = model_path
        logger.info(f"  Model: {model_path}")
        
        # Save normalizer
        scaler_path = os.path.join(self.config.output_dir, f"{base_name}_scalers.pkl")
        self.normalizer.save(scaler_path)
        paths['scaler_path'] = scaler_path
        logger.info(f"  Scalers: {scaler_path}")
        
        # Save feature names
        features_path = os.path.join(self.config.output_dir, f"{base_name}_features.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        paths['features_path'] = features_path
        logger.info(f"  Features: {features_path}")
        
        # Save feature selector (for analysis)
        selector_path = os.path.join(self.config.output_dir, f"{base_name}_selector.pkl")
        self.feature_selector.save(selector_path)
        paths['selector_path'] = selector_path
        logger.info(f"  Selector: {selector_path}")
        
        # Save config
        config_path = os.path.join(self.config.output_dir, f"{base_name}_config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        paths['config_path'] = config_path
        logger.info(f"  Config: {config_path}")
        
        return paths
    
    
    def run(self, data_dir: str = 'data') -> TrainingResult:
        """
        Run the complete training pipeline.
        
        This is the main entry point that executes all steps in the correct order.
        
        Args:
            data_dir: Base directory for data files
            
        Returns:
            TrainingResult with metrics and artifact paths
        """
        pipeline_start = time.time()
        
        logger.info("=" * 60)
        logger.info(f"TRAINING PIPELINE: {self.config.ticker}")
        logger.info(f"Model: {self.config.model_type}")
        logger.info("=" * 60)
        
        # Step 1: Load data
        df_intraday, df_daily = self.load_data(data_dir)
        
        # Step 2: Generate features
        features_df = self.generate_features(df_intraday)
        
        # Step 3: Create labels
        labels = self.create_labels(features_df)
        
        # Step 4: Split data (BEFORE any fitting!)
        self.split_data(features_df, labels)
        
        # Step 5: Fit feature selection on training data only
        self.fit_feature_selection()
        
        # Step 6: Fit normalization on training data only
        self.fit_normalization()
        
        # Step 7: Create and train model
        self.create_model()
        test_metrics = self.train()
        
        # Step 8: Evaluate on training data (for comparison)
        feature_cols = self._get_feature_columns()
        X_train = self.X_train[feature_cols].values
        train_metrics = self._evaluate(X_train, self.y_train, "Training")
        
        # Step 9: Final validation
        valid_metrics = self.validate()
        
        # Step 10: Save artifacts
        paths = self.save_artifacts()
        
        # Compute class distribution
        valid_labels = labels.dropna()
        class_dist = valid_labels.value_counts(normalize=True).to_dict()
        
        pipeline_elapsed = time.time() - pipeline_start
        
        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETE | Total time: {pipeline_elapsed:.1f}s")
        logger.info("=" * 60)
        
        return TrainingResult(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            valid_metrics=valid_metrics,
            paths=paths,
            feature_count=len(self.feature_names),
            class_distribution=class_dist
        )


def main():
    """Example usage of the training pipeline."""
    
    # Create configuration
    config = TrainingConfig(
        ticker='NVDA',
        model_type='gradient_boosting',
        forecast_horizon=30,
        train_ratio=0.6,
        test_ratio=0.2,
        valid_ratio=0.2
    )
    
    # Run pipeline
    pipeline = TrainingPipeline(config)
    result = pipeline.run()
    
    # Print results
    print(f"\nFinal Results:")
    print(f"  Features selected: {result.feature_count}")
    print(f"  Validation ROC-AUC: {result.valid_metrics.roc_auc:.4f}")
    print(f"  Class distribution: {result.class_distribution}")


if __name__ == '__main__':
    main()
