import os
import pickle
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data import DataFetcher
from ml.preprocessing import PreProcessing
from ml.features import FeatureSelection
from ml.normalization import Normalization


class TrainingPipeline:
    """
    Complete training pipeline for ML trading strategies.
    """
    
    def __init__(
        self,
        ticker: str,
        model_type: str = 'gradient_boosting',
        lookback_window: int = 30,
        forecast_horizon: int = 30,
        return_threshold: float = None,  # None = use IQR-based
        output_dir: str = 'ml/models'
    ):
        self.ticker = ticker
        self.model_type = model_type
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.return_threshold = return_threshold
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Pipeline components
        self.preprocessor = PreProcessing(min_lookback=200)
        self.feature_selector = FeatureSelection()
        self.normalizer = Normalization(scaler_type='minmax')
        
        # Model
        self.model = None
        
        # Data
        self.X_train = None
        self.X_test = None
        self.X_valid = None
        self.y_train = None
        self.y_test = None
        self.y_valid = None
        
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data using existing DataFetcher.
        
        Returns:
            - Tuple containing intraday and daily data
                - df_intraday: DataFrame containing intraday data
                - df_daily: DataFrame containing daily data
        """
        if os.path.exists(f'data/{self.ticker}'):
            # Don't use index_col - keep all columns as regular columns
            df_intraday = pd.read_csv(
                f'data/{self.ticker}/{self.ticker}_intra_feed_20251222135246.csv'
            )
            df_daily = pd.read_csv(
                f'data/{self.ticker}/{self.ticker}_daily_feed_20251222135246.csv'
            )
            
            # Drop the unnamed index column if present
            if 'Unnamed: 0' in df_intraday.columns:
                df_intraday = df_intraday.drop(columns=['Unnamed: 0'])
            if 'Unnamed: 0' in df_daily.columns:
                df_daily = df_daily.drop(columns=['Unnamed: 0'])
            
            # Filter intraday by caldt
            df_intraday = df_intraday[df_intraday['caldt'] >= '2024-01-01']
            
            # Filter daily by 'day' column (check your actual column name)
            if 'day' in df_daily.columns:
                df_daily = df_daily[df_daily['day'] >= '2024-01-01']
            elif 'caldt' in df_daily.columns:
                df_daily = df_daily[df_daily['caldt'] >= '2024-01-01']
            
            # Resample to 15-minute bars
            df_intraday['caldt'] = pd.to_datetime(df_intraday['caldt'])
            df_intraday = df_intraday.set_index('caldt')
            
            df_resampled = df_intraday.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'vwap': 'mean',
                'day': 'first'
            }).dropna()
            
            return df_resampled.reset_index(), df_daily
        
        fetcher = DataFetcher(self.ticker)
        df_intraday, df_daily = fetcher.process_data()
        
        return df_intraday, df_daily
    

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate and select features.
        
        Args:
            df: DataFrame to generate features from
            
        Returns:
            DataFrame with generated features
        """
        print(f"Generating features for {self.ticker}...")
        features_df = self.preprocessor.generate_features(df)
        print(f"Generated {len(self.preprocessor.feature_names)} features")
        
        print("Selecting features...")
        features_df = self.feature_selector.fit_transform(features_df)
        print(f"Selected {len(self.feature_selector.selected_features)} features")
        
        return features_df
    

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary labels for classification.
        1 = price goes up more than threshold
        0 = otherwise
        
        Args:
            df: DataFrame to create labels from
            
        Returns:
            Series of labels
        """
        returns = df['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
        
        if self.return_threshold is None:
            #from scipy.stats import iqr
            #threshold = iqr(returns.dropna()) * 1.5
            threshold = 0.001
        else:
            threshold = self.return_threshold
        
        labels = (returns > threshold).astype(int)
        try:
            print(f"Class distribution: {labels.value_counts(normalize=True).to_dict()}")
        except:
            pass
        print(f"Threshold used: {threshold:.4f}")
        
        return labels
    
    
    def split_data(self, df: pd.DataFrame, labels: pd.Series) -> None:
        """
        Split into train/test/valid sets.
        
        Args:
            df: DataFrame to split
            labels: Series of labels
        """
        # Remove last forecast_horizon rows (no labels)
        valid_idx = ~labels.isna()
        df = df[valid_idx].reset_index(drop=True)
        labels = labels[valid_idx].reset_index(drop=True)
        
        # Temporal split
        n = len(df)
        train_end = int(n * 0.335)
        test_end = int(n * 0.67)
        
        self.X_train = df.iloc[:train_end]
        self.X_test = df.iloc[train_end:test_end]
        self.X_valid = df.iloc[test_end:]
        
        self.y_train = labels.iloc[:train_end]
        self.y_test = labels.iloc[train_end:test_end]
        self.y_valid = labels.iloc[test_end:]
        
        print(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}, Valid: {len(self.X_valid)}")
    

    def normalize_data(self):
        """
        Fit normalizer on train+test, transform all sets.
        """
        # Fit on train + test
        train_test = pd.concat([self.X_train, self.X_test], axis=0)
        self.normalizer.fit(train_test)
        
        self.X_train = self.normalizer.transform(self.X_train)
        self.X_test = self.normalizer.transform(self.X_test)
        self.X_valid = self.normalizer.transform(self.X_valid)
    
    
    def get_feature_columns(self) -> List[str]:
        """
        Get feature column names (excluding date/metadata).
        
        Returns:
            List of feature column names
        """
        exclude = ['date', 'day', 'caldt']
        return [c for c in self.X_train.columns if c not in exclude]
    
    
    def create_model(self):
        """
        Create the ML model.
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=1,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    
    def train(self) -> Dict[str, float]:
        """
        Train the model and return metrics.
        
        Returns:
            - Dictionary containing training metrics
                - accuracy: Accuracy score
                - precision: Precision score
                - recall: Recall score
                - f1: F1 score
                - roc_auc: ROC-AUC score
        """
        feature_cols = self.get_feature_columns()
        
        X_train = self.X_train[feature_cols].values
        X_test = self.X_test[feature_cols].values
        
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, self.y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_proba)
        }
        
        print("Test Set Metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        
        return metrics
    
    
    def validate(self) -> Dict[str, float]:
        """Evaluate on validation set.
        
        Returns:
            - Dictionary containing validation metrics
                - accuracy: Accuracy score
                - precision: Precision score
                - recall: Recall score
                - f1: F1 score
                - roc_auc: ROC-AUC score
        """
        feature_cols = self.get_feature_columns()
        X_valid = self.X_valid[feature_cols].values
        
        y_pred = self.model.predict(X_valid)
        y_proba = self.model.predict_proba(X_valid)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_valid, y_pred),
            'precision': precision_score(self.y_valid, y_pred, zero_division=0),
            'recall': recall_score(self.y_valid, y_pred, zero_division=0),
            'f1': f1_score(self.y_valid, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_valid, y_proba)
        }
        
        print("Validation Set Metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        
        return metrics
    
    
    def save(self, suffix: str = '') -> Dict[str, str]:
        """
        Save model and artifacts.
        
        Args:
            suffix: Suffix for the saved files
            
        Returns:
            - Dictionary containing paths to saved model, scalers, and features
                - model_path: Path to saved model
                - scaler_path: Path to saved scalers
                - features_path: Path to saved features
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.ticker}_{self.model_type}_{timestamp}{suffix}"
        
        # Save model
        model_path = os.path.join(self.output_dir, f"{base_name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save normalizer
        scaler_path = os.path.join(self.output_dir, f"{base_name}_scalers.pkl")
        self.normalizer.save(scaler_path)
        
        # Save feature names
        features_path = os.path.join(self.output_dir, f"{base_name}_features.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_selector.selected_features, f)
        
        print(f"Saved model to: {model_path}")
        print(f"Saved scalers to: {scaler_path}")
        print(f"Saved features to: {features_path}")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'features_path': features_path
        }
    
    
    def run_pipeline(self) -> Dict[str, Dict[str, float] | Dict[str, str] | int]:
        """
        Run the complete training pipeline.
        
        Returns:
            - Dictionary containing training metrics, validation metrics, 
            paths to saved artifacts, and number of features
                - train_metrics: Dictionary containing training metrics
                - valid_metrics: Dictionary containing validation metrics
                - paths: Dictionary containing paths to saved model, scalers, and features
                - feature_count: Number of features
        """
        # Load data
        df_intraday, df_daily = self.load_data()
        
        # Generate features
        features_df = self.prepare_features(df_intraday)
        
        # Create labels
        labels = self.create_labels(features_df)
        
        # Split data
        self.split_data(features_df, labels)
        
        # Normalize
        self.normalize_data()
        
        # Create and train model
        self.create_model()
        train_metrics = self.train()
        
        # Validate
        valid_metrics = self.validate()
        
        # Save artifacts
        paths = self.save()
        
        return {
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'paths': paths,
            'feature_count': len(self.feature_selector.selected_features)
        }


# Example usage
if __name__ == '__main__':
    pipeline = TrainingPipeline(
        ticker='NVDA',
        model_type='gradient_boosting',
        lookback_window=30
    )
    
    results = pipeline.run_pipeline()
    print("\nPipeline complete!")
    print(f"Validation ROC-AUC: {results['valid_metrics']['roc_auc']:.4f}")
