# algo-backtests/ml/feature_normalization.py
import logging
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class FeatureNormalization:
    """
    Per-feature normalization for ML trading strategies.
    
    Example:
        normalizer = FeatureNormalization(scaler_type='minmax')
        normalizer.fit(X_train)
        
        X_train_norm = normalizer.transform(X_train)
        X_test_norm = normalizer.transform(X_test)
        X_valid_norm = normalizer.transform(X_valid)
    
    Supported scaler types:
        - 'minmax': Scales to [0, 1] range. Good for bounded features.
        - 'standard': Zero mean, unit variance. Good for normally distributed.
        - 'robust': Uses median/IQR. Good for data with outliers.
    """
    
    SCALER_TYPES = {
        'minmax': MinMaxScaler,
        'standard': StandardScaler,
        'robust': RobustScaler
    }
    
    # Columns that should never be normalized
    EXCLUDE_COLUMNS = {'date', 'day', 'caldt', 'timestamp', 'index'}
    
    def __init__(
        self,
        scaler_type: str = 'minmax',
        clip_outliers: bool = True,
        outlier_std: float = 5.0
    ):
        """
        Initialize the feature normalizer class.
        
        Args:
            scaler_type: Type of scaler to use ('minmax', 'standard', 'robust')
            clip_outliers: Whether to clip outliers before normalization
            outlier_std: Number of standard deviations for outlier clipping
            
        Raises:
            ValueError: If scaler_type is not supported
        """
        if scaler_type not in self.SCALER_TYPES:
            raise ValueError(
                f"Unknown scaler type: {scaler_type}. "
                f"Supported types: {list(self.SCALER_TYPES.keys())}"
            )
        
        self.scaler_type = scaler_type
        self.scaler_class = self.SCALER_TYPES[scaler_type]
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        
        # Fitted state
        self.scalers: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self._is_fitted: bool = False
        
        # Statistics for outlier clipping (computed during fit)
        self._clip_bounds: Dict[str, Tuple[float, float]] = {}
    
    
    def fit(self, X: pd.DataFrame, verbose: bool = True) -> 'FeatureNormalization':
        """
        Fit scalers on training data (only) to prevent data leakage.
        
        Args:
            X: Training DataFrame
            verbose: Whether to log progress
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        
        # Identify feature columns (exclude metadata)
        self.feature_columns = [
            c for c in X.columns
            if c.lower() not in self.EXCLUDE_COLUMNS
        ]
        
        if len(self.feature_columns) == 0:
            raise ValueError("No feature columns found to normalize")
        
        if verbose:
            logger.info(f"Fitting {self.scaler_type} normalization on {len(self.feature_columns)} features...")
        
        # Fit a scaler for each feature column
        for i, col in enumerate(self.feature_columns):
            values = X[col].values.reshape(-1, 1)
            
            # Compute clipping bounds if enabled
            if self.clip_outliers:
                mean = np.nanmean(values)
                std = np.nanstd(values)
                lower = mean - self.outlier_std * std
                upper = mean + self.outlier_std * std
                self._clip_bounds[col] = (lower, upper)
                
                # Clip before fitting scaler
                values = np.clip(values, lower, upper)
            
            # Fit scaler
            scaler = self.scaler_class()
            scaler.fit(values)
            self.scalers[col] = scaler
            
            # Progress logging every 50 features
            if verbose and ((i + 1) % 50 == 0 or (i + 1) == len(self.feature_columns)):
                elapsed = time.time() - start_time
                features_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"  [{i+1}/{len(self.feature_columns)}] scalers fitted | "
                           f"{elapsed:.1f}s | {features_per_sec:.1f} feat/s")
        
        self._is_fitted = True
        
        elapsed = time.time() - start_time
        if verbose:
            logger.info(f"Normalization fitting complete: {len(self.scalers)} scalers in {elapsed:.1f}s")
        
        return self
    
    
    def transform(self, X: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Transform feature data using fitted scalers.
        
        Args:
            X: DataFrame to transform
            verbose: Whether to log progress
            
        Returns:
            Normalized DataFrame
            
        Raises:
            RuntimeError: If feature normalizer has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Normalization must be fitted before transform")
        
        start_time = time.time()
        result = X.copy()
        
        for col in self.feature_columns:
            if col not in X.columns:
                continue
            
            if col not in self.scalers:
                continue
            
            values = X[col].values.reshape(-1, 1)
            
            # Apply clipping if enabled
            if self.clip_outliers and col in self._clip_bounds:
                lower, upper = self._clip_bounds[col]
                values = np.clip(values, lower, upper)
            
            # Transform
            result[col] = self.scalers[col].transform(values).flatten()
        
        if verbose:
            elapsed = time.time() - start_time
            logger.info(f"Transformation complete: {len(result):,} rows in {elapsed:.1f}s")
        
        return result
    
    
    def fit_transform(self, X: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Training DataFrame
            verbose: Whether to log progress
            
        Returns:
            Normalized DataFrame
        """
        return self.fit(X, verbose=verbose).transform(X, verbose=verbose)
    
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale.
        
        Note: If outlier clipping was applied, the inverse transform
        will not recover values that were clipped.
        
        Args:
            X: Normalized DataFrame
            
        Returns:
            DataFrame in original scale
        """
        if not self._is_fitted:
            raise RuntimeError("Normalization must be fitted before inverse_transform")
        
        result = X.copy()
        
        for col in self.feature_columns:
            if col not in X.columns:
                continue
            
            if col not in self.scalers:
                continue
            
            values = X[col].values.reshape(-1, 1)
            result[col] = self.scalers[col].inverse_transform(values).flatten()
        
        return result
    
    
    def get_scaler_params(self, column: str) -> Optional[Dict[str, Any]]:
        """
        Get the fitted parameters for a specific column's scaler.
        
        Args:
            column: Column name
            
        Returns:
            Dictionary of scaler parameters, or None if column not found
        """
        if column not in self.scalers:
            return None
        
        scaler = self.scalers[column]
        
        if self.scaler_type == 'minmax':
            return {
                'min': scaler.data_min_[0],
                'max': scaler.data_max_[0],
                'scale': scaler.scale_[0]
            }
        elif self.scaler_type == 'standard':
            return {
                'mean': scaler.mean_[0],
                'std': scaler.scale_[0]
            }
        elif self.scaler_type == 'robust':
            return {
                'center': scaler.center_[0],
                'scale': scaler.scale_[0]
            }
        
        return None
    
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted normalizer to file.
        
        Args:
            filepath: Path to save the normalizer
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted Normalization")
        
        state = {
            'scaler_type': self.scaler_type,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'clip_outliers': self.clip_outliers,
            'outlier_std': self.outlier_std,
            '_clip_bounds': self._clip_bounds,
            '_is_fitted': self._is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureNormalization':
        """
        Load a fitted normalizer from file.
        
        Args:
            filepath: Path to load the feature normalizer from
            
        Returns:
            Loaded Normalization instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        normalizer = cls(
            scaler_type=state['scaler_type'],
            clip_outliers=state.get('clip_outliers', True),
            outlier_std=state.get('outlier_std', 5.0)
        )
        normalizer.scalers = state['scalers']
        normalizer.feature_columns = state['feature_columns']
        normalizer._clip_bounds = state.get('_clip_bounds', {})
        normalizer._is_fitted = state['_is_fitted']
        
        return normalizer
    
    
    def get_scalers_dict(self) -> Dict[str, Any]:
        """
        Get the raw scalers dictionary for external use.
        
        This is provided for compatibility with existing code that
        expects direct access to scalers.
        
        Returns:
            Dictionary mapping column names to fitted scalers
        """
        return self.scalers.copy()


def create_temporal_split(
    data: pd.DataFrame,
    labels: pd.Series,
    train_ratio: float = 0.6,
    test_ratio: float = 0.2,
    valid_ratio: float = 0.2
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series
]:
    """
    Split data temporally (no shuffle) into train/test/validation.
    
    For time series data, temporal splitting is essential to prevent
    look-ahead bias. This function ensures:
    1. Training data comes first chronologically
    2. Test data comes after training
    3. Validation data comes last
    
    Args:
        data: Feature DataFrame
        labels: Target Series (must have same index as data)
        train_ratio: Fraction for training (default 0.6)
        test_ratio: Fraction for testing (default 0.2)
        valid_ratio: Fraction for validation (default 0.2)
        
    Returns:
        Tuple of (X_train, X_test, X_valid, y_train, y_test, y_valid)
        
    Raises:
        ValueError: If ratios don't sum to 1.0 or data/labels length mismatch
    """
    # Validate ratios
    total_ratio = train_ratio + test_ratio + valid_ratio
    if not np.isclose(total_ratio, 1.0, atol=0.01):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Validate data/labels alignment
    if len(data) != len(labels):
        raise ValueError(
            f"Data length ({len(data)}) != labels length ({len(labels)})"
        )
    
    n = len(data)
    train_end = int(n * train_ratio)
    test_end = int(n * (train_ratio + test_ratio))
    
    # Split features
    X_train = data.iloc[:train_end].reset_index(drop=True)
    X_test = data.iloc[train_end:test_end].reset_index(drop=True)
    X_valid = data.iloc[test_end:].reset_index(drop=True)
    
    # Split labels
    y_train = labels.iloc[:train_end].reset_index(drop=True)
    y_test = labels.iloc[train_end:test_end].reset_index(drop=True)
    y_valid = labels.iloc[test_end:].reset_index(drop=True)
    
    return X_train, X_test, X_valid, y_train, y_test, y_valid
