"""
Feature normalization for ML strategies.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import List, Dict, Tuple, Optional
import pickle


class Normalization:
    """
    Per-feature normalization to avoid data leakage.
    Fits scalers on training data and applies to all sets.
    """
    
    SCALER_TYPES = {
        'minmax': MinMaxScaler,
        'standard': StandardScaler,
        'robust': RobustScaler
    }
    
    def __init__(self, scaler_type: str = 'minmax'):
        if scaler_type not in self.SCALER_TYPES:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.scaler_type = scaler_type
        self.scaler_class = self.SCALER_TYPES[scaler_type]
        self.scalers: Dict[str, object] = {}
        self.feature_columns: List[str] = []
        self.exclude_columns = ['date', 'day', 'caldt']
    
    def fit(self, X: pd.DataFrame) -> 'Normalization':
        """
        Fit scalers on training data.
        
        Args:
            X: Training DataFrame
        """
        self.feature_columns = [c for c in X.columns if c not in self.exclude_columns]
        
        for col in self.feature_columns:
            scaler = self.scaler_class()
            values = X[col].values.reshape(-1, 1)
            scaler.fit(values)
            self.scalers[col] = scaler
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scalers.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            Normalized DataFrame
        """
        result = X.copy()
        
        for col in self.feature_columns:
            if col in X.columns and col in self.scalers:
                values = X[col].values.reshape(-1, 1)
                result[col] = self.scalers[col].transform(values).flatten()
        
        return result
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform normalized data."""
        result = X.copy()
        
        for col in self.feature_columns:
            if col in X.columns and col in self.scalers:
                values = X[col].values.reshape(-1, 1)
                result[col] = self.scalers[col].inverse_transform(values).flatten()
        
        return result
    
    def save(self, filepath: str):
        """Save scalers to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler_type': self.scaler_type,
                'scalers': self.scalers,
                'feature_columns': self.feature_columns
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'Normalization':
        """Load scalers from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        normalizer = cls(scaler_type=data['scaler_type'])
        normalizer.scalers = data['scalers']
        normalizer.feature_columns = data['feature_columns']
        
        return normalizer


def create_train_test_valid_split(
    data: pd.DataFrame,
    train_ratio: float = 0.335,
    test_ratio: float = 0.335,
    valid_ratio: float = 0.33
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (no shuffle) into train/test/validation.
    
    Args:
        data: Full DataFrame
        train_ratio: Fraction for training
        test_ratio: Fraction for testing
        valid_ratio: Fraction for validation
        
    Returns:
        (X_train, X_test, X_valid)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    test_end = int(n * (train_ratio + test_ratio))
    
    X_train = data.iloc[:train_end].reset_index(drop=True)
    X_test = data.iloc[train_end:test_end].reset_index(drop=True)
    X_valid = data.iloc[test_end:].reset_index(drop=True)
    
    return X_train, X_test, X_valid
