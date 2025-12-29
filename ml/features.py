import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import List, Tuple, Dict, Any
from scipy.stats import iqr


class FeatureSelection:
    """
    Two-stage feature selection:
    1. Remove low-variance features (VarianceThreshold)
    2. Model-based selection using RandomForest + cross-validation
    """
    
    def __init__(
        self, 
        variance_threshold: float = 0.16,  # 0.8 * (1 - 0.8)
        roc_auc_threshold: float = 0.5,    # Better than random
        cv_folds: int = 5,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.variance_threshold = variance_threshold
        self.roc_auc_threshold = roc_auc_threshold
        self.cv_folds = cv_folds
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        self.variance_selector = None
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series = None,
        target_column: str = 'close',
        return_threshold: float = None
    ) -> 'FeatureSelection':
        """
        Fit the feature selector on training data.
        
        Args:
            X: Feature DataFrame
            y: Binary target (optional - will be computed if not provided)
            target_column: Column to use for computing target
            return_threshold: Threshold for binary target (IQR-based if None)
        """
        # Exclude non-feature columns
        exclude_cols = ['date', 'day', 'caldt']
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        X_features = X[feature_cols].copy()
        
        # Stage 1: Variance threshold
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        self.variance_selector.fit(X_features)
        
        variance_mask = self.variance_selector.get_support()
        features_after_variance = [f for f, keep in zip(feature_cols, variance_mask) if keep]
        
        print(f"Stage 1: {len(feature_cols)} → {len(features_after_variance)} features (variance filter)")
        
        # Compute target if not provided
        if y is None and target_column in X.columns:
            returns = X[target_column].pct_change().fillna(0)
            if return_threshold is None:
                return_threshold = iqr(returns.dropna()) * 1.5
            y = (returns > return_threshold).astype(int)
        
        if y is None:
            print("Warning: No target provided, skipping model-based selection")
            self.selected_features = features_after_variance
            return self
        
        # Stage 2: Model-based selection
        X_filtered = X_features[features_after_variance]
        
        self.feature_scores = {}
        for feature in features_after_variance:
            try:
                X_single = X_filtered[[feature]].values
                clf = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                scores = cross_val_score(clf, X_single, y, cv=self.cv_folds, scoring='roc_auc')
                self.feature_scores[feature] = scores.mean()
            except Exception:
                self.feature_scores[feature] = 0.5  # Default to random
        
        # Keep features above threshold
        self.selected_features = [
            f for f, score in self.feature_scores.items() 
            if score > self.roc_auc_threshold
        ]
        
        # Always keep OHLCV
        must_keep = ['open', 'high', 'low', 'close', 'volume']
        for col in must_keep:
            if col in feature_cols and col not in self.selected_features:
                self.selected_features.append(col)
        
        print(f"Stage 2: {len(features_after_variance)} → {len(self.selected_features)} features (model-based)")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to selected features only.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            DataFrame with selected features
        """
        # Keep date/day columns if present
        keep_cols = ['date', 'day', 'caldt']
        result_cols = [c for c in keep_cols if c in X.columns]
        result_cols.extend([f for f in self.selected_features if f in X.columns])
        
        return X[result_cols].copy()
    
    
    def fit_transform(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: DataFrame to fit and transform
            **kwargs: Additional arguments for fit method
            
        Returns:
            DataFrame with selected features
        """
        self.fit(X, **kwargs)
        return self.transform(X)
    
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top n features by score.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of top features by score
        """
        sorted_features = sorted(
            self.feature_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_features[:n]
