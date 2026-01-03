# algo-backtests/ml/feature_selection.py
import logging
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)


# This class must be fit ONLY on training data to prevent data leakage.
# The same fitted selector is then used to transform test and validation sets.
class FeatureSelection:
    """
    Two-stage feature selection for ML trading strategies.
    
    Stage 1: Remove low-variance features using VarianceThreshold
    Stage 2: Model-based selection using RandomForest + cross-validation
    
    
    Example:
        selector = FeatureSelection()
        selector.fit(X_train, y_train)
        
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        X_valid_selected = selector.transform(X_valid)
    """
    
    # Columns that should never be treated as features
    EXCLUDE_COLUMNS = {'date', 'day', 'caldt', 'timestamp', 'index'}
    
    # Core OHLCV columns that should always be kept if present
    CORE_COLUMNS = {'open', 'high', 'low', 'close', 'volume'}
    
    def __init__(
        self,
        variance_threshold: float = 0.01,
        roc_auc_threshold: float = 0.52,
        cv_folds: int = 5,
        n_estimators: int = 50,
        max_features_to_evaluate: int = 100,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the feature selector.
        
        Args:
            variance_threshold: Minimum variance for a feature to be kept.
            roc_auc_threshold: Minimum ROC-AUC score for model-based selection.
            cv_folds: Number of cross-validation folds for model-based selection.
            n_estimators: Number of trees in RandomForest for evaluation.
            max_features_to_evaluate: Maximum features to evaluate in stage 2
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 for all cores).
        """
        self.variance_threshold = variance_threshold
        self.roc_auc_threshold = roc_auc_threshold
        self.cv_folds = cv_folds
        self.n_estimators = n_estimators
        self.max_features_to_evaluate = max_features_to_evaluate
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Fitted state
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        self.variance_selector: Optional[VarianceThreshold] = None
        self._is_fitted: bool = False
        
        # Track which features passed each stage
        self._features_after_variance: List[str] = []
    
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> 'FeatureSelection':
        """
        Fit the feature selector on training data.
        
        Args:
            X: Training feature DataFrame
            y: Training target Series (binary classification labels)
            verbose: Whether to print progress information
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        
        # Identify feature columns (exclude metadata)
        feature_cols = [
            c for c in X.columns 
            if c.lower() not in self.EXCLUDE_COLUMNS
        ]
        
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found in DataFrame")
        
        X_features = X[feature_cols].copy()
        
        if verbose:
            logger.info(f"Starting feature selection with {len(feature_cols)} features")
        
        # Stage 1: Variance threshold
        self._fit_variance_stage(X_features, verbose)
        
        # Stage 2: Model-based selection
        if len(self._features_after_variance) > 0:
            self._fit_model_stage(X_features, y, verbose)
        else:
            self.selected_features = []
        
        # Always keep core OHLCV columns if they exist
        self._ensure_core_columns(feature_cols)
        
        self._is_fitted = True
        
        elapsed = time.time() - start_time
        if verbose:
            logger.info(f"Feature selection complete: {len(self.selected_features)} features selected in {elapsed:.1f}s")
        
        return self
    
    
    def _fit_variance_stage(
        self,
        X: pd.DataFrame,
        verbose: bool
    ) -> None:
        """
        Stage 1: Remove low-variance features.
        
        Args:
            X: Feature DataFrame
            verbose: Whether to print progress
        """
        stage_start = time.time()
        feature_cols = list(X.columns)
        
        # Fit variance threshold selector
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        
        try:
            self.variance_selector.fit(X)
            variance_mask = self.variance_selector.get_support()
            self._features_after_variance = [
                f for f, keep in zip(feature_cols, variance_mask) if keep
            ]
        except Exception as e:
            # If variance selection fails, keep all features
            if verbose:
                logger.warning(f"Variance selection failed ({e}), keeping all features")
            self._features_after_variance = feature_cols
        
        elapsed = time.time() - stage_start
        removed = len(feature_cols) - len(self._features_after_variance)
        
        if verbose:
            logger.info(
                f"  Stage 1 (variance threshold={self.variance_threshold}): "
                f"{len(feature_cols)} → {len(self._features_after_variance)} features "
                f"(-{removed} removed) | {elapsed:.1f}s"
            )
    
    
    def _fit_model_stage(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool
    ) -> None:
        """
        Stage 2: Model-based feature selection using cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            verbose: Whether to print progress
        """
        stage_start = time.time()
        features_to_evaluate = self._features_after_variance
        
        # Limit features to evaluate for performance
        if len(features_to_evaluate) > self.max_features_to_evaluate:
            if verbose:
                logger.info(f"  Limiting evaluation to top {self.max_features_to_evaluate} features by variance")
            # Prioritize by variance (higher variance = more informative)
            variances = X[features_to_evaluate].var().sort_values(ascending=False)
            features_to_evaluate = variances.head(self.max_features_to_evaluate).index.tolist()
        
        X_filtered = X[features_to_evaluate]
        self.feature_scores = {}
        
        total_features = len(features_to_evaluate)
        if verbose:
            logger.info(f"  Stage 2 (model-based): Evaluating {total_features} features...")
        
        # Evaluate each feature individually
        for i, feature in enumerate(features_to_evaluate):
            try:
                X_single = X_filtered[[feature]].values
                
                # Skip if feature has no variance in this subset
                if np.std(X_single) == 0:
                    self.feature_scores[feature] = 0.5
                    continue
                
                clf = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=1  # Single job per feature, parallelism at feature level
                )
                
                scores = cross_val_score(
                    clf, X_single, y,
                    cv=self.cv_folds,
                    scoring='roc_auc',
                    n_jobs=1
                )
                self.feature_scores[feature] = scores.mean()
            
            # Default feature score to random (0.5) on any error
            except Exception:
                self.feature_scores[feature] = 0.5
            
            # Progress logging every 20 features or at the end
            if verbose and ((i + 1) % 20 == 0 or (i + 1) == total_features):
                elapsed = time.time() - stage_start
                features_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = total_features - (i + 1)
                eta = remaining / features_per_sec if features_per_sec > 0 else 0
                logger.info(
                    f"    [{i+1}/{total_features}] features evaluated | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"{features_per_sec:.1f} feat/s | "
                    f"ETA: {eta:.1f}s"
                )
        
        # Select features above ROC-AUC threshold
        self.selected_features = [
            f for f, score in self.feature_scores.items()
            if score > self.roc_auc_threshold
        ]
        
        elapsed = time.time() - stage_start
        if verbose:
            above_threshold = len(self.selected_features)
            logger.info(
                f"  Stage 2 complete: {above_threshold} features above "
                f"ROC-AUC threshold ({self.roc_auc_threshold}) | {elapsed:.1f}s"
            )
    
    
    def _ensure_core_columns(self, available_cols: List[str]) -> None:
        """
        Ensure core OHLCV columns are included if available.
        
        Args:
            available_cols: List of available column names
        """
        for col in self.CORE_COLUMNS:
            if col in available_cols and col not in self.selected_features:
                self.selected_features.append(col)
    
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to selected features only.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            DataFrame with only selected features
            
        Raises:
            RuntimeError: If selector has not been fitted
            ValueError: If required features are missing from input
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureSelection must be fitted before transform")
        
        # Keep metadata columns if present
        keep_cols = [c for c in self.EXCLUDE_COLUMNS if c in X.columns]
        
        # Check for missing features
        missing = set(self.selected_features) - set(X.columns)
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required features: {sorted(missing)[:5]}..."
            )
        
        # Build result with metadata + selected features
        result_cols = keep_cols + [f for f in self.selected_features if f in X.columns]
        
        return X[result_cols].copy()
    
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Training feature DataFrame
            y: Training target Series
            verbose: Whether to print progress
            
        Returns:
            DataFrame with selected features
        """
        self.fit(X, y, verbose=verbose)
        return self.transform(X)
    
    
    def get_top_features(self, n: int = 20, verbose: bool = True) -> List[Tuple[str, float]]:
        """
        Get top n features by ROC-AUC score.
        
        Args:
            n: Number of top features to return
            verbose: Whether to print progress
            
        Returns:
            List of (feature_name, score) tuples sorted by score descending
        """
        if not self.feature_scores:
            if verbose:
                logger.info("No feature scores available")
            return []
        
        sorted_features = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Print top n features by ROC-AUC score (bullet points)
        if verbose:
            logger.info(f"Top {n} features by ROC-AUC score:")
            for feature, score in sorted_features[:n]:
                logger.info(f"  - {feature}: {score:.4f}")
        
        return sorted_features[:n]
    
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Get a detailed report of feature scores.
        
        Returns:
            DataFrame with feature names, scores, and selection status
        """
        if not self.feature_scores:
            return pd.DataFrame()
        
        data = []
        for feature, score in self.feature_scores.items():
            data.append({
                'feature': feature,
                'roc_auc_score': score,
                'selected': feature in self.selected_features,
                'above_threshold': score > self.roc_auc_threshold
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('roc_auc_score', ascending=False).reset_index(drop=True)
    
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted selector to file.
        
        Args:
            filepath: Path to save the selector
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted FeatureSelection")
        
        state = {
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'variance_threshold': self.variance_threshold,
            'roc_auc_threshold': self.roc_auc_threshold,
            '_features_after_variance': self._features_after_variance,
            '_is_fitted': self._is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureSelection':
        """
        Load a fitted selector from file.
        
        Args:
            filepath: Path to load the selector from
            
        Returns:
            Loaded FeatureSelection instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        selector = cls(
            variance_threshold=state.get('variance_threshold', 0.01),
            roc_auc_threshold=state.get('roc_auc_threshold', 0.52)
        )
        selector.selected_features = state['selected_features']
        selector.feature_scores = state['feature_scores']
        selector._features_after_variance = state.get('_features_after_variance', [])
        selector._is_fitted = state['_is_fitted']
        
        return selector
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded feature selector.
        
        Returns:
            Dictionary with feature selector metadata
        """
        return {
            'loaded': True,
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'variance_threshold': self.variance_threshold,
            'roc_auc_threshold': self.roc_auc_threshold,
            'cv_folds': self.cv_folds,
            'n_estimators': self.n_estimators,
            'max_features_to_evaluate': self.max_features_to_evaluate,
            'random_state': self.random_state,
            '_is_fitted': self._is_fitted
        }
