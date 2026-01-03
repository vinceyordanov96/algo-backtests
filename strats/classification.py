# algo-backtests/ml/classification.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union

from ml.feature_generation import FeatureGeneration
from ml.model_artifacts import ModelArtifacts

# Import base strategy - handle case where it might not exist yet
try:
    from strats.base import BaseStrategy
    HAS_BASE_STRATEGY = True
except ImportError:
    HAS_BASE_STRATEGY = False
    BaseStrategy = object  # Fallback to object if base doesn't exist


class SupervisedStrategy(BaseStrategy if HAS_BASE_STRATEGY else object):
    """
    ML-based trading strategy that integrates with BackTest class.
    
    This class wraps a trained ML model and generates signals compatible
    with the existing backtesting infrastructure. It is model-agnostic
    and works with any scikit-learn compatible classifier.
    
    Key design principles:
        1. Feature generation uses the SAME code path as training (FeatureGeneration)
        2. Normalization uses the SAME fitted scalers from training
        3. Signals are generated point-in-time (no look-ahead)
        4. Model type doesn't matter - XGBoost, RF, GB all work identically
    
    Example usage in BackTest:
        strategy = SupervisedStrategy.from_artifacts(
            model_path='ml/models/NVDA_xgboost_model.pkl',
            scaler_path='ml/models/NVDA_xgboost_scalers.pkl',
            features_path='ml/models/NVDA_xgboost_features.pkl'
        )
        
        signals = strategy.generate_signals(features_df)
    """
    
    # Signal constants (matching BaseStrategy if available)
    SIGNAL_LONG = 1
    SIGNAL_EXIT = -1
    SIGNAL_HOLD = 0
    
    def __init__(
        self,
        artifacts: Optional[ModelArtifacts] = None,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        use_probability_thresholds: bool = True
    ):
        """
        Initialize the supervised strategy.
        
        Args:
            artifacts: Loaded model artifacts (model, normalizer, features)
            buy_threshold: Probability threshold for buy signal (default 0.55)
            sell_threshold: Probability threshold for sell/exit signal (default 0.45)
            use_probability_thresholds: If True, use probability thresholds.
                                       If False, use raw model predictions.
        """
        if HAS_BASE_STRATEGY:
            super().__init__()
        
        self.artifacts = artifacts
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_probability_thresholds = use_probability_thresholds
        
        # Preprocessor for feature generation (shared with training)
        self.preprocessor = FeatureGeneration()
        
        # Action mapping for discrete predictions
        # Model output -> Signal: 0=Hold, 1=Buy/Long, 2=Sell/Exit
        self.action_map = {0: 0, 1: 1, 2: -1}
    
    
    @classmethod
    def from_artifacts(
        cls,
        model_path: str,
        scaler_path: str,
        features_path: str,
        **kwargs
    ) -> 'SupervisedStrategy':
        """
        Create strategy instance from saved artifact files.
        
        This is the recommended way to instantiate the strategy for backtesting.
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved normalizer
            features_path: Path to saved feature names
            **kwargs: Additional arguments passed to __init__
                     (buy_threshold, sell_threshold, use_probability_thresholds)
            
        Returns:
            Initialized SupervisedStrategy
        """
        artifacts = ModelArtifacts.load(model_path, scaler_path, features_path)
        return cls(artifacts=artifacts, **kwargs)
    
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SupervisedStrategy':
        """
        Factory method to create strategy from configuration dictionary.
        
        Required config keys:
            - model_path: Path to saved model
            - scaler_path: Path to saved normalizer
            - features_path: Path to saved feature names
        
        Optional config keys:
            - buy_threshold: Probability threshold for buy (default 0.55)
            - sell_threshold: Probability threshold for sell (default 0.45)
            - use_probability_thresholds: Whether to use prob thresholds (default True)
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Initialized SupervisedStrategy
        """
        model_path = config.get('model_path')
        scaler_path = config.get('scaler_path')
        features_path = config.get('features_path')
        
        if not all([model_path, scaler_path, features_path]):
            raise ValueError(
                "Config must contain model_path, scaler_path, and features_path"
            )
        
        return cls.from_artifacts(
            model_path=model_path,
            scaler_path=scaler_path,
            features_path=features_path,
            buy_threshold=config.get('buy_threshold', 0.55),
            sell_threshold=config.get('sell_threshold', 0.45),
            use_probability_thresholds=config.get('use_probability_thresholds', True)
        )
    
    
    def get_parameter_grid(self) -> Dict[str, List]:
        """
        Get parameter grid for strategy optimization.
        
        Compatible with Simulation class parameter sweeps.
        
        Returns:
            Dictionary of parameter names to lists of values
        """
        return {
            'buy_thresholds': [0.55, 0.60, 0.65, 0.70],
            'sell_thresholds': [0.35, 0.40, 0.45, 0.50],
        }
    
    
    def get_min_required_bars(self) -> int:
        """
        Get minimum number of bars required for signal generation.
        
        Returns:
            Minimum bars needed (based on feature generation lookback)
        """
        return self.preprocessor.min_lookback
    
    
    def generate_signals(
        self,
        df: Union[pd.DataFrame, np.ndarray],
        required_features: Optional[List[str]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate trading signals from OHLCV data.
        
        This is the main method called during backtesting. It:
        1. Generates features using the same FeatureGeneration as training
        2. Validates all required features are present
        3. Normalizes using the fitted scalers
        4. Generates predictions
        5. Converts to signals
        
        Args:
            df: DataFrame with OHLCV columns (and optionally 'vwap')
                OR numpy array of close prices (limited functionality)
            required_features: Optional override for feature names.
                              If None, uses artifacts.feature_names.
            **kwargs: Additional arguments (ignored, for API compatibility)
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
            
        Raises:
            RuntimeError: If model artifacts not loaded
            ValueError: If required features cannot be generated
        """
        if self.artifacts is None:
            raise RuntimeError(
                "Model artifacts not loaded. Use from_artifacts() or set artifacts."
            )
        
        # Handle numpy array input (limited - just return hold signals)
        if isinstance(df, np.ndarray):
            return np.zeros(len(df), dtype=np.int32)
        
        feature_names = required_features or self.artifacts.feature_names
        n = len(df)
        
        # Generate features using unified preprocessing
        try:
            features_df = self.preprocessor.generate_features_for_inference(
                df,
                required_features=feature_names
            )
        except ValueError as e:
            # If features can't be generated, return hold signals
            print(f"Warning: Feature generation failed: {e}")
            return np.zeros(n, dtype=np.int32)
        
        # Normalize features using fitted scalers
        features_normalized = self.artifacts.normalizer.transform(features_df)
        
        # Generate predictions and convert to signals
        if self.use_probability_thresholds:
            signals = self._generate_signals_from_probabilities(
                features_normalized[feature_names].values
            )
        else:
            signals = self._generate_signals_from_predictions(
                features_normalized[feature_names].values
            )
        
        return signals
    
    
    def _generate_signals_from_probabilities(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Generate signals using probability thresholds.
        
        Uses predict_proba to get class probabilities, then applies
        buy/sell thresholds with position tracking.
        
        Args:
            X: Normalized feature array
            
        Returns:
            Array of signals
        """
        n = len(X)
        signals = np.zeros(n, dtype=np.int32)
        
        # Check if model supports predict_proba
        if not hasattr(self.artifacts.model, 'predict_proba'):
            return self._generate_signals_from_predictions(X)
        
        try:
            # Get probabilities for class 1 (bullish)
            probabilities = self.artifacts.model.predict_proba(X)[:, 1]
        except Exception:
            return self._generate_signals_from_predictions(X)
        
        # Track position state for proper signal generation
        in_position = False
        
        for i in range(n):
            prob = probabilities[i]
            
            if prob >= self.buy_threshold and not in_position:
                signals[i] = self.SIGNAL_LONG  # Enter long
                in_position = True
            elif prob <= self.sell_threshold and in_position:
                signals[i] = self.SIGNAL_EXIT  # Exit
                in_position = False
            else:
                signals[i] = self.SIGNAL_HOLD  # Hold
        
        return signals
    
    
    def _generate_signals_from_predictions(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Generate signals from discrete model predictions.
        
        Uses raw predict() output for models without probability support.
        
        Args:
            X: Normalized feature array
            
        Returns:
            Array of signals
        """
        n = len(X)
        signals = np.zeros(n, dtype=np.int32)
        
        try:
            predictions = self.artifacts.model.predict(X)
        except Exception:
            return signals
        
        # Track position state
        in_position = False
        
        for i in range(n):
            pred = predictions[i]
            
            if isinstance(pred, np.ndarray):
                # Neural network output - take argmax
                pred = pred.argmax()
            
            # For binary classification: 1 = bullish signal
            if pred == 1 and not in_position:
                signals[i] = self.SIGNAL_LONG  # Enter
                in_position = True
            elif pred == 0 and in_position:
                signals[i] = self.SIGNAL_EXIT  # Exit
                in_position = False
        
        return signals
    
    
    def generate_signals_for_backtest(
        self,
        current_day_info: Dict[str, Any],
        precomputed_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Generate signals from BackTest's precomputed data format.
        
        This method provides compatibility with the BackTest class's
        internal data structures.
        
        Args:
            current_day_info: Dictionary with precomputed day data:
                - close_prices: np.ndarray
                - volumes: np.ndarray
                - high_prices: np.ndarray (optional)
                - low_prices: np.ndarray (optional)
                - vwap: np.ndarray (optional)
                - open: float
                - min_from_open: np.ndarray (optional)
                - ticker_dvol: float (optional)
            precomputed_data: Full precomputed data dict (unused, for API compat)
            
        Returns:
            Array of signals for the day
        """
        # Build DataFrame from precomputed arrays
        close_prices = current_day_info['close_prices']
        n = len(close_prices)
        
        # Use actual high/low if available, otherwise estimate from close
        high_prices = current_day_info.get('high_prices')
        if high_prices is None:
            high_prices = np.maximum.accumulate(close_prices)
        
        low_prices = current_day_info.get('low_prices')
        if low_prices is None:
            low_prices = np.minimum.accumulate(close_prices)
        
        df = pd.DataFrame({
            'close': close_prices,
            'volume': current_day_info.get('volumes', np.zeros(n)),
            'open': np.full(n, current_day_info.get('open', close_prices[0])),
            'high': high_prices,
            'low': low_prices,
        })
        
        # Add time features - critical for models trained with time data
        if 'min_from_open' in current_day_info:
            min_from_open = current_day_info['min_from_open']
            df['min_from_open'] = min_from_open
            df['minute_of_day'] = min_from_open + 570  # 9:30 AM = 570 minutes
            df['time_of_day_pct'] = min_from_open / 390.0
            df['hour_of_day'] = ((min_from_open + 570) // 60).astype(int)
            df['near_open'] = (min_from_open <= 30).astype(int)
            df['near_close'] = (min_from_open >= 360).astype(int)
        else:
            # Create time features from array index (assumes 1-minute data)
            df['min_from_open'] = np.arange(n)
            df['minute_of_day'] = df['min_from_open'] + 570
            df['time_of_day_pct'] = df['min_from_open'] / 390.0
            df['hour_of_day'] = ((df['min_from_open'] + 570) // 60).astype(int)
            df['near_open'] = (df['min_from_open'] <= 30).astype(int)
            df['near_close'] = (df['min_from_open'] >= 360).astype(int)
        
        # Add day column for groupby operations in preprocessing
        df['day'] = 'current_day'
        
        # Add optional columns
        if 'vwap' in current_day_info:
            df['vwap'] = current_day_info['vwap']
        
        return self.generate_signals(df)
    
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Useful for debugging and logging.
        
        Returns:
            Dictionary with model metadata
        """
        if self.artifacts is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'model_type': type(self.artifacts.model).__name__,
            'n_features': len(self.artifacts.feature_names),
            'feature_names': self.artifacts.feature_names[:10],  # First 10
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'scaler_type': self.artifacts.normalizer.scaler_type
        }
    
    
    def get_probabilities(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get raw model probabilities for the entire dataset.
        
        This is used for pre-computing probabilities once, then generating
        signals with different thresholds without reloading the model.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            Array of probabilities for class 1 (bullish), or None if failed
        """
        if self.artifacts is None:
            return None
        
        feature_names = self.artifacts.feature_names
        
        try:
            # Generate features
            features_df = self.preprocessor.generate_features_for_inference(
                df,
                required_features=feature_names
            )
            
            # Normalize
            features_normalized = self.artifacts.normalizer.transform(features_df)
            
            # Get probabilities
            X = features_normalized[feature_names].values
            
            if hasattr(self.artifacts.model, 'predict_proba'):
                return self.artifacts.model.predict_proba(X)[:, 1]
            else:
                # For models without predict_proba, return predictions as 0/1
                return self.artifacts.model.predict(X).astype(np.float64)
                
        except Exception as e:
            print(f"Warning: Failed to get probabilities: {e}")
            return None
    
    
    def generate_signals_from_probabilities(
        self,
        probabilities: np.ndarray,
        buy_threshold: Optional[float] = None,
        sell_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate signals from pre-computed probabilities.
        
        This is MUCH faster than generate_signals() when running multiple
        backtests with different thresholds on the same data.
        
        Args:
            probabilities: Pre-computed probability array
            buy_threshold: Override buy threshold (default: use self.buy_threshold)
            sell_threshold: Override sell threshold (default: use self.sell_threshold)
            
        Returns:
            Array of signals
        """
        buy_thresh = buy_threshold if buy_threshold is not None else self.buy_threshold
        sell_thresh = sell_threshold if sell_threshold is not None else self.sell_threshold
        
        n = len(probabilities)
        signals = np.zeros(n, dtype=np.int32)
        
        in_position = False
        
        for i in range(n):
            prob = probabilities[i]
            
            if np.isnan(prob):
                signals[i] = self.SIGNAL_HOLD
                continue
            
            if prob >= buy_thresh and not in_position:
                signals[i] = self.SIGNAL_LONG
                in_position = True
            elif prob <= sell_thresh and in_position:
                signals[i] = self.SIGNAL_EXIT
                in_position = False
            else:
                signals[i] = self.SIGNAL_HOLD
        
        return signals
    
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information.
        
        Returns:
            Dictionary with strategy metadata
        """
        info = {
            'name': self.__class__.__name__,
            'min_required_bars': self.get_min_required_bars(),
            'parameter_grid': self.get_parameter_grid(),
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'use_probability_thresholds': self.use_probability_thresholds
        }
        info.update(self.get_model_info())
        return info


def generate_ml_signals(
    df: pd.DataFrame,
    model_path: str,
    scaler_path: str,
    features_path: str,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45
) -> np.ndarray:
    """
    Convenience function to generate signals without creating strategy object.
    
    Useful for one-off signal generation or testing.
    
    Args:
        df: DataFrame with OHLCV data
        model_path: Path to saved model
        scaler_path: Path to saved normalizer
        features_path: Path to saved feature names
        buy_threshold: Probability threshold for buying
        sell_threshold: Probability threshold for selling
        
    Returns:
        Array of signals
    """
    strategy = SupervisedStrategy.from_artifacts(
        model_path=model_path,
        scaler_path=scaler_path,
        features_path=features_path,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold
    )
    
    return strategy.generate_signals(df)
