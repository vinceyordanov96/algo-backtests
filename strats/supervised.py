"""
ML Strategy wrapper for integration with existing backtesting framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import pickle


class SupervisedStrategy:
    """
    ML-based trading strategy that integrates with existing BackTest class.
    
    This class wraps a trained ML model (or RL agent) and generates signals
    compatible with simulate_positions_numba.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        window_size: int = 30,
        action_map: Dict[int, int] = None
    ):
        """
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scalers
            feature_names: List of feature column names
            window_size: Lookback window for observations
            action_map: Map model outputs to signals {model_output: signal}
                       Default: {0: 0, 1: 1, 2: -1} (Hold, Buy, Sell)
        """
        self.model = None
        self.scalers = None
        self.feature_names = feature_names or []
        self.window_size = window_size
        self.action_map = action_map or {0: 0, 1: 1, 2: -1}  # Hold, Buy, Sell
        
        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_scalers(scaler_path)
    
    
    def load_model(self, path: str):
        """
        Load trained model from file.
        
        Args:
            path: Path to saved model
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
    
    
    def load_scalers(self, path: str):
        """
        Load scalers from file.
        
        Args:
            path: Path to saved scalers
        """
        with open(path, 'rb') as f:
            self.scalers = pickle.load(f)
    
    
    def get_parameter_grid(self) -> Dict[str, list]:
        """
        Get parameter grid for ML strategy (for compatibility with Simulation class).
        
        Returns:
            - Dictionary of parameter names to lists of values to test
                - window_sizes: List of window sizes to test
                - model_types: List of model types to test
        """
        return {
            'window_sizes': [20, 30, 50],
            'model_types': ['random_forest', 'xgboost', 'dqn']
        }
    
    
    @staticmethod
    def generate_signals(
        features_df: pd.DataFrame,
        model,
        scalers: Dict[str, Any],
        feature_names: List[str],
        window_size: int = 30
    ) -> np.ndarray:
        """
        Generate trading signals using the ML model.
        
        This method is designed to be compatible with your existing
        backtesting infrastructure.
        
        Args:
            features_df: DataFrame with all features for the day
            model: Trained model with predict() method
            scalers: Dictionary of fitted scalers per feature
            feature_names: List of feature column names to use
            window_size: Lookback window for observations
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
        """
        n = len(features_df)
        signals = np.zeros(n, dtype=np.int32)
        
        if n < window_size:
            return signals
        
        # Normalize features
        X_normalized = features_df[feature_names].copy()
        for col in feature_names:
            if col in scalers:
                values = X_normalized[col].values.reshape(-1, 1)
                X_normalized[col] = scalers[col].transform(values).flatten()
        
        # Generate signals for each timestep
        for i in range(window_size, n):
            # Extract windowed observation
            window = X_normalized.iloc[i-window_size:i].values
            
            # Flatten for model input (shape: [1, window_size * n_features])
            obs = window.flatten().reshape(1, -1)
            
            try:
                # Get model prediction
                action = model.predict(obs)[0]
                
                # Map to signal
                if isinstance(action, np.ndarray):
                    action = action.argmax()  # For neural network outputs
                
                signals[i] = {0: 0, 1: 1, 2: -1}.get(action, 0)
                
            except Exception:
                signals[i] = 0  # Default to hold on error
        
        return signals
    
    
    @staticmethod
    def generate_signals_from_probabilities(
        features_df: pd.DataFrame,
        model,
        scalers: Dict[str, Any],
        feature_names: List[str],
        window_size: int = 30,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4
    ) -> np.ndarray:
        """
        Convert model probability outputs to discrete signals.
        
        Args:
            features_df: DataFrame with all features for the day
            model: Trained model with predict_proba() method
            scalers: Dictionary of fitted scalers per feature
            feature_names: List of feature column names to use
            window_size: Lookback window for observations
            buy_threshold: Threshold for buying
            sell_threshold: Threshold for selling
            
        Returns:
            Array of signals: 1 (long), -1 (exit), 0 (hold)
        """
        n = len(features_df)
        signals = np.zeros(n, dtype=np.int32)
        
        if n < window_size:
            return signals
        
        # Normalize features
        X_normalized = features_df[feature_names].copy()
        for col in feature_names:
            if col in scalers:
                values = X_normalized[col].values.reshape(-1, 1)
                X_normalized[col] = scalers[col].transform(values).flatten()
        
        in_position = False
        
        for i in range(window_size, n):
            window = X_normalized.iloc[i-window_size:i].values
            obs = window.flatten().reshape(1, -1)
            
            try:
                # Get probability of class 1 (bullish)
                prob = model.predict_proba(obs)[0, 1]
                
                if prob >= buy_threshold and not in_position:
                    signals[i] = 1  # Enter long
                    in_position = True
                elif prob <= sell_threshold and in_position:
                    signals[i] = -1  # Exit
                    in_position = False
                else:
                    signals[i] = 0  # Hold
                    
            except Exception:
                signals[i] = 0
        
        return signals
