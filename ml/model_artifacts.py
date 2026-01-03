# algo-backtests/ml/model_artifacts.py
import os
import pickle
from dataclasses import dataclass
from typing import List, Any, Dict, Optional
from datetime import datetime
from .feature_normalization import FeatureNormalization
from .training_config import TrainingConfig


@dataclass
class ModelArtifacts:
    """
    Container for loaded model artifacts - just holds data.
    
    Args:
        model: Trained model
        normalizer: Fitted normalizer
        feature_names: List of feature names
        config: Training configuration
    """
    model: Any
    normalizer: FeatureNormalization
    feature_names: List[str]
    config: Optional[TrainingConfig] = None
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str, features_path: str) -> 'ModelArtifacts':
        """
        Load artifacts from files.
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
            features_path: Path to features file
            
        Returns:
            ModelArtifacts instance
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        normalizer = FeatureNormalization.load(scaler_path)
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        return cls(model=model, normalizer=normalizer, feature_names=feature_names)


class ArtifactManager:
    """
    Handles saving/loading/finding artifacts.
    
    Args:
        base_dir: Base directory for artifacts
    """
    
    def __init__(self, base_dir: str = 'outputs/models'):
        self.base_dir = base_dir
    
    def save(
        self,
        artifacts: ModelArtifacts,
        ticker: str,
        model_type: str,
        config: Optional[TrainingConfig] = None
    ) -> Dict[str, str]:
        """
        Save artifacts and return paths.
        
        Args:
            artifacts: ModelArtifacts instance
            ticker: Stock ticker symbol
            model_type: Model type (e.g. 'xgboost', 'gradient_boosting', 'random_forest')
            config: Training configuration
        
        Returns:
            Dictionary with artifact paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.base_dir, ticker)
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = f"{ticker}_{model_type}_{timestamp}"
        paths = {}
        
        # Save model
        paths['model_path'] = os.path.join(output_dir, f"{base_name}_model.pkl")
        with open(paths['model_path'], 'wb') as f:
            pickle.dump(artifacts.model, f)
        
        # Save normalizer
        paths['scaler_path'] = os.path.join(output_dir, f"{base_name}_scalers.pkl")
        artifacts.normalizer.save(paths['scaler_path'])
        
        # Save feature names
        paths['features_path'] = os.path.join(output_dir, f"{base_name}_features.pkl")
        with open(paths['features_path'], 'wb') as f:
            pickle.dump(artifacts.feature_names, f)
        
        # Save config if provided
        if config:
            paths['config_path'] = os.path.join(output_dir, f"{base_name}_config.pkl")
            with open(paths['config_path'], 'wb') as f:
                pickle.dump(config, f)
        
        return paths
    

    def find_latest(self, ticker: str, model_type: str) -> Dict[str, str]:
        """
        Find most recent artifacts for ticker/model_type.
        
        Args:
            ticker: Stock ticker symbol
            model_type: Model type (e.g. 'xgboost', 'gradient_boosting', 'random_forest')
            
        Returns:
            Dictionary with artifact paths
        """
        models_dir = os.path.join(self.base_dir, ticker)
        
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Find all matching model files
        pattern = f'{ticker}_{model_type}_'
        model_files = [
            f for f in os.listdir(models_dir) 
            if f.startswith(pattern) and f.endswith('_model.pkl')
        ]
        
        if not model_files:
            raise FileNotFoundError(f"No {model_type} model found for {ticker}")
        
        # Sort by modification time
        model_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(models_dir, x)),
            reverse=True
        )
        
        # Extract base name (without _model.pkl)
        latest = model_files[0]
        base = latest.replace('_model.pkl', '')
        
        return {
            'model_path': os.path.join(models_dir, f'{base}_model.pkl'),
            'scaler_path': os.path.join(models_dir, f'{base}_scalers.pkl'),
            'features_path': os.path.join(models_dir, f'{base}_features.pkl'),
            'config_path': os.path.join(models_dir, f'{base}_config.pkl'),
        }
    

    def load_latest(self, ticker: str, model_type: str) -> ModelArtifacts:
        """
        Convenience method to find and load latest artifacts.
        
        Args:
            ticker: Stock ticker symbol
            model_type: Model type (e.g. 'xgboost', 'gradient_boosting', 'random_forest')
            
        Returns:
            ModelArtifacts instance
        """
        paths = self.find_latest(ticker, model_type)
        return ModelArtifacts.load(
            paths['model_path'],
            paths['scaler_path'],
            paths['features_path']
        )
