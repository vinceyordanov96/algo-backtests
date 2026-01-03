# algo-backtests/ml/training_config.py
from typing import Optional
from dataclasses import dataclass



@dataclass
class TrainingConfig:
    """
    Configuration for the training pipeline.
    """
    
    # Data settings
    ticker: str
    start_date: str = '2024-01-01'
    resample_freq: Optional[str] = None
    
    # Model settings
    model_type: str = 'gradient_boosting'
    
    # Label settings
    forecast_horizon: int = 30
    return_threshold: Optional[float] = None
    
    # Split ratios
    train_ratio: float = 0.6
    test_ratio: float = 0.2
    valid_ratio: float = 0.2
    
    # Feature selection settings
    variance_threshold: float = 0.01
    roc_auc_threshold: float = 0.52
    
    # Normalization settings
    scaler_type: str = 'minmax'
    clip_outliers: bool = True
    
    # Output settings
    output_dir: Optional[str] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f'models/{self.ticker}'
    
    def validate(self) -> None:
        """
        Validate configuration values.
        """
        if self.train_ratio + self.test_ratio + self.valid_ratio != 1.0:
            raise ValueError("Split ratios must sum to 1.0")
        
        if self.model_type not in ['random_forest', 'gradient_boosting', 'xgboost']:
            raise ValueError(f"Unknown model type: {self.model_type}")
