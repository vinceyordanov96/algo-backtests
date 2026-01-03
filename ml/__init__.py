"""
ML module for trading strategies.

This module provides a complete machine learning pipeline for developing
and deploying supervised learning trading strategies.

Components:
    - FeatureGeneration: Unified feature generation for training and inference
    - FeatureSelection: Two-stage feature selection with data leakage prevention
    - FeatureNormalization: Per-feature normalization with proper train/test separation
    - TrainingPipeline: Complete training workflow with proper ordering
    - SupervisedStrategy: Inference strategy for backtesting integration

Example - Training:
    from ml import TrainingPipeline, TrainingConfig
    
    config = TrainingConfig(
        ticker='NVDA',
        model_type='xgboost',
        forecast_horizon=30
    )
    
    pipeline = TrainingPipeline(config)
    result = pipeline.run()
    
    print(f"Validation ROC-AUC: {result.valid_metrics.roc_auc:.4f}")

Example - Inference:
    from ml import SupervisedStrategy
    
    strategy = SupervisedStrategy.from_artifacts(
        model_path='ml/models/NVDA_xgboost_model.pkl',
        scaler_path='ml/models/NVDA_xgboost_scalers.pkl',
        features_path='ml/models/NVDA_xgboost_features.pkl'
    )
    
    signals = strategy.generate_signals(df_ohlcv)
"""

from .feature_generation import FeatureGeneration, FeatureConfig
from .feature_selection import FeatureSelection
from .feature_normalization import FeatureNormalization, create_temporal_split
from .training_config import TrainingConfig
from .training_metrics import TrainingMetrics
from .training_results import TrainingResult
from .training_pipeline import TrainingPipeline
from .model_artifacts import ModelArtifacts, ArtifactManager


__all__ = [
    # Feature Generation
    'FeatureGeneration',
    'FeatureConfig',
    
    # Feature Selection
    'FeatureSelection',
    
    # Feature Normalization
    'FeatureNormalization',
    'create_temporal_split',
    
    # Training
    'TrainingPipeline',
    'TrainingConfig',
    'TrainingMetrics',
    'TrainingResult',
    
    # Inference / Strategy
    'ModelArtifacts',
    'ArtifactManager',
]
