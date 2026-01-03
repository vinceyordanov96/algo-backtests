# algo-backtests/ml/training_results.py
from typing import Dict
from dataclasses import dataclass

from .training_metrics import TrainingMetrics


@dataclass
class TrainingResult:
    """Container for training pipeline results."""
    
    train_metrics: TrainingMetrics
    test_metrics: TrainingMetrics
    valid_metrics: TrainingMetrics
    paths: Dict[str, str]
    feature_count: int
    class_distribution: Dict[str, float]
