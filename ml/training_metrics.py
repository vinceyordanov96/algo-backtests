# algo-backtests/ml/training_metrics.py
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """Container for training/evaluation metrics."""
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc
        }
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        return (
            f"Accuracy:  {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\n"
            f"F1 Score:  {self.f1:.4f}\n"
            f"ROC-AUC:   {self.roc_auc:.4f}"
        )
