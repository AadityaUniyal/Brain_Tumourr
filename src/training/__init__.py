"""
Training loops, evaluation engines, and performance metrics.
"""
from src.training.metrics import compute_clinical_metrics
from src.training.train import Trainer

__all__ = ["compute_clinical_metrics", "Trainer"]
