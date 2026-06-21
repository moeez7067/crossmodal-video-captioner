"""
Training Module for Fusion Model.

This module contains all training-related components:
- Dataset preparation
- Training scripts
- Training metrics
- Enhanced trainer
"""

from ..train_fusion import MultimodalDataset, FusionTrainer
from ..enhanced_trainer import EnhancedFusionTrainer
from ..dataset_preparation import DatasetPreparator, prepare_training_dataset
from ..training_metrics import TrainingMetrics

__all__ = [
    'MultimodalDataset',
    'FusionTrainer',
    'EnhancedFusionTrainer',
    'DatasetPreparator',
    'prepare_training_dataset',
    'TrainingMetrics',
]
