"""
Multimodal Fusion Module.

This module provides tools for fusing audio and visual embeddings using
transformer-based architectures with cross-attention mechanisms.
"""

from .fusion_pipeline import FusionPipeline
from .fusion_service import FusionService
from .multimodal_transformer import MultimodalTransformer, create_multimodal_transformer
from .cross_attention import CrossAttention, BidirectionalCrossAttention
from .utils import align_embeddings_with_timestamps, pool_embeddings

# Training and visualization
from .train import (
    MultimodalDataset,
    FusionTrainer,
    EnhancedFusionTrainer,
    DatasetPreparator,
    prepare_training_dataset,
    TrainingMetrics
)
from .attention_visualization import AttentionVisualizer

__all__ = [
    # Core fusion components
    'FusionPipeline',
    'FusionService',
    'MultimodalTransformer',
    'create_multimodal_transformer',
    'CrossAttention',
    'BidirectionalCrossAttention',
    
    # Utilities
    'align_embeddings_with_timestamps',
    'pool_embeddings',
    
    # Training
    'MultimodalDataset',
    'FusionTrainer',
    'EnhancedFusionTrainer',
    'DatasetPreparator',
    'prepare_training_dataset',
    'TrainingMetrics',
    
    # Visualization
    'AttentionVisualizer',
]
