"""
Dataset loading and ground truth management.
"""

from .dataset_loader import VideoCaptionDataset
from .ground_truth import GroundTruthManager

__all__ = ['VideoCaptionDataset', 'GroundTruthManager']

