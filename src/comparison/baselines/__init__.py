"""
Baseline models for video captioning comparison.
"""

from .audio_only import AudioOnlyBaseline
from .visual_only import VisualOnlyBaseline
from .simple_fusion import (
    SimpleConcatenationFusion,
    AdditionFusion,
    GatedFusion
)

__all__ = [
    "AudioOnlyBaseline",
    "VisualOnlyBaseline",
    "SimpleConcatenationFusion",
    "AdditionFusion",
    "GatedFusion"
]

