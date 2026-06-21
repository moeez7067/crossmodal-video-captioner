"""
Script to register all models in the model registry.
Run this to ensure all models are available for experiments.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.comparison.model_registry import ModelRegistry
from src.comparison.baselines import (
    AudioOnlyBaseline,
    VisualOnlyBaseline,
    SimpleConcatenationFusion,
    AdditionFusion,
    GatedFusion
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def register_all_models():
    """Register all baseline models."""
    # Register baseline models
    ModelRegistry.register("audio_only", AudioOnlyBaseline)
    ModelRegistry.register("visual_only", VisualOnlyBaseline)
    ModelRegistry.register("concatenation", SimpleConcatenationFusion)
    ModelRegistry.register("addition", AdditionFusion)
    ModelRegistry.register("gated", GatedFusion)
    
    logger.info("Registered all baseline models")
    logger.info(f"Available models: {ModelRegistry.list_models()}")


if __name__ == "__main__":
    register_all_models()
    print(f"\nRegistered models: {', '.join(ModelRegistry.list_models())}")
