"""
Comparative analysis module for video captioning.
"""

from .base_model import BaseCaptionModel
from .model_registry import ModelRegistry, register_model, create_model, list_models

__all__ = [
    "BaseCaptionModel",
    "ModelRegistry",
    "register_model",
    "create_model",
    "list_models"
]

