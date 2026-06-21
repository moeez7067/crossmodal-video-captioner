"""
Experiment runners and orchestration for comparative analysis.
"""

from .runner import ExperimentRunner
from .analyzer import ResultsAnalyzer
from .config import (
    EXPERIMENT_CONFIG,
    DEFAULT_MODELS,
    EVALUATION_METRICS,
    get_model_config,
    get_experiment_config
)

__all__ = [
    "ExperimentRunner",
    "ResultsAnalyzer",
    "EXPERIMENT_CONFIG",
    "DEFAULT_MODELS",
    "EVALUATION_METRICS",
    "get_model_config",
    "get_experiment_config"
]

