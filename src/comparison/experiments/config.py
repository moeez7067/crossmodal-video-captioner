"""
Configuration for comparative analysis experiments.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import config as project_config

# Experiment settings
EXPERIMENT_NAME = "comparative_analysis"
RESULTS_DIR = Path("results") / EXPERIMENT_NAME
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
SUMMARY_DIR = RESULTS_DIR / "summary"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"

# Create directories
for dir_path in [RESULTS_DIR, PREDICTIONS_DIR, SUMMARY_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Models to run (can be customized)
DEFAULT_MODELS = [
    "audio_only",
    "visual_only",
    "concatenation",
    "addition",
    "gated"
]

# Evaluation settings
EVALUATION_METRICS = [
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "cider",
    "meteor",
    "rouge_l"
]

# Experiment configuration
EXPERIMENT_CONFIG = {
    "experiment_name": EXPERIMENT_NAME,
    "models": DEFAULT_MODELS,
    "metrics": EVALUATION_METRICS,
    "results_dir": str(RESULTS_DIR),
    "predictions_dir": str(PREDICTIONS_DIR),
    "summary_dir": str(SUMMARY_DIR),
    "visualizations_dir": str(VISUALIZATIONS_DIR),
    "save_predictions": True,
    "save_evaluations": True,
    "generate_visualizations": True,
    "generate_report": True,
}

# Model-specific configurations (optional)
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "audio_only": {},
    "visual_only": {
        "pooling_method": "mean"
    },
    "concatenation": {
        "use_projection": True
    },
    "addition": {
        "embedding_dim": 512
    },
    "gated": {
        "embedding_dim": 512
    }
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name, {})

def get_experiment_config() -> Dict[str, Any]:
    """Get complete experiment configuration."""
    return EXPERIMENT_CONFIG.copy()
