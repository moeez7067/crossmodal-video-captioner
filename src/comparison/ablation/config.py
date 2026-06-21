"""
Configuration for ablation studies.
Defines ablation variants and experiment settings.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import config as project_config

# Experiment settings
ABLATION_EXPERIMENT_NAME = "ablation_study"
ABLATION_RESULTS_DIR = Path("results") / ABLATION_EXPERIMENT_NAME
ABLATION_PREDICTIONS_DIR = ABLATION_RESULTS_DIR / "predictions"
ABLATION_SUMMARY_DIR = ABLATION_RESULTS_DIR / "summary"
ABLATION_VISUALIZATIONS_DIR = ABLATION_RESULTS_DIR / "visualizations"

# Create directories
for dir_path in [ABLATION_RESULTS_DIR, ABLATION_PREDICTIONS_DIR, 
                 ABLATION_SUMMARY_DIR, ABLATION_VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Base models to perform ablation on
BASE_MODELS_FOR_ABLATION = [
    "gated",  # Primary model for ablation
    "concatenation",  # Can also ablate this
]

# Ablation variants - components to remove
ABLATION_VARIANTS = {
    # Gated Fusion Ablations
    "gated_no_gating": {
        "base_model": "gated",
        "description": "Gated fusion without gating mechanism (falls back to addition)",
        "ablation_type": "remove_gating",
        "components_removed": ["gating_network"]
    },
    "gated_no_audio": {
        "base_model": "gated",
        "description": "Gated fusion without audio modality",
        "ablation_type": "remove_modality",
        "components_removed": ["audio"],
        "modality_removed": "audio"
    },
    "gated_no_visual": {
        "base_model": "gated",
        "description": "Gated fusion without visual modality",
        "ablation_type": "remove_modality",
        "components_removed": ["visual"],
        "modality_removed": "visual"
    },
    "gated_no_projection": {
        "base_model": "gated",
        "description": "Gated fusion without projection layers",
        "ablation_type": "remove_projection",
        "components_removed": ["audio_projection", "visual_projection"]
    },
    
    # Concatenation Ablations
    "concatenation_no_projection": {
        "base_model": "concatenation",
        "description": "Concatenation fusion without projection layer",
        "ablation_type": "remove_projection",
        "components_removed": ["projection"]
    },
    "concatenation_no_audio": {
        "base_model": "concatenation",
        "description": "Concatenation fusion without audio modality",
        "ablation_type": "remove_modality",
        "components_removed": ["audio"],
        "modality_removed": "audio"
    },
    "concatenation_no_visual": {
        "base_model": "concatenation",
        "description": "Concatenation fusion without visual modality",
        "ablation_type": "remove_modality",
        "components_removed": ["visual"],
        "modality_removed": "visual"
    },
}

# Evaluation metrics (same as comparative analysis)
EVALUATION_METRICS = [
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "cider",
    "meteor",
    "rouge_l"
]

# Ablation experiment configuration
ABLATION_CONFIG = {
    "experiment_name": ABLATION_EXPERIMENT_NAME,
    "base_models": BASE_MODELS_FOR_ABLATION,
    "variants": ABLATION_VARIANTS,
    "metrics": EVALUATION_METRICS,
    "results_dir": str(ABLATION_RESULTS_DIR),
    "predictions_dir": str(ABLATION_PREDICTIONS_DIR),
    "summary_dir": str(ABLATION_SUMMARY_DIR),
    "visualizations_dir": str(ABLATION_VISUALIZATIONS_DIR),
    "save_predictions": True,
    "save_evaluations": True,
    "generate_visualizations": True,
    "generate_report": True,
    "compare_with_baseline": True,  # Compare each ablation with full model
}

def get_ablation_config() -> Dict[str, Any]:
    """Get complete ablation experiment configuration."""
    return ABLATION_CONFIG.copy()

def get_ablation_variants(base_model: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get ablation variants, optionally filtered by base model.
    
    Args:
        base_model: Optional base model name to filter variants
        
    Returns:
        Dictionary of ablation variants
    """
    if base_model is None:
        return ABLATION_VARIANTS.copy()
    
    return {
        name: variant
        for name, variant in ABLATION_VARIANTS.items()
        if variant.get("base_model") == base_model
    }

def get_all_ablation_model_names() -> List[str]:
    """Get list of all ablation variant model names."""
    return list(ABLATION_VARIANTS.keys())
