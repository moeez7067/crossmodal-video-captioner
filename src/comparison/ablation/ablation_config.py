"""
Configuration for ablation studies.
Defines which components to ablate and how to run ablation experiments.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import config as project_config

# Ablation experiment settings
ABLATION_EXPERIMENT_NAME = "ablation_study"
ABLATION_RESULTS_DIR = Path("results") / ABLATION_EXPERIMENT_NAME
ABLATION_PREDICTIONS_DIR = ABLATION_RESULTS_DIR / "predictions"
ABLATION_SUMMARY_DIR = ABLATION_RESULTS_DIR / "summary"
ABLATION_VISUALIZATIONS_DIR = ABLATION_RESULTS_DIR / "visualizations"

# Create directories
for dir_path in [ABLATION_RESULTS_DIR, ABLATION_PREDICTIONS_DIR, 
                 ABLATION_SUMMARY_DIR, ABLATION_VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Base model for ablation (full model with all components)
BASE_MODEL_NAME = "gated"  # Use gated fusion as the full model

# Ablation variants to test
ABLATION_VARIANTS = {
    # Full model (baseline)
    "full": {
        "name": "full",
        "description": "Full model with all components",
        "ablation_type": "baseline",
        "components": {
            "use_audio": True,
            "use_visual": True,
            "use_gating": True,
            "use_projection": True,
            "use_temporal_alignment": True,
            "gating_complexity": "full"  # "full", "simple", "none"
        }
    },
    
    # Remove gating mechanism
    "no_gating": {
        "name": "no_gating",
        "description": "Remove gating mechanism, use simple addition",
        "ablation_type": "component_removal",
        "components": {
            "use_audio": True,
            "use_visual": True,
            "use_gating": False,
            "use_projection": True,
            "use_temporal_alignment": True,
            "gating_complexity": "none"
        }
    },
    
    # Remove audio modality
    "no_audio": {
        "name": "no_audio",
        "description": "Remove audio modality, visual only",
        "ablation_type": "modality_removal",
        "components": {
            "use_audio": False,
            "use_visual": True,
            "use_gating": False,  # No gating needed with single modality
            "use_projection": True,
            "use_temporal_alignment": False,
            "gating_complexity": "none"
        }
    },
    
    # Remove visual modality
    "no_visual": {
        "name": "no_visual",
        "description": "Remove visual modality, audio only",
        "ablation_type": "modality_removal",
        "components": {
            "use_audio": True,
            "use_visual": False,
            "use_gating": False,  # No gating needed with single modality
            "use_projection": True,
            "use_temporal_alignment": False,
            "gating_complexity": "none"
        }
    },
    
    # Remove projection layers
    "no_projection": {
        "name": "no_projection",
        "description": "Remove projection layers",
        "ablation_type": "component_removal",
        "components": {
            "use_audio": True,
            "use_visual": True,
            "use_gating": True,
            "use_projection": False,
            "use_temporal_alignment": True,
            "gating_complexity": "full"
        }
    },
    
    # Simplified gating (single layer instead of multi-layer)
    "simple_gating": {
        "name": "simple_gating",
        "description": "Simplify gating network (single layer)",
        "ablation_type": "component_simplification",
        "components": {
            "use_audio": True,
            "use_visual": True,
            "use_gating": True,
            "use_projection": True,
            "use_temporal_alignment": True,
            "gating_complexity": "simple"
        }
    },
    
    # Remove temporal alignment
    "no_temporal_alignment": {
        "name": "no_temporal_alignment",
        "description": "Remove temporal alignment between audio and visual",
        "ablation_type": "component_removal",
        "components": {
            "use_audio": True,
            "use_visual": True,
            "use_gating": True,
            "use_projection": True,
            "use_temporal_alignment": False,
            "gating_complexity": "full"
        }
    },
    
    # Remove both gating and projection
    "no_gating_no_projection": {
        "name": "no_gating_no_projection",
        "description": "Remove both gating and projection, use raw concatenation",
        "ablation_type": "component_removal",
        "components": {
            "use_audio": True,
            "use_visual": True,
            "use_gating": False,
            "use_projection": False,
            "use_temporal_alignment": True,
            "gating_complexity": "none"
        }
    }
}

# Evaluation metrics (same as comparative analysis)
ABLATION_METRICS = [
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
    "base_model": BASE_MODEL_NAME,
    "variants": list(ABLATION_VARIANTS.keys()),
    "metrics": ABLATION_METRICS,
    "results_dir": str(ABLATION_RESULTS_DIR),
    "predictions_dir": str(ABLATION_PREDICTIONS_DIR),
    "summary_dir": str(ABLATION_SUMMARY_DIR),
    "visualizations_dir": str(ABLATION_VISUALIZATIONS_DIR),
    "save_predictions": True,
    "save_evaluations": True,
    "generate_visualizations": True,
    "generate_report": True,
}

def get_ablation_variant(variant_name: str) -> Dict[str, Any]:
    """Get configuration for a specific ablation variant."""
    return ABLATION_VARIANTS.get(variant_name, {}).copy()

def get_ablation_config() -> Dict[str, Any]:
    """Get complete ablation experiment configuration."""
    return ABLATION_CONFIG.copy()

def list_ablation_variants() -> List[str]:
    """List all available ablation variants."""
    return list(ABLATION_VARIANTS.keys())

def get_ablation_description(variant_name: str) -> str:
    """Get human-readable description of an ablation variant."""
    variant = ABLATION_VARIANTS.get(variant_name, {})
    return variant.get("description", f"Unknown variant: {variant_name}")
