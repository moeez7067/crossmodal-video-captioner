"""
Main script to run complete comparative analysis.
Orchestrates experiment runner and results analyzer.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.comparison.experiments.runner import ExperimentRunner
from src.comparison.experiments.analyzer import ResultsAnalyzer
from src.data.dataset_loader import VideoCaptionDataset, GroundTruthManager
from src.comparison.model_registry import ModelRegistry
from src.comparison.baselines import (
    AudioOnlyBaseline,
    VisualOnlyBaseline,
    SimpleConcatenationFusion,
    AdditionFusion,
    GatedFusion
)
from src.utils.logger import get_logger
from src.comparison.experiments.config import get_experiment_config

logger = get_logger(__name__)


def register_all_models():
    """Register all baseline models."""
    ModelRegistry.register("audio_only", AudioOnlyBaseline)
    ModelRegistry.register("visual_only", VisualOnlyBaseline)
    ModelRegistry.register("concatenation", SimpleConcatenationFusion)
    ModelRegistry.register("addition", AdditionFusion)
    ModelRegistry.register("gated", GatedFusion)
    logger.info("Registered all models")


def main():
    """Run complete comparative analysis."""
    print("=" * 80)
    print("COMPARATIVE ANALYSIS - VIDEO CAPTIONING")
    print("=" * 80)
    print()
    
    # Register models
    print("Step 1: Registering models...")
    register_all_models()
    print(f"   Registered {len(ModelRegistry.list_models())} models")
    print()
    
    # Auto-scan for videos in the videos directory
    print("Step 2: Scanning for videos...")
    try:
        # Import the scan function
        import importlib.util
        prepare_script = project_root / "scripts" / "prepare_test_dataset.py"
        spec = importlib.util.spec_from_file_location("prepare_test_dataset", prepare_script)
        prepare_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare_module)
        added = prepare_module.scan_and_add_all_videos(category="lecture", split="test")
        if added > 0:
            print(f"   Added {added} new video(s) to dataset")
        else:
            print("   No new videos found (using existing metadata)")
    except Exception as e:
        logger.warning(f"Could not auto-scan for videos: {e}")
        print("   Skipping auto-scan (using existing metadata)")
    
    # Load dataset
    print("Step 2b: Loading dataset...")
    dataset = VideoCaptionDataset()
    if len(dataset) == 0:
        print("   ERROR: No videos found in dataset!")
        print("   Please add videos using: python scripts/prepare_test_dataset.py --add-video <path>")
        return
    
    print(f"   Loaded {len(dataset)} videos")
    print()
    
    # Run experiments
    print("Step 3: Running experiments...")
    print("   This may take a while depending on video length and number of models...")
    print()
    
    runner = ExperimentRunner(dataset=dataset)
    results = runner.run_all_models(split="test")  # Use test split, or None for all
    
    # Save experiment results
    runner.save_results()
    print()
    
    # Analyze results
    print("Step 4: Analyzing results...")
    predictions = runner.get_predictions_for_evaluation()
    
    if not predictions:
        print("   ERROR: No predictions found!")
        print("   Check experiment results for errors.")
        return
    
    analyzer = ResultsAnalyzer(predictions)
    
    # Compute metrics
    print("   Computing evaluation metrics...")
    analyzer.compute_metrics()
    
    # Generate comparison table
    print("   Generating comparison table...")
    comparison_table = analyzer.generate_comparison_table()
    print(comparison_table)
    print()
    
    # Save all results
    print("Step 5: Saving results...")
    analyzer.save_all_results()
    print()
    
    # Generate report
    print("Step 6: Generating report...")
    report = analyzer.generate_report()
    print(report)
    print()
    
    print("=" * 80)
    print("COMPARATIVE ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("Results saved to:")
    print(f"  - Predictions: results/comparative_analysis/predictions/")
    print(f"  - Summary: results/comparative_analysis/summary/")
    print(f"  - Visualizations: results/comparative_analysis/visualizations/")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running comparative analysis: {e}", exc_info=True)
        print(f"\n\nERROR: {e}")
        sys.exit(1)
