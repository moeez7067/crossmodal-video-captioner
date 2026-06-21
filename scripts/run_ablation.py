"""
Main script to run complete ablation study.
Orchestrates ablation runner and analyzer.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.comparison.ablation import AblationRunner, AblationAnalyzer
from src.comparison.ablation.config import get_ablation_config, get_all_ablation_model_names
from src.data.dataset_loader import VideoCaptionDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run complete ablation study."""
    print("=" * 80)
    print("ABLATION STUDY - VIDEO CAPTIONING")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Step 1: Loading dataset...")
    try:
        # Auto-scan for videos
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
    
    dataset = VideoCaptionDataset()
    if len(dataset) == 0:
        print("   ERROR: No videos found in dataset!")
        print("   Please add videos using: python scripts/prepare_test_dataset.py --add-video <path>")
        return
    
    print(f"   Loaded {len(dataset)} videos")
    print()
    
    # Get ablation variants
    print("Step 2: Preparing ablation variants...")
    variants = get_all_ablation_model_names()
    print(f"   Found {len(variants)} ablation variants:")
    for variant in variants:
        print(f"     - {variant}")
    print()
    
    # Run ablation experiments
    print("Step 3: Running ablation experiments...")
    print("   This may take a while depending on video length and number of variants...")
    print()
    
    config = get_ablation_config()
    runner = AblationRunner(dataset=dataset, config=config)
    results = runner.run_all_variants(split="test")
    
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
    
    analyzer = AblationAnalyzer(predictions, config=config)
    
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
    report = analyzer.generate_ablation_report()
    print(report)
    print()
    
    print("=" * 80)
    print("ABLATION STUDY COMPLETE!")
    print("=" * 80)
    print()
    print("Results saved to:")
    print(f"  - Predictions: results/ablation_study/predictions/")
    print(f"  - Summary: results/ablation_study/summary/")
    print(f"  - Visualizations: results/ablation_study/visualizations/")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAblation study interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running ablation study: {e}", exc_info=True)
        print(f"\n\nERROR: {e}")
        sys.exit(1)
