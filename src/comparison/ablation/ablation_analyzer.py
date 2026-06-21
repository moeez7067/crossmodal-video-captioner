"""
Ablation study results analyzer.
Compares ablated models with baseline models and generates analysis reports.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd

from src.evaluation.evaluator import CaptionEvaluator
from src.data.dataset_loader import GroundTruthManager
from src.comparison.ablation.config import (
    ABLATION_SUMMARY_DIR,
    ABLATION_VISUALIZATIONS_DIR,
    ABLATION_CONFIG,
    ABLATION_VARIANTS
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AblationAnalyzer:
    """
    Analyzes ablation study results.
    Compares ablated models with baseline models and identifies component contributions.
    """
    
    def __init__(self, predictions: Dict[str, Dict[str, Dict]], config: Optional[Dict[str, Any]] = None):
        """
        Initialize ablation analyzer.
        
        Args:
            predictions: Dictionary mapping variant_name -> video_id -> prediction data
            config: Experiment configuration (uses default if None)
        """
        self.predictions = predictions
        self.config = config or ABLATION_CONFIG
        self.gt_manager = GroundTruthManager()
        self.evaluator = CaptionEvaluator()
        
        # Results storage
        self.metrics_results = {}
        self.comparison_data = []
        
    def compute_metrics(self):
        """Compute evaluation metrics for all ablation variants."""
        logger.info("Computing evaluation metrics for ablation variants")
        
        metrics_to_compute = self.config.get("metrics", [])
        
        for variant_name, variant_predictions in self.predictions.items():
            logger.info(f"Computing metrics for variant: {variant_name}")
            
            variant_metrics = {
                "variant_name": variant_name,
                "videos": {},
                "average_metrics": {}
            }
            
            all_metrics = {metric: [] for metric in metrics_to_compute}
            
            for video_id, pred_data in variant_predictions.items():
                # Get ground truth
                gt_captions = self.gt_manager.get_ground_truth(video_id)
                
                if not gt_captions:
                    logger.warning(f"No ground truth found for {video_id}, skipping")
                    continue
                
                # Format predictions and ground truth
                pred_text = pred_data.get("formatted_text", "")
                gt_texts = [cap.get("text", "") for cap in gt_captions if cap.get("text")]
                
                if not pred_text or not gt_texts:
                    logger.warning(f"Empty predictions or ground truth for {video_id}, skipping")
                    continue
                
                # Evaluate
                video_metrics = self.evaluator.evaluate([pred_text], gt_texts)
                
                variant_metrics["videos"][video_id] = video_metrics
                
                # Accumulate for average
                for metric in metrics_to_compute:
                    if metric in video_metrics:
                        all_metrics[metric].append(video_metrics[metric])
            
            # Compute averages
            for metric in metrics_to_compute:
                if all_metrics[metric]:
                    variant_metrics["average_metrics"][metric] = sum(all_metrics[metric]) / len(all_metrics[metric])
                else:
                    variant_metrics["average_metrics"][metric] = 0.0
            
            self.metrics_results[variant_name] = variant_metrics
            
            # Add to comparison data
            row = {"variant": variant_name}
            row.update(variant_metrics["average_metrics"])
            self.comparison_data.append(row)
        
        logger.info("Finished computing metrics")
    
    def compare_with_baseline(self) -> Dict[str, Dict[str, float]]:
        """
        Compare each ablation variant with its baseline model.
        
        Returns:
            Dictionary mapping variant_name -> metric -> performance_delta
        """
        logger.info("Comparing ablation variants with baseline models")
        
        comparison_results = {}
        
        for variant_name, variant_metrics in self.metrics_results.items():
            variant_info = ABLATION_VARIANTS.get(variant_name, {})
            base_model = variant_info.get("base_model")
            
            if not base_model:
                logger.warning(f"No base model found for variant {variant_name}")
                continue
            
            # Get baseline metrics (would need to load from comparative analysis results)
            # For now, we'll compute relative to the full model if available
            baseline_name = base_model
            baseline_metrics = self.metrics_results.get(baseline_name, {})
            
            if not baseline_metrics:
                logger.warning(f"Baseline metrics not found for {baseline_name}")
                continue
            
            variant_avg = variant_metrics.get("average_metrics", {})
            baseline_avg = baseline_metrics.get("average_metrics", {})
            
            deltas = {}
            for metric in variant_avg.keys():
                if metric in baseline_avg:
                    delta = variant_avg[metric] - baseline_avg[metric]
                    delta_percent = (delta / baseline_avg[metric] * 100) if baseline_avg[metric] != 0 else 0
                    deltas[metric] = {
                        "absolute": delta,
                        "percent": delta_percent
                    }
            
            comparison_results[variant_name] = {
                "base_model": base_model,
                "deltas": deltas,
                "components_removed": variant_info.get("components_removed", [])
            }
        
        return comparison_results
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table of all ablation variants.
        
        Returns:
            DataFrame with metrics for each variant
        """
        if not self.comparison_data:
            logger.warning("No comparison data available")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.comparison_data)
        
        # Add variant descriptions
        descriptions = []
        for variant in df["variant"]:
            variant_info = ABLATION_VARIANTS.get(variant, {})
            descriptions.append(variant_info.get("description", variant))
        
        df.insert(1, "description", descriptions)
        
        # Sort by variant name
        df = df.sort_values("variant").reset_index(drop=True)
        
        return df
    
    def generate_ablation_report(self) -> str:
        """
        Generate detailed ablation study report.
        
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ABLATION STUDY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total variants tested: {len(self.metrics_results)}")
        report_lines.append(f"Total videos: {sum(len(v['videos']) for v in self.metrics_results.values())}")
        report_lines.append("")
        
        # Comparison table
        report_lines.append("METRICS COMPARISON")
        report_lines.append("-" * 80)
        df = self.generate_comparison_table()
        if not df.empty:
            report_lines.append(df.to_string(index=False))
        report_lines.append("")
        
        # Component contribution analysis
        report_lines.append("COMPONENT CONTRIBUTION ANALYSIS")
        report_lines.append("-" * 80)
        
        comparison_results = self.compare_with_baseline()
        
        for variant_name, comparison in comparison_results.items():
            variant_info = ABLATION_VARIANTS.get(variant_name, {})
            components_removed = variant_info.get("components_removed", [])
            
            report_lines.append(f"\nVariant: {variant_name}")
            report_lines.append(f"  Base Model: {comparison['base_model']}")
            report_lines.append(f"  Components Removed: {', '.join(components_removed)}")
            report_lines.append(f"  Performance Impact:")
            
            deltas = comparison.get("deltas", {})
            for metric, delta_info in deltas.items():
                abs_delta = delta_info["absolute"]
                pct_delta = delta_info["percent"]
                sign = "+" if abs_delta >= 0 else ""
                report_lines.append(f"    {metric}: {sign}{abs_delta:.4f} ({sign}{pct_delta:.2f}%)")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_all_results(self):
        """Save all analysis results to files."""
        summary_dir = Path(self.config.get("summary_dir", ABLATION_SUMMARY_DIR))
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics results
        metrics_file = summary_dir / "ablation_metrics.json"
        try:
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metrics results to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
        
        # Save comparison table
        comparison_file = summary_dir / "ablation_comparison_table.csv"
        try:
            df = self.generate_comparison_table()
            if not df.empty:
                df.to_csv(comparison_file, index=False)
                logger.info(f"Saved comparison table to {comparison_file}")
        except Exception as e:
            logger.error(f"Error saving comparison table: {e}")
        
        # Save markdown comparison table
        comparison_md_file = summary_dir / "ablation_comparison_table.md"
        try:
            df = self.generate_comparison_table()
            if not df.empty:
                df.to_markdown(comparison_md_file, index=False)
                logger.info(f"Saved markdown comparison table to {comparison_md_file}")
        except Exception as e:
            logger.error(f"Error saving markdown table: {e}")
        
        # Save report
        report_file = summary_dir / "ablation_report.txt"
        try:
            report = self.generate_ablation_report()
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Saved ablation report to {report_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
        
        # Save component contribution analysis
        comparison_file = summary_dir / "component_contribution.json"
        try:
            comparison_results = self.compare_with_baseline()
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved component contribution analysis to {comparison_file}")
        except Exception as e:
            logger.error(f"Error saving component contribution: {e}")
