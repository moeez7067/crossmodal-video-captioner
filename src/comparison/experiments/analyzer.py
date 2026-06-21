"""
Results analyzer for comparative analysis.

This module provides comprehensive analysis of experiment results, including:
- Metric computation across all models and videos
- Comparison table generation
- Visualization creation
- Report generation

The analyzer computes standard captioning metrics (BLEU, CIDEr, METEOR, ROUGE-L)
and generates visualizations and reports to compare model performance.

Example Usage:
    >>> from src.comparison.experiments.analyzer import ResultsAnalyzer
    >>> 
    >>> # Initialize with predictions
    >>> predictions = {
    ...     "video_001": {
    ...         "gated": [...captions...],
    ...         "concatenation": [...captions...]
    ...     }
    ... }
    >>> analyzer = ResultsAnalyzer(predictions)
    >>> 
    >>> # Compute metrics
    >>> metrics = analyzer.compute_metrics()
    >>> 
    >>> # Generate comparison table
    >>> table = analyzer.generate_comparison_table()
    >>> 
    >>> # Generate report
    >>> report = analyzer.generate_report()
"""

import json
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.evaluator import CaptionEvaluator
from src.data.dataset_loader import GroundTruthManager
from src.utils.logger import get_logger
from .config import (
    SUMMARY_DIR,
    VISUALIZATIONS_DIR,
    EXPERIMENT_CONFIG,
    EVALUATION_METRICS
)
import config as project_config

logger = get_logger(__name__)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('dark_background')
sns.set_palette("husl")


class ResultsAnalyzer:
    """
    Analyzes experiment results and generates comparison reports.
    
    This class provides comprehensive analysis capabilities for comparative experiments:
    - Computes evaluation metrics (BLEU, CIDEr, METEOR, ROUGE-L) for all models
    - Aggregates results across videos
    - Generates comparison tables and visualizations
    - Creates detailed analysis reports
    
    The analyzer works with predictions in the format:
        {
            "video_id": {
                "model_name": [list of caption dictionaries]
            }
        }
    
    Each caption dictionary should have: 'text', 'start_time', 'end_time'
    """
    
    def __init__(self, 
                 predictions: Dict[str, Dict[str, List[Dict]]],
                 ground_truth_manager: Optional[GroundTruthManager] = None):
        """
        Initialize results analyzer.
        
        Args:
            predictions: Nested dictionary structure:
                {
                    "video_id": {
                        "model_name": [
                            {"text": "...", "start_time": 0.0, "end_time": 5.0},
                            ...
                        ]
                    }
                }
            ground_truth_manager: Optional GroundTruthManager instance.
                If None, creates a new instance.
        
        Note:
            Ground truth captions are loaded automatically when computing metrics.
        """
        self.predictions = predictions
        self.gt_manager = ground_truth_manager or GroundTruthManager()
        self.evaluator = CaptionEvaluator()
        
        # Results storage
        self.metrics_results = {}
        self.comparison_table = None
        
        logger.info("Initialized ResultsAnalyzer")
    
    def _format_captions_text(self, captions: List[Dict]) -> str:
        """
        Format captions list into single text string for evaluation.
        
        Combines all caption texts into a single string, which is required
        for most evaluation metrics (BLEU, CIDEr, etc.).
        
        Args:
            captions: List of caption dictionaries with 'text' field
            
        Returns:
            Combined text string with all captions joined by spaces
        """
        if not captions:
            return ""
        # Extract text from each caption and join with spaces
        texts = [cap.get("text", "").strip() for cap in captions if cap.get("text")]
        return " ".join(texts)
    
    def _format_ground_truth_text(self, gt_captions: List[Dict]) -> str:
        """
        Format ground truth captions into single text string.
        
        Similar to _format_captions_text but for ground truth data.
        Ensures consistent formatting for evaluation.
        
        Args:
            gt_captions: List of ground truth caption dictionaries
            
        Returns:
            Combined text string with all ground truth captions
        """
        if not gt_captions:
            return ""
        texts = [cap.get("text", "").strip() for cap in gt_captions if cap.get("text")]
        return " ".join(texts)
    
    def compute_metrics(self) -> Dict:
        """
        Compute evaluation metrics for all models on all videos.
        
        Iterates through all videos and models, computes standard captioning metrics
        (BLEU-1/2/3/4, CIDEr, METEOR, ROUGE-L) by comparing predictions against
        ground truth captions.
        
        Uses segment-by-segment evaluation for more accurate metrics, comparing
        each predicted caption with temporally aligned ground truth captions.
        
        Returns:
            Nested dictionary structure:
            {
                "video_id": {
                    "model_name": {
                        "bleu_1": 0.45,
                        "bleu_2": 0.32,
                        "cider": 0.67,
                        ...
                    }
                }
            }
        
        Note:
            Videos without ground truth are skipped with a warning.
            Models with empty predictions are skipped with a warning.
        """
        logger.info("Computing evaluation metrics...")
        
        all_results = {}
        
        # Iterate through each video in the dataset
        for video_id, model_predictions in self.predictions.items():
            # Load ground truth captions for this video
            gt_captions = self.gt_manager.load_ground_truth(video_id)
            if not gt_captions:
                logger.warning(f"No ground truth found for {video_id}, skipping")
                continue
            
            video_results = {}
            
            # Evaluate each model's predictions on this video
            for model_name, pred_captions in model_predictions.items():
                if not pred_captions:
                    logger.warning(f"Empty predictions for {model_name} on {video_id}")
                    continue
                
                # Use segment-by-segment evaluation for more accurate metrics
                try:
                    eval_results = self._evaluate_segments(pred_captions, gt_captions, video_id, model_name)
                    video_results[model_name] = eval_results
                    logger.debug(f"Computed metrics for {model_name} on {video_id}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {video_id}: {e}", exc_info=True)
                    continue
            
            # Only add video results if at least one model was successfully evaluated
            if video_results:
                all_results[video_id] = video_results
        
        self.metrics_results = all_results
        logger.info(f"Computed metrics for {len(all_results)} videos")
        
        return all_results
    
    def _evaluate_segments(self, pred_captions: List[Dict], gt_captions: List[Dict], 
                         video_id: str, model_name: str) -> Dict:
        """
        Evaluate predictions segment-by-segment against ground truth.
        
        This method aligns predicted captions with ground truth captions temporally
        and evaluates each segment individually, then aggregates the results.
        This provides more accurate metrics than concatenating all text.
        
        Args:
            pred_captions: List of predicted caption dictionaries
            gt_captions: List of ground truth caption dictionaries
            video_id: Video ID for logging
            model_name: Model name for logging
            
        Returns:
            Dictionary of metric scores
        """
        # Extract texts for segment-by-segment comparison
        pred_texts = []
        gt_texts = []
        
        # Align predictions with ground truth by temporal overlap
        for pred_cap in pred_captions:
            pred_start = pred_cap.get("start_time", 0.0)
            pred_end = pred_cap.get("end_time", pred_start + 1.0)
            pred_text = pred_cap.get("text", "").strip()
            
            if not pred_text:
                continue
            
            # Find ground truth captions that overlap with this prediction
            overlapping_gt = []
            for gt_cap in gt_captions:
                gt_start = gt_cap.get("start_time", 0.0)
                gt_end = gt_cap.get("end_time", gt_start + 1.0)
                
                # Check for temporal overlap
                if not (pred_end < gt_start or pred_start > gt_end):
                    overlapping_gt.append(gt_cap.get("text", "").strip())
            
            # If no overlap, use nearest ground truth caption
            if not overlapping_gt:
                # Find nearest GT caption by start time
                nearest_idx = min(
                    range(len(gt_captions)),
                    key=lambda i: abs(gt_captions[i].get("start_time", 0.0) - pred_start)
                )
                overlapping_gt = [gt_captions[nearest_idx].get("text", "").strip()]
            
            pred_texts.append(pred_text)
            gt_texts.append(overlapping_gt)  # List of reference texts for this segment
        
        # If no valid segments, return zero scores
        if not pred_texts:
            logger.warning(f"No valid segments for {model_name} on {video_id}")
            return {metric: 0.0 for metric in EVALUATION_METRICS}
        
        # Check for exact match (which would indicate a problem)
        exact_matches = sum(1 for p, gts in zip(pred_texts, gt_texts) 
                          if p in gts or any(p.strip().lower() == gt.strip().lower() for gt in gts))
        match_ratio = exact_matches / len(pred_texts) if pred_texts else 0.0
        
        # Also check if concatenated text matches exactly
        pred_full = " ".join(pred_texts).strip().lower()
        gt_full = " ".join([gt[0] if gt else "" for gt in gt_texts]).strip().lower()
        full_match = pred_full == gt_full
        
        if match_ratio > 0.9 or full_match:
            logger.warning(
                f"⚠️  SUSPICIOUS EVALUATION for {model_name} on {video_id}:\n"
                f"   - Exact segment match ratio: {match_ratio:.2%}\n"
                f"   - Full text exact match: {full_match}\n"
                f"   - This suggests ground truth may be the same transcript used by the model.\n"
                f"   - Ground truth should be human-written captions, not raw transcripts.\n"
                f"   - Sample prediction: {pred_texts[0][:80] if pred_texts else 'N/A'}...\n"
                f"   - Sample ground truth: {gt_texts[0][0][:80] if gt_texts and gt_texts[0] else 'N/A'}..."
            )
        
        # Log sample comparison for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Evaluating {model_name} on {video_id}:")
            logger.debug(f"  Predictions: {len(pred_texts)} segments")
            logger.debug(f"  Sample pred: {pred_texts[0][:100] if pred_texts else 'N/A'}")
            logger.debug(f"  Sample GT: {gt_texts[0][0][:100] if gt_texts and gt_texts[0] else 'N/A'}")
            logger.debug(f"  Exact match ratio: {match_ratio:.2%}")
        
        # Compute metrics using segment-by-segment evaluation
        try:
            eval_results = self.evaluator.evaluate(
                predictions=pred_texts,
                references=gt_texts,  # List of reference lists (one per segment)
                video_ids=[video_id] * len(pred_texts)
            )
            
            overall = eval_results.get("overall", {})
            
            # Log if scores are suspiciously high
            if overall.get("bleu_1", 0.0) > 0.95:
                logger.warning(
                    f"Suspiciously high BLEU-1 score ({overall.get('bleu_1', 0.0):.4f}) for {model_name} on {video_id}. "
                    f"This may indicate predictions match ground truth too closely."
                )
            
            return overall
            
        except Exception as e:
            logger.warning(f"Segment-by-segment evaluation failed for {model_name} on {video_id}: {e}. "
                         f"Falling back to concatenated text evaluation.")
            # Fallback to old method
            pred_text = self._format_captions_text(pred_captions)
            gt_text = self._format_ground_truth_text(gt_captions)
            
            eval_results = self.evaluator.evaluate(
                predictions=[pred_text],
                references=[[gt_text]],
                video_ids=[video_id]
            )
            
            return eval_results.get("overall", {})
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table of all models across all metrics.
        
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        if not self.metrics_results:
            logger.warning("No metrics computed yet. Run compute_metrics() first.")
            return pd.DataFrame()
        
        # Step 1: Aggregate metrics across all videos for each model
        # Collect all metric values for each model across all videos
        model_metrics = {}
        
        for video_id, video_results in self.metrics_results.items():
            for model_name, metrics in video_results.items():
                # Initialize metric lists for this model if not seen before
                if model_name not in model_metrics:
                    model_metrics[model_name] = {metric: [] for metric in EVALUATION_METRICS}
                
                # Collect metric values for averaging
                for metric in EVALUATION_METRICS:
                    if metric in metrics:
                        model_metrics[model_name][metric].append(metrics[metric])
        
        # Step 2: Compute average metrics for each model
        # Average across all videos to get overall model performance
        comparison_data = {}
        for model_name, metrics_dict in model_metrics.items():
            comparison_data[model_name] = {}
            for metric, values in metrics_dict.items():
                if values:
                    # Compute mean across all videos
                    comparison_data[model_name][metric] = np.mean(values)
                else:
                    comparison_data[model_name][metric] = 0.0
        
        # Create DataFrame
        self.comparison_table = pd.DataFrame(comparison_data).T
        self.comparison_table = self.comparison_table.round(4)
        
        logger.info("Generated comparison table")
        return self.comparison_table
    
    def save_comparison_table(self, output_path: Optional[Path] = None, 
                            formats: List[str] = ["csv", "md"]):
        """
        Save comparison table to file.
        
        Args:
            output_path: Output file path (without extension)
            formats: List of formats to save ('csv', 'md', 'html', 'latex')
        """
        if self.comparison_table is None or self.comparison_table.empty:
            logger.warning("No comparison table to save")
            return
        
        if output_path is None:
            output_path = SUMMARY_DIR / "comparison_table"
        
        output_path = Path(output_path)
        
        for fmt in formats:
            if fmt == "csv":
                file_path = output_path.with_suffix(".csv")
                self.comparison_table.to_csv(file_path)
                logger.info(f"Saved comparison table (CSV): {file_path}")
            
            elif fmt == "md":
                file_path = output_path.with_suffix(".md")
                md_table = self.comparison_table.to_markdown()
                file_path.write_text(md_table, encoding='utf-8')
                logger.info(f"Saved comparison table (Markdown): {file_path}")
            
            elif fmt == "html":
                file_path = output_path.with_suffix(".html")
                html_table = self.comparison_table.to_html()
                file_path.write_text(html_table, encoding='utf-8')
                logger.info(f"Saved comparison table (HTML): {file_path}")
            
            elif fmt == "latex":
                file_path = output_path.with_suffix(".tex")
                latex_table = self.comparison_table.to_latex()
                file_path.write_text(latex_table, encoding='utf-8')
                logger.info(f"Saved comparison table (LaTeX): {file_path}")
    
    def generate_visualizations(self):
        """Generate visualization plots for comparison."""
        if self.comparison_table is None or self.comparison_table.empty:
            logger.warning("No comparison table to visualize")
            return
        
        logger.info("Generating visualizations...")
        
        # 1. Bar chart for each metric
        for metric in EVALUATION_METRICS:
            if metric not in self.comparison_table.columns:
                continue
            
            plt.figure(figsize=(10, 6))
            self.comparison_table[metric].plot(kind='bar', color='steelblue')
            plt.title(f'{metric.upper()} Score by Model', fontsize=14, fontweight='bold')
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            output_file = VISUALIZATIONS_DIR / f"{metric}_comparison.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization: {output_file}")
        
        # 2. Heatmap of all metrics
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.comparison_table.T, annot=True, fmt='.4f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Score'})
        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        plt.tight_layout()
        
        output_file = VISUALIZATIONS_DIR / "performance_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {output_file}")
        
        # 3. Combined bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        self.comparison_table.plot(kind='bar', ax=ax, width=0.8)
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_file = VISUALIZATIONS_DIR / "all_metrics_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {output_file}")
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Report text
        """
        if self.comparison_table is None or self.comparison_table.empty:
            logger.warning("No comparison table available for report")
            return ""
        
        report_lines = [
            "=" * 80,
            "COMPARATIVE ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY",
            "-" * 80,
            f"Number of models compared: {len(self.comparison_table)}",
            f"Number of metrics: {len(self.comparison_table.columns)}",
            f"Number of videos: {len(self.metrics_results)}",
            "",
            "COMPARISON TABLE",
            "-" * 80,
            "",
            self.comparison_table.to_string(),
            "",
            "BEST PERFORMING MODELS",
            "-" * 80,
        ]
        
        # Find best model for each metric
        for metric in EVALUATION_METRICS:
            if metric in self.comparison_table.columns:
                best_model = self.comparison_table[metric].idxmax()
                best_score = self.comparison_table[metric].max()
                report_lines.append(f"  {metric:20s}: {best_model:20s} ({best_score:.4f})")
        
        report_lines.extend([
            "",
            "=" * 80,
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding='utf-8')
            logger.info(f"Saved report to {output_path}")
        
        return report
    
    def save_all_results(self):
        """Save all analysis results."""
        # Save metrics results
        metrics_file = SUMMARY_DIR / "metrics_results.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metrics results: {metrics_file}")
        
        # Save comparison table
        self.save_comparison_table()
        
        # Generate visualizations
        if EXPERIMENT_CONFIG.get("generate_visualizations", True):
            self.generate_visualizations()
        
        # Generate report
        if EXPERIMENT_CONFIG.get("generate_report", True):
            report_file = SUMMARY_DIR / "analysis_report.txt"
            self.generate_report(report_file)
