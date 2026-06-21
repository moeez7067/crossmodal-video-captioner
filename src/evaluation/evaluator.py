"""
Evaluation pipeline for video captioning models.
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
import json
from .metrics import MetricsCalculator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CaptionEvaluator:
    """
    Comprehensive caption evaluator that runs all metrics on predictions vs references.
    """
    
    def __init__(self):
        """Initialize evaluator with metrics calculator."""
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate(self, predictions: Union[str, List[str]], 
                 references: Union[str, List[str], List[List[str]]],
                 video_ids: Optional[List[str]] = None) -> Dict:
        """
        Evaluate predictions against references using all metrics.
        
        Args:
            predictions: Single prediction string or list of predictions
            references: Single reference string, list of references, or list of reference lists
            video_ids: Optional list of video IDs for tracking
            
        Returns:
            Dictionary with:
            - 'overall': Average scores across all videos
            - 'per_video': Scores for each video (if video_ids provided)
            - 'metrics': List of all computed metrics
        """
        # Normalize inputs
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references[0], str):
            references = [[ref] for ref in references]
        
        # Compute overall metrics
        overall_scores = self.metrics_calculator.compute_all(predictions, references)
        
        # Compute per-video metrics if video_ids provided
        per_video_scores = {}
        if video_ids and len(video_ids) == len(predictions):
            for vid_id, pred, refs in zip(video_ids, predictions, references):
                per_video_scores[vid_id] = self.metrics_calculator.compute_all([pred], [refs])
        
        result = {
            'overall': overall_scores,
            'metrics': list(overall_scores.keys())
        }
        
        if per_video_scores:
            result['per_video'] = per_video_scores
        
        return result
    
    def evaluate_batch(self, predictions: List[str], 
                      references: List[Union[str, List[str]]],
                      video_ids: Optional[List[str]] = None) -> Dict:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of prediction strings
            references: List of reference strings or list of reference lists
            video_ids: Optional list of video IDs
            
        Returns:
            Evaluation results dictionary
        """
        return self.evaluate(predictions, references, video_ids)
    
    def generate_report(self, evaluation_results: Dict, 
                       output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluate() method
            output_path: Optional path to save report JSON
            
        Returns:
            Formatted report string
        """
        overall = evaluation_results.get('overall', {})
        per_video = evaluation_results.get('per_video', {})
        
        # Create report
        report_lines = [
            "=" * 80,
            "VIDEO CAPTIONING EVALUATION REPORT",
            "=" * 80,
            "",
            "OVERALL METRICS:",
            "-" * 80,
        ]
        
        # Format overall scores
        for metric, score in sorted(overall.items()):
            report_lines.append(f"  {metric:25s}: {score:.4f}")
        
        # Per-video breakdown
        if per_video:
            report_lines.extend([
                "",
                "PER-VIDEO METRICS:",
                "-" * 80,
            ])
            
            for vid_id, scores in per_video.items():
                report_lines.append(f"\n  Video: {vid_id}")
                for metric, score in sorted(scores.items()):
                    report_lines.append(f"    {metric:25s}: {score:.4f}")
        
        report_lines.extend([
            "",
            "=" * 80,
        ])
        
        report = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save both text and JSON
            output_path.with_suffix('.txt').write_text(report, encoding='utf-8')
            output_path.with_suffix('.json').write_text(
                json.dumps(evaluation_results, indent=2), 
                encoding='utf-8'
            )
            
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report

