"""
Base model interface for video captioning models.
All captioning models should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from pathlib import Path


class BaseCaptionModel(ABC):
    """
    Abstract base class for all video captioning models.
    Defines the common interface that all models must implement.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        self.config = config_dict or {}
        self.model_name = self.__class__.__name__
    
    @abstractmethod
    def generate_captions(self, video_path: str) -> List[Dict]:
        """
        Generate captions for a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of caption dictionaries, each with:
            - 'text': Caption text
            - 'start_time': Start time in seconds
            - 'end_time': End time in seconds
        """
        pass
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process a video and return full results.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with:
            - 'captions': List of caption dictionaries
            - 'metadata': Processing metadata
        """
        captions = self.generate_captions(video_path)
        
        return {
            'captions': captions,
            'metadata': {
                'model_name': self.model_name,
                'video_path': str(video_path),
                'num_captions': len(captions)
            }
        }
    
    def evaluate(self, video_path: str, ground_truth: List[Dict]) -> Dict:
        """
        Evaluate model predictions against ground truth.
        
        Args:
            video_path: Path to video file
            ground_truth: List of ground truth caption dictionaries
            
        Returns:
            Evaluation results dictionary
        """
        from src.evaluation.evaluator import CaptionEvaluator
        
        # Generate predictions
        predictions = self.generate_captions(video_path)
        
        # Extract text from predictions and ground truth
        pred_texts = [cap['text'] for cap in predictions]
        ref_texts = [cap['text'] for cap in ground_truth]
        
        # Evaluate
        evaluator = CaptionEvaluator()
        results = evaluator.evaluate(pred_texts, ref_texts)
        
        return results
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.model_name}(config={self.config})"

