"""
Ground truth management for video captioning evaluation.
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
import json
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


class GroundTruthManager:
    """
    Manager for loading and accessing ground truth captions.
    """
    
    def __init__(self, ground_truth_dir: Optional[Union[str, Path]] = None):
        """
        Initialize ground truth manager.
        
        Args:
            ground_truth_dir: Directory containing ground truth JSON files
                             Defaults to tests/test_set/ground_truth/
        """
        if ground_truth_dir is None:
            ground_truth_dir = config.TEST_DATA_DIR / "ground_truth"
        
        self.ground_truth_dir = Path(ground_truth_dir)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    def load_ground_truth(self, video_id: str) -> Optional[List[Dict]]:
        """
        Load ground truth captions for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of caption dictionaries with 'text', 'start_time', 'end_time'
            or None if not found
        """
        gt_path = self.ground_truth_dir / f"{video_id}.json"
        
        if not gt_path.exists():
            logger.warning(f"Ground truth not found for {video_id}")
            return None
        
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("captions", data.get("segments", []))
            else:
                logger.error(f"Invalid ground truth format for {video_id}")
                return None
        except Exception as e:
            logger.error(f"Error loading ground truth for {video_id}: {e}")
            return None
    
    def save_ground_truth(self, video_id: str, captions: List[Dict]):
        """
        Save ground truth captions for a video.
        
        Args:
            video_id: Video identifier
            captions: List of caption dictionaries
        """
        gt_path = self.ground_truth_dir / f"{video_id}.json"
        
        data = {
            "video_id": video_id,
            "captions": captions,
            "num_captions": len(captions)
        }
        
        try:
            with open(gt_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved ground truth for {video_id}: {len(captions)} captions")
        except Exception as e:
            logger.error(f"Error saving ground truth for {video_id}: {e}")
            raise

