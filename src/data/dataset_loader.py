"""
Dataset loader for video captioning test set.
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
import json
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


class VideoCaptionDataset:
    """
    Dataset loader for video captioning evaluation.
    Loads videos and their ground truth captions.
    """
    
    def __init__(self, metadata_path: Optional[Union[str, Path]] = None):
        """
        Initialize dataset loader.
        
        Args:
            metadata_path: Path to dataset metadata JSON file
                          Defaults to tests/test_set/metadata.json
        """
        if metadata_path is None:
            metadata_path = config.TEST_DATA_DIR / "metadata.json"
        
        self.metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata()
        self.videos = self.metadata.get("videos", [])
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata from JSON file."""
        if not self.metadata_path.exists():
            logger.warning(f"Metadata file not found: {self.metadata_path}")
            return {"videos": [], "total_videos": 0}
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded dataset metadata: {metadata.get('total_videos', 0)} videos")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {"videos": [], "total_videos": 0}
    
    def get_video(self, video_id: str) -> Optional[Dict]:
        """
        Get video information by ID.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Video dictionary or None if not found
        """
        for video in self.videos:
            if video.get("video_id") == video_id:
                return video
        return None
    
    def get_videos_by_category(self, category: str) -> List[Dict]:
        """
        Get all videos in a category.
        
        Args:
            category: Category name (lecture, tutorial, conversation, presentation)
            
        Returns:
            List of video dictionaries
        """
        return [v for v in self.videos if v.get("category") == category]
    
    def get_videos_by_split(self, split: str) -> List[Dict]:
        """
        Get all videos in a split (train/test).
        
        Args:
            split: Split name ('train' or 'test')
            
        Returns:
            List of video dictionaries
        """
        return [v for v in self.videos if v.get("split") == split]
    
    def get_all_videos(self) -> List[Dict]:
        """
        Get all videos in the dataset.
        
        Returns:
            List of all video dictionaries
        """
        return self.videos.copy()
    
    def get_categories(self) -> List[str]:
        """
        Get list of all categories in the dataset.
        
        Returns:
            List of category names
        """
        categories = set(v.get("category") for v in self.videos if v.get("category"))
        return sorted(list(categories))
    
    def __len__(self) -> int:
        """Return number of videos in dataset."""
        return len(self.videos)
    
    def __getitem__(self, index: int) -> Dict:
        """Get video by index."""
        return self.videos[index]


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

