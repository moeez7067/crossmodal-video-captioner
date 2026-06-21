"""
Ablation study runner.
Executes ablation experiments by running ablated model variants.
"""

import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm

from src.comparison.ablation.config import (
    ABLATION_PREDICTIONS_DIR,
    ABLATION_CONFIG,
    get_ablation_variants,
    get_all_ablation_model_names
)
from src.comparison.ablation.ablation_models import create_ablated_model
from src.data.dataset_loader import VideoCaptionDataset, GroundTruthManager
from src.utils.logger import get_logger
import config as project_config

logger = get_logger(__name__)


class AblationRunner:
    """
    Runs ablation study experiments.
    Executes ablated model variants on test videos and collects predictions.
    """
    
    def __init__(
        self,
        dataset: Optional[VideoCaptionDataset] = None,
        variants: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ablation runner.
        
        Args:
            dataset: VideoCaptionDataset instance (creates new if None)
            variants: List of ablation variant names to run (uses all if None)
            config: Experiment configuration (uses default if None)
        """
        self.dataset = dataset or VideoCaptionDataset()
        self.config = config or ABLATION_CONFIG
        self.variants_to_run = variants or get_all_ablation_model_names()
        self.gt_manager = GroundTruthManager()
        
        # Results storage
        self.results = {
            "experiment_name": self.config.get("experiment_name", "ablation_study"),
            "variants": {},
            "videos": {},
            "summary": {
                "total_videos": len(self.dataset),
                "total_variants": len(self.variants_to_run),
                "completed_variants": 0,
                "failed_variants": 0
            }
        }
        
        logger.info(f"Initialized AblationRunner with {len(self.variants_to_run)} variants")
        logger.info(f"Dataset contains {len(self.dataset)} videos")
    
    def _format_captions_for_evaluation(self, captions: List[Dict]) -> str:
        """
        Format captions list into a single string for evaluation.
        
        Args:
            captions: List of caption dictionaries with 'text', 'start_time', 'end_time'
            
        Returns:
            Combined caption text
        """
        if not captions:
            return ""
        
        texts = [cap.get("text", "").strip() for cap in captions if cap.get("text")]
        return " ".join(texts)
    
    def _save_predictions(self, variant_name: str, video_id: str, captions: List[Dict]):
        """
        Save model predictions to file.
        
        Args:
            variant_name: Name of the ablation variant
            video_id: Video identifier
            captions: List of caption dictionaries
        """
        if not self.config.get("save_predictions", True):
            return
        
        pred_file = ABLATION_PREDICTIONS_DIR / f"{variant_name}_{video_id}.json"
        
        data = {
            "variant_name": variant_name,
            "video_id": video_id,
            "captions": captions,
            "num_captions": len(captions),
            "formatted_text": self._format_captions_for_evaluation(captions)
        }
        
        try:
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved predictions: {pred_file}")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def run_variant_on_video(self, variant_name: str, video_info: Dict) -> Optional[List[Dict]]:
        """
        Run a single ablation variant on a single video.
        
        Args:
            variant_name: Name of the ablation variant
            video_info: Video information dictionary
            
        Returns:
            List of captions or None if failed
        """
        video_id = video_info.get("video_id")
        video_path = video_info.get("file_path")
        
        # Resolve full path
        video_path_obj = Path(video_path)
        if not video_path_obj.is_absolute():
            relative_path = Path(video_path)
            video_path = project_config.TEST_DATA_DIR / relative_path
        
        video_path = Path(video_path).resolve()
        
        # Try to find video if path doesn't exist
        if not video_path.exists():
            search_locations = [
                project_config.TEST_DATA_DIR / "videos",  # Standard test set location
                project_config.DATA_DIR / "test_set" / "videos",  # Legacy location (fallback)
                project_config.DATA_DIR / video_id / "metadata",
                project_config.DATA_DIR / video_id,
            ]
            
            for search_dir in search_locations:
                if not search_dir.exists():
                    continue
                
                for ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm']:
                    potential_file = search_dir / f"{video_id}{ext}"
                    if potential_file.exists():
                        video_path = potential_file
                        break
                    
                    video_files = list(search_dir.glob(f"*{ext}"))
                    if video_files:
                        video_path = video_files[0]
                        break
                
                if video_path.exists():
                    break
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        logger.info(f"Running {variant_name} on {video_id}")
        
        try:
            # Create ablated model instance
            model = create_ablated_model(variant_name)
            
            # Generate captions
            start_time = time.time()
            captions = model.generate_captions(str(video_path))
            elapsed_time = time.time() - start_time
            
            logger.info(f"Generated {len(captions)} captions in {elapsed_time:.2f}s")
            
            # Save predictions
            self._save_predictions(variant_name, video_id, captions)
            
            # Store results
            if variant_name not in self.results["variants"]:
                self.results["variants"][variant_name] = {
                    "videos": {},
                    "total_time": 0.0,
                    "total_captions": 0
                }
            
            self.results["variants"][variant_name]["videos"][video_id] = {
                "num_captions": len(captions),
                "elapsed_time": elapsed_time,
                "success": True
            }
            self.results["variants"][variant_name]["total_time"] += elapsed_time
            self.results["variants"][variant_name]["total_captions"] += len(captions)
            
            if video_id not in self.results["videos"]:
                self.results["videos"][video_id] = {}
            
            self.results["videos"][video_id][variant_name] = {
                "num_captions": len(captions),
                "success": True
            }
            
            return captions
            
        except Exception as e:
            logger.error(f"Error running {variant_name} on {video_id}: {e}", exc_info=True)
            
            # Store failure
            if variant_name not in self.results["variants"]:
                self.results["variants"][variant_name] = {
                    "videos": {},
                    "total_time": 0.0,
                    "total_captions": 0,
                    "failures": 0
                }
            
            self.results["variants"][variant_name]["videos"][video_id] = {
                "success": False,
                "error": str(e)
            }
            self.results["variants"][variant_name]["failures"] = \
                self.results["variants"][variant_name].get("failures", 0) + 1
            
            return None
    
    def run_all_variants(self, split: Optional[str] = "test") -> Dict[str, Any]:
        """
        Run all ablation variants on all videos in the dataset.
        
        Args:
            split: Dataset split to use ("test", "train", "val", or None for all)
            
        Returns:
            Results dictionary
        """
        logger.info("Starting ablation study experiments")
        logger.info(f"Running {len(self.variants_to_run)} variants on {len(self.dataset)} videos")
        
        # Get videos for the specified split
        if split:
            videos = [v for v in self.dataset if v.get("split") == split]
        else:
            videos = list(self.dataset)
        
        if not videos:
            logger.warning(f"No videos found for split: {split}")
            return self.results
        
        logger.info(f"Processing {len(videos)} videos")
        
        # Run each variant on each video
        total_runs = len(self.variants_to_run) * len(videos)
        with tqdm(total=total_runs, desc="Running ablation variants") as pbar:
            for variant_name in self.variants_to_run:
                logger.info(f"Running variant: {variant_name}")
                
                for video_info in videos:
                    self.run_variant_on_video(variant_name, video_info)
                    pbar.update(1)
                
                # Update summary
                variant_results = self.results["variants"].get(variant_name, {})
                if variant_results.get("videos"):
                    self.results["summary"]["completed_variants"] += 1
                else:
                    self.results["summary"]["failed_variants"] += 1
        
        logger.info("Ablation study experiments completed")
        return self.results
    
    def get_predictions_for_evaluation(self) -> Dict[str, Dict[str, Dict]]:
        """
        Get predictions in format suitable for evaluation.
        
        Returns:
            Dictionary mapping variant_name -> video_id -> prediction data
        """
        predictions = {}
        
        for variant_name in self.variants_to_run:
            predictions[variant_name] = {}
            
            for video_id, video_data in self.results.get("videos", {}).items():
                if variant_name in video_data and video_data[variant_name].get("success"):
                    # Load predictions from file
                    pred_file = ABLATION_PREDICTIONS_DIR / f"{variant_name}_{video_id}.json"
                    if pred_file.exists():
                        try:
                            with open(pred_file, 'r', encoding='utf-8') as f:
                                pred_data = json.load(f)
                                predictions[variant_name][video_id] = pred_data
                        except Exception as e:
                            logger.warning(f"Error loading predictions from {pred_file}: {e}")
        
        return predictions
    
    def save_results(self):
        """Save experiment results to file."""
        results_file = Path(self.config.get("summary_dir", ABLATION_PREDICTIONS_DIR.parent / "summary")) / "ablation_results.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved ablation results to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
