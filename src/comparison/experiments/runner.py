"""
Experiment runner for comparative analysis.

This module provides the infrastructure for running comparative experiments across
multiple video captioning models. It orchestrates model execution, collects predictions,
and stores results for later analysis.

Key Features:
- Runs multiple models on the same dataset
- Handles video path resolution and error recovery
- Saves predictions in structured format
- Tracks experiment progress and statistics
- Provides results for downstream analysis

Example Usage:
    >>> from src.comparison.experiments.runner import ExperimentRunner
    >>> 
    >>> # Initialize runner
    >>> runner = ExperimentRunner()
    >>> 
    >>> # Run all models on test split
    >>> results = runner.run_all_models(split="test")
    >>> 
    >>> # Save results
    >>> runner.save_results()
    >>> 
    >>> # Get predictions for evaluation
    >>> predictions = runner.get_predictions_for_evaluation()
"""

import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm

from src.comparison.model_registry import ModelRegistry, create_model
from src.data.dataset_loader import VideoCaptionDataset, GroundTruthManager
from src.utils.logger import get_logger
from .config import (
    PREDICTIONS_DIR,
    EXPERIMENT_CONFIG,
    get_model_config,
    get_experiment_config
)
import config as project_config

logger = get_logger(__name__)


class ExperimentRunner:
    """
    Runs comparative analysis experiments.
    Executes all models on test videos and collects predictions.
    """
    
    def __init__(self, 
                 dataset: Optional[VideoCaptionDataset] = None,
                 models: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize experiment runner.
        
        Args:
            dataset: VideoCaptionDataset instance (creates new if None)
            models: List of model names to run (uses default if None)
            config: Experiment configuration (uses default if None)
        """
        self.dataset = dataset or VideoCaptionDataset()
        self.config = config or get_experiment_config()
        self.models_to_run = models or self.config.get("models", [])
        self.gt_manager = GroundTruthManager()
        
        # Results storage
        self.results = {
            "experiment_name": self.config.get("experiment_name", "comparative_analysis"),
            "models": {},
            "videos": {},
            "summary": {
                "total_videos": len(self.dataset),
                "total_models": len(self.models_to_run),
                "completed_models": 0,
                "failed_models": 0
            }
        }
        
        logger.info(f"Initialized ExperimentRunner with {len(self.models_to_run)} models")
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
        
        # Combine all caption texts
        texts = [cap.get("text", "").strip() for cap in captions if cap.get("text")]
        return " ".join(texts)
    
    def _save_predictions(self, model_name: str, video_id: str, captions: List[Dict]):
        """
        Save model predictions to file.
        
        Args:
            model_name: Name of the model
            video_id: Video identifier
            captions: List of caption dictionaries
        """
        if not self.config.get("save_predictions", True):
            return
        
        pred_file = PREDICTIONS_DIR / f"{model_name}_{video_id}.json"
        
        data = {
            "model_name": model_name,
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
    
    def run_model_on_video(self, model_name: str, video_info: Dict) -> Optional[List[Dict]]:
        """
        Run a single model on a single video.
        
        Args:
            model_name: Name of the model to run
            video_info: Video information dictionary
            
        Returns:
            List of captions or None if failed
        """
        video_id = video_info.get("video_id")
        video_path = video_info.get("file_path")
        
        # Step 1: Resolve video path - handle both relative and absolute paths
        # Metadata may contain relative paths like "videos/test.mp4" or absolute paths
        video_path_obj = Path(video_path)
        if not video_path_obj.is_absolute():
            # Relative path: join with test_set directory
            relative_path = Path(video_path)
            video_path = project_config.TEST_DATA_DIR / relative_path
        
        video_path = Path(video_path).resolve()  # Resolve any symbolic links
        
        logger.debug(f"Resolved video path: {video_path}")
        logger.debug(f"Path exists: {video_path.exists()}")
        
        # Step 2: If file doesn't exist at resolved path, search in common locations
        # This handles cases where videos are stored in different directory structures
        if not video_path.exists():
            # Try multiple common locations where videos might be stored
            search_locations = [
                project_config.TEST_DATA_DIR / "videos",  # Standard test set location
                project_config.DATA_DIR / "test_set" / "videos",  # Legacy location (fallback)
                project_config.DATA_DIR / video_id / "metadata",   # Per-video metadata folder
                project_config.DATA_DIR / video_id,               # Direct video folder
            ]
            
            for search_dir in search_locations:
                if not search_dir.exists():
                    continue
                
                # Look for video files with common extensions
                for ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm']:
                    # Strategy 1: Try video_id with extension (e.g., "test_001.mp4")
                    potential_file = search_dir / f"{video_id}{ext}"
                    if potential_file.exists():
                        video_path = potential_file
                        logger.info(f"Found video file: {video_path}")
                        break
                    
                    # Strategy 2: Try any video file in directory (fallback)
                    video_files = list(search_dir.glob(f"*{ext}"))
                    if video_files:
                        video_path = video_files[0]  # Take first match
                        logger.info(f"Found video file: {video_path}")
                        break
                
                # If found, stop searching
                if video_path.exists():
                    break
                
                # Strategy 3: Try with original filename from metadata
                original_name = Path(video_info.get("file_path", "")).name
                if original_name:
                    potential_file = search_dir / original_name
                    if potential_file.exists():
                        video_path = potential_file
                        logger.info(f"Found video file: {video_path}")
                        break
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            logger.error(f"Tried to find video for {video_id} in {project_config.TEST_DATA_DIR / 'videos'}")
            return None
        
        logger.info(f"Running {model_name} on {video_id}")
        
        try:
            # Create model instance
            model_config = get_model_config(model_name)
            model = create_model(model_name, config=model_config)
            
            # Generate captions
            start_time = time.time()
            captions = model.generate_captions(str(video_path))
            elapsed_time = time.time() - start_time
            
            logger.info(f"Generated {len(captions)} captions in {elapsed_time:.2f}s")
            
            # Save predictions
            self._save_predictions(model_name, video_id, captions)
            
            # Store in results
            if video_id not in self.results["videos"]:
                self.results["videos"][video_id] = {}
            
            self.results["videos"][video_id][model_name] = {
                "num_captions": len(captions),
                "processing_time": elapsed_time,
                "success": True
            }
            
            return captions
            
        except Exception as e:
            logger.error(f"Error running {model_name} on {video_id}: {e}")
            
            # Store failure in results
            if video_id not in self.results["videos"]:
                self.results["videos"][video_id] = {}
            
            self.results["videos"][video_id][model_name] = {
                "success": False,
                "error": str(e)
            }
            
            return None
    
    def run_all_models(self, split: Optional[str] = None) -> Dict:
        """
        Run all models on all videos in the dataset.
        
        Args:
            split: Dataset split to use ('train', 'test', or None for all)
            
        Returns:
            Results dictionary
        """
        # Get videos to process
        if split:
            videos = self.dataset.get_videos_by_split(split)
            logger.info(f"Processing {len(videos)} videos from '{split}' split")
        else:
            videos = self.dataset.get_all_videos()
            logger.info(f"Processing all {len(videos)} videos")
        
        if not videos:
            logger.warning("No videos found in dataset")
            return self.results
        
        # Calculate total number of tasks for progress tracking
        # Total = number of models × number of videos
        total_tasks = len(self.models_to_run) * len(videos)
        
        # Use tqdm for progress bar visualization
        with tqdm(total=total_tasks, desc="Running experiments") as pbar:
            # Iterate through each model
            for model_name in self.models_to_run:
                logger.info(f"\n{'='*80}")
                logger.info(f"Running model: {model_name}")
                logger.info(f"{'='*80}")
                
                # Track statistics for this model
                model_results = {
                    "videos_processed": 0,
                    "videos_succeeded": 0,
                    "videos_failed": 0
                }
                
                # Run this model on all videos
                for video_info in videos:
                    video_id = video_info.get("video_id")
                    pbar.set_description(f"{model_name} - {video_id}")
                    
                    # Execute model on video and get captions
                    captions = self.run_model_on_video(model_name, video_info)
                    
                    # Update statistics based on success/failure
                    if captions is not None:
                        model_results["videos_succeeded"] += 1
                    else:
                        model_results["videos_failed"] += 1
                    
                    model_results["videos_processed"] += 1
                    pbar.update(1)  # Update progress bar
                
                # Store summary for this model
                self.results["models"][model_name] = model_results
                
                # Update overall experiment summary
                if model_results["videos_succeeded"] > 0:
                    self.results["summary"]["completed_models"] += 1
                else:
                    self.results["summary"]["failed_models"] += 1
        
        logger.info(f"\n{'='*80}")
        logger.info("Experiment completed!")
        logger.info(f"Models completed: {self.results['summary']['completed_models']}")
        logger.info(f"Models failed: {self.results['summary']['failed_models']}")
        logger.info(f"{'='*80}")
        
        return self.results
    
    def save_results(self, output_path: Optional[Path] = None):
        """
        Save experiment results to JSON file.
        
        Args:
            output_path: Path to save results (uses default if None)
        """
        if output_path is None:
            output_path = self.config.get("summary_dir", "results/summary")
            output_path = Path(output_path) / "experiment_results.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved experiment results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def get_predictions_for_evaluation(self) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Get all predictions formatted for evaluation.
        
        Returns:
            Dictionary: {video_id: {model_name: captions_list}}
        """
        predictions = {}
        
        for video_id, video_results in self.results["videos"].items():
            predictions[video_id] = {}
            
            for model_name in self.models_to_run:
                if model_name in video_results and video_results[model_name].get("success"):
                    # Load predictions from file
                    pred_file = PREDICTIONS_DIR / f"{model_name}_{video_id}.json"
                    if pred_file.exists():
                        try:
                            with open(pred_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            predictions[video_id][model_name] = data.get("captions", [])
                        except Exception as e:
                            logger.warning(f"Could not load predictions from {pred_file}: {e}")
        
        return predictions
