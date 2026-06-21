"""
Visual-only baseline model for video captioning.
Uses only visual information (CLIP embeddings) without audio.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path
from src.comparison.base_model import BaseCaptionModel
from src.visual.visual_embeddings import VisualEmbeddings
from src.visual.frame_extractor import FrameExtractor
from src.generation.caption_generator import CaptionGenerator
from src.preprocessing.video_processor import VideoProcessor
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


class VisualOnlyBaseline(BaseCaptionModel):
    """
    Visual-only baseline that uses CLIP visual embeddings only.
    No audio information is used.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize visual-only baseline.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        super().__init__(config_dict)
        self.visual_embeddings = VisualEmbeddings()
        self.frame_extractor = FrameExtractor()
        
        # Configuration
        self.target_fps = self.config.get("target_fps", config.VIDEO_FPS)
        self.pooling_method = self.config.get("pooling_method", "mean")  # mean, max, attention
        
        # Initialize VideoProcessor with target FPS
        self.video_processor = VideoProcessor(fps=self.target_fps)
        self.caption_generator = CaptionGenerator()
    
    def _pool_visual_embeddings(self, embeddings: np.ndarray, timestamps: List[float], 
                                segment_times: List[tuple]) -> np.ndarray:
        """
        Pool visual embeddings over time segments.
        
        Args:
            embeddings: Visual embeddings array (num_frames, embedding_dim)
            timestamps: Timestamps for each frame
            segment_times: List of (start_time, end_time) tuples for segments
            
        Returns:
            Pooled embeddings array (num_segments, embedding_dim)
        """
        # Validate input
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"embeddings must be numpy array, got {type(embeddings)}")
        
        if len(embeddings) == 0:
            return np.array([])
        
        # Ensure 2D array
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        pooled = []
        for start_time, end_time in segment_times:
            # Find frames within this time segment
            frame_indices = [
                i for i, ts in enumerate(timestamps)
                if start_time <= ts < end_time
            ]
            
            if len(frame_indices) == 0:
                # No frames in segment, use nearest frame
                if timestamps:
                    nearest_idx = min(range(len(timestamps)), 
                                     key=lambda i: abs(timestamps[i] - (start_time + end_time) / 2))
                    frame_indices = [nearest_idx]
                else:
                    # Fallback: use first embedding
                    if len(embeddings) > 0 and isinstance(embeddings, np.ndarray) and len(embeddings.shape) > 1:
                        pooled.append(embeddings[0])
                    else:
                        # If embeddings is empty or invalid, skip this segment
                        logger.warning(f"Invalid embeddings for segment {start_time}-{end_time}, skipping")
                        continue
                    continue
            
            # Pool embeddings in segment
            segment_embeddings = embeddings[frame_indices]
            
            if self.pooling_method == "mean":
                pooled_emb = np.mean(segment_embeddings, axis=0)
            elif self.pooling_method == "max":
                pooled_emb = np.max(segment_embeddings, axis=0)
            elif self.pooling_method == "attention":
                # Simple attention: weighted average (can be enhanced)
                weights = np.ones(len(segment_embeddings)) / len(segment_embeddings)
                pooled_emb = np.average(segment_embeddings, axis=0, weights=weights)
            else:
                pooled_emb = np.mean(segment_embeddings, axis=0)
            
            pooled.append(pooled_emb)
        
        return np.array(pooled)
    
    def _embeddings_to_text_prompt(self, embedding: np.ndarray) -> str:
        """
        Convert visual embedding to text prompt for T5.
        This is a simplified approach - in practice, you might use a learned projection.
        
        Args:
            embedding: Visual embedding vector
            
        Returns:
            Text prompt for caption generation
        """
        # For now, use a generic prompt since we can't directly convert embeddings to text
        # In a full implementation, you might:
        # 1. Use a learned embedding-to-text model
        # 2. Use CLIP's text encoder with visual features
        # 3. Use a visual-language model like BLIP
        
        # Simple approach: Use a generic prompt that indicates visual content
        # The actual caption generation will be limited without text input
        return "caption: visual content"
    
    def generate_captions(self, video_path: str) -> List[Dict]:
        """
        Generate captions using only visual information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of caption dictionaries with 'text', 'start_time', 'end_time'
        """
        logger.info(f"Generating captions (visual-only) for {video_path}")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Extract frames (FPS is set during VideoProcessor initialization)
            frames_info = self.video_processor.extract_frames(str(video_path))
            
            if not frames_info:
                logger.warning("No frames extracted from video")
                return []
            
            # Extract frame arrays and timestamps
            # frames_info is a list of tuples: (frame_path, timestamp)
            frames = []
            timestamps = []
            import cv2
            for frame_path_str, timestamp in frames_info:
                frame_path = Path(frame_path_str)
                if frame_path.exists():
                    try:
                        # Load frame from file path
                        frame_bgr = cv2.imread(str(frame_path))
                        if frame_bgr is None:
                            logger.warning(f"Could not load frame: {frame_path_str}, skipping")
                            continue
                        
                        # Convert BGR to RGB (OpenCV loads as BGR)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        
                        # Preprocess frame (resize and normalize)
                        frame_array = self.frame_extractor.preprocess_frame(frame_rgb)
                        
                        # Validate frame_array is a numpy array
                        if not isinstance(frame_array, np.ndarray):
                            logger.warning(f"preprocess_frame returned non-array type {type(frame_array)} for {frame_path_str}, skipping")
                            continue
                        frames.append(frame_array)
                        timestamps.append(timestamp)
                    except Exception as e:
                        logger.warning(f"Error processing frame {frame_path_str}: {e}, skipping")
                        continue
            
            if not frames:
                logger.warning("No valid frames found")
                return []
            
            # Extract visual embeddings
            logger.info(f"Extracting visual embeddings from {len(frames)} frames")
            visual_embeddings = self.visual_embeddings.extract_embeddings(frames)
            
            # Validate visual embeddings
            if not isinstance(visual_embeddings, np.ndarray):
                logger.error(f"Visual embeddings is not a numpy array (type: {type(visual_embeddings)}), got: {str(visual_embeddings)[:100]}")
                return []
            
            if len(visual_embeddings) == 0:
                logger.warning("No visual embeddings extracted")
                return []
            
            logger.debug(f"Visual embeddings shape: {visual_embeddings.shape}")
            
            # Create time segments (use fixed duration segments)
            # In practice, you might want to align with audio segments or use sliding windows
            video_duration = timestamps[-1] if timestamps else 30.0
            segment_duration = 3.0  # 3-second segments
            num_segments = int(video_duration / segment_duration) + 1
            
            segment_times = [
                (i * segment_duration, min((i + 1) * segment_duration, video_duration))
                for i in range(num_segments)
            ]
            
            # Pool embeddings over segments
            pooled_embeddings = self._pool_visual_embeddings(
                visual_embeddings, timestamps, segment_times
            )
            
            # Generate captions
            # Note: Since we only have visual embeddings and T5 needs text input,
            # we'll use a simplified approach: generate generic captions based on visual content
            # For a full implementation, you'd want to use a vision-language model
            
            captions = []
            for i, (start_time, end_time) in enumerate(segment_times):
                if i < len(pooled_embeddings):
                    # For now, create placeholder captions
                    # In a full implementation, you'd use a vision-language model
                    # or learn a mapping from embeddings to text
                    caption_text = f"[Visual content from {start_time:.1f}s to {end_time:.1f}s]"
                    
                    captions.append({
                        "text": caption_text,
                        "start_time": start_time,
                        "end_time": end_time
                    })
            
            logger.info(f"Generated {len(captions)} visual-only captions")
            return captions
            
        except Exception as e:
            logger.error(f"Error generating visual-only captions: {e}")
            raise
