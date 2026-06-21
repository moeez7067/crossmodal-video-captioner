"""
Video preprocessing module for extracting frames and metadata from video files.
"""

import cv2
import os
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import config
from src.utils.logger import get_logger
from src.utils.file_utils import ensure_directory, get_video_output_dir, get_video_frames_dir, get_video_metadata_dir, save_json

logger = get_logger(__name__)


class VideoProcessor:
    """Processes video files to extract frames and metadata."""
    
    def __init__(self, fps: Optional[float] = None):
        """
        Initialize video processor.
        
        Args:
            fps: Frames per second to extract (default: from config)
        """
        self.fps = fps if fps is not None else config.VIDEO_FPS
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Extract frames from video at specified FPS.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save frames (if None, uses video directory)
            
        Returns:
            List of tuples (frame_path, timestamp) for each extracted frame
        """
        if not self.validate_video(video_path):
            raise ValueError(f"Invalid video file: {video_path}")
        
        video_path_obj = Path(video_path)
        
        # Set output directory - use organized structure
        if output_dir is None:
            output_dir = get_video_frames_dir(video_path)
        else:
            output_dir = Path(output_dir)
            ensure_directory(output_dir)
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        
        try:
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0
            
            # Calculate frame extraction interval
            frame_interval = int(video_fps / self.fps) if self.fps > 0 else 1
            if frame_interval < 1:
                frame_interval = 1
            
            logger.info(f"Extracting frames from {video_path}")
            logger.info(f"Video FPS: {video_fps}, Target FPS: {self.fps}, Interval: {frame_interval}")
            
            extracted_frames = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / video_fps if video_fps > 0 else 0
                    
                    # Save frame
                    frame_filename = f"frame_{extracted_count:06d}_{timestamp:.3f}s.jpg"
                    frame_path = output_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append((str(frame_path), timestamp))
                    extracted_count += 1
                
                frame_count += 1
            
            logger.info(f"Extracted {extracted_count} frames from {total_frames} total frames")
            return extracted_frames
            
        finally:
            cap.release()
    
    def get_video_metadata(self, video_path: str) -> Dict:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video metadata (duration, fps, resolution, etc.)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # Get file size
            file_size = os.path.getsize(video_path)
            
            metadata = {
                "file_path": video_path,
                "file_name": Path(video_path).name,
                "file_size": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "duration": duration,
                "duration_formatted": self._format_duration(duration),
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "codec": codec,
                "aspect_ratio": width / height if height > 0 else 0,
            }
            
            return metadata
            
        finally:
            cap.release()
    
    def validate_video(self, video_path: str) -> bool:
        """
        Validate video file format and accessibility.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video is valid, False otherwise
        """
        # Check if file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return False
        
        # Check if it's a file (not directory)
        if not os.path.isfile(video_path):
            logger.error(f"Path is not a file: {video_path}")
            return False
        
        # Check file extension
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in config.SUPPORTED_VIDEO_FORMATS:
            logger.error(f"Unsupported video format: {file_ext}")
            return False
        
        # Try to open video with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return False
        
        # Check if we can read at least one frame
        ret, _ = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"Could not read frames from video: {video_path}")
            return False
        
        return True
    
    def save_metadata_to_json(self, video_path: str, metadata: Optional[Dict] = None) -> Path:
        """
        Save video metadata to JSON file.
        
        Args:
            video_path: Path to video file
            metadata: Metadata dictionary (if None, will extract it)
            
        Returns:
            Path to saved JSON file
        """
        if metadata is None:
            metadata = self.get_video_metadata(video_path)
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "video_metadata.json"
        
        save_json(metadata, str(json_path))
        logger.info(f"Saved video metadata to {json_path}")
        return json_path
    
    def save_frames_info_to_json(self, video_path: str, frames: List[Tuple[str, float]]) -> Path:
        """
        Save frame extraction information to JSON file.
        
        Args:
            video_path: Path to video file
            frames: List of (frame_path, timestamp) tuples
            
        Returns:
            Path to saved JSON file
        """
        frames_info = {
            "video_path": video_path,
            "total_frames": len(frames),
            "target_fps": self.fps,
            "frames": [
                {
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "frame_index": idx
                }
                for idx, (frame_path, timestamp) in enumerate(frames)
            ]
        }
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "frames_info.json"
        
        save_json(frames_info, str(json_path))
        logger.info(f"Saved frames info to {json_path}")
        return json_path
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string (e.g., "1h 23m 45s")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)
