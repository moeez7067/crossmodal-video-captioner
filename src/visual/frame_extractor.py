"""
Frame extraction module for processing video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import config
import torch
from src.preprocessing.video_processor import VideoProcessor
from src.utils.logger import get_logger
from src.utils.file_utils import get_video_output_dir, save_json

logger = get_logger(__name__)


class FrameExtractor:
    """Extracts and processes frames from video files."""
    
    def __init__(self, target_fps: Optional[float] = None):
        """
        Initialize frame extractor.
        
        Args:
            target_fps: Target frames per second to extract (default: from config)
        """
        self.target_fps = target_fps if target_fps is not None else config.VIDEO_FPS
        self.video_processor = VideoProcessor(fps=self.target_fps)
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None, 
                      return_arrays: bool = True) -> List[Tuple[np.ndarray, float]]:
        """
        Extract frames from video at specified FPS.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames (optional)
            return_arrays: If True, return frame arrays; if False, only return paths
            
        Returns:
            List of tuples (frame_array, timestamp) for each frame
        """
        logger.info(f"Extracting frames from {video_path} at {self.target_fps} FPS")
        
        # Use VideoProcessor to extract frames
        frame_paths = self.video_processor.extract_frames(video_path, output_dir)
        
        if not return_arrays:
            # Return paths with timestamps only
            return [(path, timestamp) for path, timestamp in frame_paths]
        
        # Load frames as numpy arrays
        frames = []
        for frame_path, timestamp in frame_paths:
            try:
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((frame_rgb, timestamp))
                else:
                    logger.warning(f"Could not load frame: {frame_path}")
            except Exception as e:
                logger.error(f"Error loading frame {frame_path}: {e}")
        
        logger.info(f"Extracted {len(frames)} frames as numpy arrays")
        return frames
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess frame for model input.
        
        Args:
            frame: Input frame as numpy array (RGB format)
            target_size: Target size (width, height) for resizing (default: from config)
            
        Returns:
            Preprocessed frame (normalized, resized)
        """
        if target_size is None:
            target_size = config.FRAME_PREPROCESS_SIZE
        
        # Resize frame
        if frame.shape[:2] != target_size[::-1]:  # OpenCV uses (height, width)
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize pixel values to [0, 1] range
        frame_normalized = frame.astype(np.float32) / 255.0
        
        return frame_normalized
    
    def detect_slides(self, frames: List[np.ndarray], timestamps: Optional[List[float]] = None,
                     threshold: float = 0.3) -> List[Dict]:
        """
        Detect slide transitions in video frames.
        
        Args:
            frames: List of frame arrays
            timestamps: Optional list of timestamps for each frame
            threshold: Threshold for detecting significant changes (0.0 to 1.0)
            
        Returns:
            List of dictionaries with slide detection results
        """
        if len(frames) < 2:
            return []
        
        logger.info(f"Detecting slide transitions in {len(frames)} frames")
        
        slide_changes = []
        prev_frame = None
        
        for idx, frame in enumerate(frames):
            if prev_frame is None:
                prev_frame = frame
                continue
            
            # Calculate structural similarity between consecutive frames
            similarity = self._calculate_frame_similarity(prev_frame, frame)
            
            # Detect significant change (slide transition)
            if similarity < (1.0 - threshold):
                timestamp = timestamps[idx] if timestamps and idx < len(timestamps) else idx / self.target_fps
                slide_changes.append({
                    "frame_index": idx,
                    "timestamp": timestamp,
                    "similarity": similarity,
                    "change_magnitude": 1.0 - similarity
                })
                logger.debug(f"Slide transition detected at frame {idx}, timestamp: {timestamp:.2f}s")
            
            prev_frame = frame
        
        logger.info(f"Detected {len(slide_changes)} slide transitions")
        return slide_changes
    
    def _calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate similarity between two frames using structural similarity.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Resize frames to same size if needed
        if frame1.shape != frame2.shape:
            target_size = (224, 224)
            frame1 = cv2.resize(frame1, target_size)
            frame2 = cv2.resize(frame2, target_size)
        
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) if len(frame2.shape) == 3 else frame2
        
        # Calculate structural similarity index (SSIM)
        try:
            from skimage.metrics import structural_similarity as ssim
            similarity = ssim(gray1, gray2, data_range=255)
        except ImportError:
            # Fallback to simple correlation if scikit-image not available
            logger.warning("scikit-image not available, using simple correlation")
            gray1_norm = gray1.astype(np.float32) / 255.0
            gray2_norm = gray2.astype(np.float32) / 255.0
            correlation = np.corrcoef(gray1_norm.flatten(), gray2_norm.flatten())[0, 1]
            similarity = max(0.0, correlation) if not np.isnan(correlation) else 0.0
        
        return similarity
    
    def extract_on_screen_text(self, frame: np.ndarray, use_easyocr: bool = True) -> str:
        """
        Extract text visible on screen using OCR.
        
        Args:
            frame: Input frame (RGB numpy array)
            use_easyocr: If True, use EasyOCR; if False, use pytesseract
            
        Returns:
            Extracted text string
        """
        try:
            if use_easyocr:
                return self._extract_text_easyocr(frame)
            else:
                return self._extract_text_tesseract(frame)
        except Exception as e:
            logger.error(f"Error extracting text from frame: {e}")
            return ""
    
    def _extract_text_easyocr(self, frame: np.ndarray) -> str:
        """Extract text using EasyOCR."""
        try:
            import easyocr
            # Initialize EasyOCR reader (lazy loading)
            if not hasattr(self, '_easyocr_reader'):
                logger.info("Initializing EasyOCR reader")
                self._easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            
            # Convert numpy array to PIL Image format that EasyOCR expects
            results = self._easyocr_reader.readtext(frame)
            
            # Extract text from results
            text_parts = [result[1] for result in results]
            extracted_text = " ".join(text_parts)
            
            return extracted_text
            
        except ImportError:
            logger.warning("EasyOCR not available, falling back to pytesseract")
            return self._extract_text_tesseract(frame)
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return ""
    
    def _extract_text_tesseract(self, frame: np.ndarray) -> str:
        """Extract text using pytesseract."""
        try:
            import pytesseract
            from PIL import Image
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Extract text
            text = pytesseract.image_to_string(pil_image)
            
            return text.strip()
            
        except ImportError:
            logger.error("pytesseract not available. Install with: pip install pytesseract")
            return ""
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return ""
    
    def save_slide_detection_to_json(self, video_path: str, slide_changes: List[Dict]) -> Path:
        """
        Save slide detection results to JSON file.
        
        Args:
            video_path: Path to original video file (for output directory)
            slide_changes: List of slide detection results from detect_slides()
            
        Returns:
            Path to saved JSON file
        """
        from src.utils.file_utils import get_video_metadata_dir
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "slide_detection.json"
        
        slide_data = {
            "video_path": video_path,
            "total_transitions": len(slide_changes),
            "transitions": slide_changes
        }
        
        save_json(slide_data, str(json_path))
        logger.info(f"Saved slide detection to {json_path}")
        return json_path
