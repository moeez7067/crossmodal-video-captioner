"""
Visual embeddings module for extracting semantic embeddings from video frames.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image
import config
from src.utils.logger import get_logger
from src.utils.file_utils import get_video_output_dir, get_video_embeddings_dir, get_video_metadata_dir, save_json
from typing import Dict

logger = get_logger(__name__)


class VisualEmbeddings:
    """Extracts visual embeddings using CLIP or similar models."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize visual embedding model.
        
        Args:
            model_name: CLIP model name (ViT-B/32, ViT-L/14, etc.) (default: from config)
            device: Device to run model on (cuda/cpu, None for auto-detection)
        """
        self.model_name = model_name if model_name is not None else config.CLIP_MODEL
        self.device = device if device is not None else config.CLIP_DEVICE
        self.model = None
        self.preprocess = None
        self._is_loaded = False
    
    def load_model(self):
        """Load the CLIP model and preprocessing function."""
        if self._is_loaded and self.model is not None:
            logger.debug("CLIP model already loaded")
            return
        
        logger.info(f"Loading CLIP model: {self.model_name} on device: {self.device}")
        
        try:
            import clip
            
            # Load model and preprocessing
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()  # Set to evaluation mode
            
            self._is_loaded = True
            logger.info(f"Successfully loaded CLIP model: {self.model_name}")
            
        except ImportError:
            error_msg = (
                "CLIP not installed. Install it with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"Could not load CLIP model {self.model_name}: {e}") from e
    
    def extract_embeddings(self, frames: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings from a list of frames.
        
        Args:
            frames: List of frame arrays (numpy arrays in RGB format)
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings with shape (num_frames, embedding_dim)
        """
        if not self._is_loaded:
            self.load_model()
        
        if not frames:
            return np.array([])
        
        logger.info(f"Extracting embeddings from {len(frames)} frames")
        
        # Convert frames to PIL Images and preprocess
        processed_images = []
        for frame in frames:
            # Convert numpy array to PIL Image
            if isinstance(frame, np.ndarray):
                # Ensure frame is in correct format (RGB, uint8)
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                pil_image = Image.fromarray(frame)
            else:
                pil_image = frame
            
            # Preprocess for CLIP
            processed = self.preprocess(pil_image)
            processed_images.append(processed)
        
        # Convert to tensor
        image_tensor = torch.stack(processed_images).to(self.device)
        
        # Extract embeddings in batches
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(processed_images), batch_size):
                batch = image_tensor[i:i + batch_size]
                embeddings = self.model.encode_image(batch)
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        
        logger.info(f"Extracted embeddings: shape {embeddings_array.shape}")
        return embeddings_array
    
    def extract_single_embedding(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single frame.
        
        Args:
            frame: Single frame array (numpy array in RGB format)
            
        Returns:
            Embedding vector
        """
        return self.extract_embeddings([frame])[0]
    
    def temporal_pooling(self, embeddings: np.ndarray, method: str = "mean") -> np.ndarray:
        """
        Apply temporal pooling to frame embeddings.
        
        Args:
            embeddings: Array of frame embeddings (num_frames, embedding_dim)
            method: Pooling method ('mean', 'max', 'attention')
            
        Returns:
            Pooled embedding vector
        """
        if embeddings.size == 0:
            return np.array([])
        
        if len(embeddings.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
        
        if method == "mean":
            pooled = np.mean(embeddings, axis=0)
        elif method == "max":
            pooled = np.max(embeddings, axis=0)
        elif method == "attention":
            # Simple attention pooling: weighted average based on embedding magnitude
            weights = np.linalg.norm(embeddings, axis=1, keepdims=True)
            weights = weights / (weights.sum() + 1e-8)  # Normalize
            pooled = np.sum(embeddings * weights, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {method}. Use 'mean', 'max', or 'attention'")
        
        return pooled
    
    def align_with_audio(self, visual_embeddings: np.ndarray, 
                        frame_timestamps: List[float],
                        audio_timestamps: List[float]) -> np.ndarray:
        """
        Align visual embeddings with audio timestamps.
        
        Args:
            visual_embeddings: Frame embeddings array (num_frames, embedding_dim)
            frame_timestamps: List of timestamps for each frame
            audio_timestamps: List of audio segment timestamps (start times)
            
        Returns:
            Aligned visual embeddings for each audio segment (num_audio_segments, embedding_dim)
        """
        if visual_embeddings.size == 0 or not frame_timestamps or not audio_timestamps:
            return np.array([])
        
        if len(visual_embeddings) != len(frame_timestamps):
            raise ValueError(
                f"Mismatch: {len(visual_embeddings)} embeddings but {len(frame_timestamps)} timestamps"
            )
        
        logger.info(f"Aligning {len(visual_embeddings)} visual embeddings with {len(audio_timestamps)} audio segments")
        
        aligned_embeddings = []
        
        for audio_start in audio_timestamps:
            # Find frames that fall within this audio segment
            # For simplicity, find the closest frame to the audio start time
            frame_indices = []
            for idx, frame_time in enumerate(frame_timestamps):
                if abs(frame_time - audio_start) <= 1.0:  # Within 1 second
                    frame_indices.append(idx)
            
            if frame_indices:
                # Pool embeddings from relevant frames
                relevant_embeddings = visual_embeddings[frame_indices]
                pooled = self.temporal_pooling(relevant_embeddings, method=config.FRAME_TEMPORAL_POOLING)
            else:
                # No frames found, use nearest frame
                nearest_idx = min(range(len(frame_timestamps)), 
                                 key=lambda i: abs(frame_timestamps[i] - audio_start))
                pooled = visual_embeddings[nearest_idx]
            
            aligned_embeddings.append(pooled)
        
        aligned_array = np.array(aligned_embeddings)
        logger.info(f"Aligned embeddings shape: {aligned_array.shape}")
        return aligned_array
    
    def extract_embeddings_with_timestamps(self, frames: List[np.ndarray], 
                                          timestamps: List[float],
                                          batch_size: int = 32) -> Tuple[np.ndarray, List[float]]:
        """
        Extract embeddings with associated timestamps.
        
        Args:
            frames: List of frame arrays
            timestamps: List of timestamps for each frame
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (embeddings_array, timestamps_list)
        """
        embeddings = self.extract_embeddings(frames, batch_size)
        return embeddings, timestamps
    
    def save_embeddings(self, video_path: str, embeddings: np.ndarray) -> Path:
        """
        Save visual embeddings to .npy file.
        
        Args:
            video_path: Path to original video file (for output directory)
            embeddings: Embeddings array (num_frames, embedding_dim)
            
        Returns:
            Path to saved .npy file
        """
        embeddings_dir = get_video_embeddings_dir(video_path)
        npy_path = embeddings_dir / "visual_embeddings.npy"
        
        np.save(str(npy_path), embeddings)
        logger.info(f"Saved visual embeddings to {npy_path} (shape: {embeddings.shape})")
        return npy_path
    
    def save_embeddings_info_to_json(self, video_path: str, embeddings: np.ndarray, 
                                     timestamps: Optional[List[float]] = None) -> Path:
        """
        Save visual embeddings metadata to JSON file.
        
        Args:
            video_path: Path to original video file (for output directory)
            embeddings: Embeddings array (num_frames, embedding_dim)
            timestamps: Optional list of timestamps for each embedding
            
        Returns:
            Path to saved JSON file
        """
        from src.utils.file_utils import get_video_metadata_dir
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "visual_embeddings_info.json"
        
        # Get embeddings file path
        embeddings_dir = get_video_embeddings_dir(video_path)
        embeddings_file = embeddings_dir / "visual_embeddings.npy"
        
        embeddings_info = {
            "video_path": video_path,
            "model_name": self.model_name,
            "num_frames": len(embeddings) if len(embeddings) > 0 else 0,
            "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
            "embeddings_shape": list(embeddings.shape) if len(embeddings) > 0 else [],
            "has_timestamps": timestamps is not None,
            "timestamps": timestamps if timestamps else [],
            "embeddings_file": str(embeddings_file),
            "note": "Actual embeddings are saved to .npy file. Use numpy.load() to load embeddings."
        }
        
        save_json(embeddings_info, str(json_path))
        logger.info(f"Saved visual embeddings info to {json_path}")
        return json_path
