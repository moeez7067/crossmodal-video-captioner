"""
Simple fusion models for video captioning.

This module implements baseline fusion strategies for combining audio and visual
embeddings. These models serve as baselines to compare against more sophisticated
approaches like cross-attention transformers.

Implemented Models:
- SimpleConcatenationFusion: Concatenates audio and visual embeddings
- AdditionFusion: Element-wise addition of embeddings (requires same dimension)
- GatedFusion: Learnable gating mechanism to weight each modality

These models are used in comparative analysis to establish baseline performance
and demonstrate the value of more advanced fusion techniques.

Example Usage:
    >>> from src.comparison.baselines.simple_fusion import GatedFusion
    >>> 
    >>> model = GatedFusion()
    >>> captions = model.generate_captions("path/to/video.mp4")
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from src.comparison.base_model import BaseCaptionModel
from src.audio.speech_to_text import SpeechToText
from src.visual.visual_embeddings import VisualEmbeddings
from src.visual.frame_extractor import FrameExtractor
from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.audio_extractor import AudioExtractor
from src.generation.caption_generator import CaptionGenerator
from src.fusion.utils import align_embeddings_with_timestamps, pool_embeddings
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


class SimpleConcatenationFusion(BaseCaptionModel):
    """
    Simple concatenation fusion model.
    
    This is the simplest fusion strategy: concatenates audio and visual embeddings
    along the feature dimension, then optionally projects to a target dimension.
    
    Architecture:
        1. Extract audio embeddings (Whisper) and visual embeddings (CLIP)
        2. Align embeddings temporally
        3. Concatenate: [audio_emb; visual_emb] → [audio_dim + visual_dim]
        4. Optional linear projection to reduce dimension
        5. Generate captions from fused embeddings
    
    This serves as a baseline to show that simple concatenation works but
    can be improved with attention mechanisms.
    
    Args:
        config_dict: Optional configuration with keys:
            - "use_projection": Whether to use projection layer (default: True)
            - "output_dim": Output dimension after projection (default: 512)
            - "audio_dim": Audio embedding dimension (default: 768)
            - "visual_dim": Visual embedding dimension (default: 512)
            - "target_fps": Target frame rate for processing (default: from config)
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize concatenation fusion model.
        
        Sets up all required components:
        - Audio processing (Whisper for transcription and embeddings)
        - Visual processing (CLIP for visual embeddings)
        - Video/audio extraction utilities
        - Caption generation
        
        Args:
            config_dict: Optional configuration dictionary (see class docstring)
        """
        super().__init__(config_dict)
        self.speech_to_text = SpeechToText()
        self.visual_embeddings = VisualEmbeddings()
        self.frame_extractor = FrameExtractor()
        self.audio_extractor = AudioExtractor()
        self.caption_generator = CaptionGenerator()
        
        # Configuration
        self.target_fps = self.config.get("target_fps", config.VIDEO_FPS)
        
        # Initialize VideoProcessor with target FPS
        self.video_processor = VideoProcessor(fps=self.target_fps)
        # Default dimensions (will be detected at runtime)
        # Whisper base: 512, Whisper large: 1280
        # CLIP ViT-B/32: 512
        self.audio_dim = self.config.get("audio_dim", None)  # Will be detected from actual embeddings
        self.visual_dim = self.config.get("visual_dim", None)  # Will be detected from actual embeddings
        self.fused_dim = None  # Will be calculated from actual dimensions
        
        # Projection layer will be created dynamically based on actual fused dimension
        # This avoids dimension mismatches when actual embeddings differ from assumptions
        self.projection = None
        self.use_projection = self.config.get("use_projection", True)
        self.output_dim = self.config.get("output_dim", 512)
    
    def _fuse_embeddings(self, audio_emb: np.ndarray, visual_emb: np.ndarray) -> np.ndarray:
        """
        Fuse audio and visual embeddings by concatenation.
        
        This is the simplest fusion method: combines embeddings by concatenating
        them along the feature dimension. The resulting embedding has dimension
        equal to the sum of input dimensions.
        
        Process:
            1. Align sequences to same length (take minimum)
            2. Concatenate: [audio_features; visual_features]
            3. Optionally project to lower dimension
        
        Args:
            audio_emb: Audio embeddings array [num_segments, audio_dim]
            visual_emb: Visual embeddings array [num_segments, visual_dim]
            
        Returns:
            Fused embeddings [num_segments, fused_dim] where:
            - fused_dim = audio_dim + visual_dim (if no projection)
            - fused_dim = output_dim (if projection is used)
        """
        # Validate inputs
        if not isinstance(audio_emb, np.ndarray) or not isinstance(visual_emb, np.ndarray):
            raise TypeError("Both embeddings must be numpy arrays")
        
        if len(audio_emb.shape) != 2 or len(visual_emb.shape) != 2:
            raise ValueError(f"Embeddings must be 2D arrays. Got audio: {audio_emb.shape}, visual: {visual_emb.shape}")
        
        # Step 1: Ensure same number of temporal segments
        # Take minimum to avoid padding issues
        min_segments = min(len(audio_emb), len(visual_emb))
        audio_emb = audio_emb[:min_segments]
        visual_emb = visual_emb[:min_segments]
        
        # Get actual dimensions
        actual_audio_dim = audio_emb.shape[1]
        actual_visual_dim = visual_emb.shape[1]
        actual_fused_dim = actual_audio_dim + actual_visual_dim
        
        logger.debug(f"Fusing embeddings: audio {audio_emb.shape}, visual {visual_emb.shape}, fused will be {actual_fused_dim}")
        
        # Step 2: Concatenate along feature dimension (axis=1)
        # Result: [num_segments, audio_dim + visual_dim]
        fused = np.concatenate([audio_emb, visual_emb], axis=1)
        
        # Step 3: Apply linear projection if configured
        if self.use_projection:
            # Create or update projection layer to match actual fused dimension
            if self.projection is None or self.projection.in_features != actual_fused_dim:
                logger.info(
                    f"Creating projection layer: {actual_fused_dim} -> {self.output_dim} "
                    f"(previous: {self.projection.in_features if self.projection else 'None'} -> {self.projection.out_features if self.projection else 'None'})"
                )
                self.projection = nn.Linear(actual_fused_dim, self.output_dim)
            
            fused_tensor = torch.from_numpy(fused).float()
            fused_tensor = self.projection(fused_tensor)
            fused = fused_tensor.detach().numpy()
        
        return fused
    
    def generate_captions(self, video_path: str) -> List[Dict]:
        """
        Generate captions using concatenation fusion.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of caption dictionaries
        """
        logger.info(f"Generating captions (concatenation fusion) for {video_path}")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Extract audio and get transcription
            audio_path = self.audio_extractor.extract_audio(str(video_path))
            transcription = self.speech_to_text.transcribe(
                audio_path,
                extract_embeddings=True
            )
            
            # Get audio embeddings
            audio_embeddings = transcription.get("embeddings")
            audio_timestamps = transcription.get("embedding_timestamps", [])
            
            logger.debug(f"Retrieved audio_embeddings type: {type(audio_embeddings)}, value preview: {str(audio_embeddings)[:100] if audio_embeddings is not None else None}")
            
            # Validate embeddings are numpy arrays
            if audio_embeddings is not None:
                if isinstance(audio_embeddings, str):
                    # Might be a file path - try to load it
                    if Path(audio_embeddings).exists() and audio_embeddings.endswith('.npy'):
                        logger.info(f"Loading audio embeddings from file: {audio_embeddings}")
                        try:
                            audio_embeddings = np.load(audio_embeddings)
                        except Exception as e:
                            logger.error(f"Could not load embeddings from file: {e}")
                            audio_embeddings = None
                    else:
                        logger.error(f"Audio embeddings is a string but not a valid file path: {audio_embeddings}")
                        audio_embeddings = None
                elif not isinstance(audio_embeddings, np.ndarray):
                    logger.warning(f"Audio embeddings is not a numpy array (type: {type(audio_embeddings)}), converting...")
                    try:
                        audio_embeddings = np.array(audio_embeddings)
                    except Exception as e:
                        logger.error(f"Could not convert audio embeddings to numpy array: {e}")
                        audio_embeddings = None
                
                # Final validation - ensure it's a valid numpy array
                if audio_embeddings is not None:
                    if not isinstance(audio_embeddings, np.ndarray):
                        logger.error(f"Audio embeddings is still not a numpy array after conversion")
                        audio_embeddings = None
                    elif len(audio_embeddings) == 0:
                        logger.warning("Audio embeddings array is empty")
                        audio_embeddings = None
                    else:
                        # Ensure it's a 2D array
                        if len(audio_embeddings.shape) == 1:
                            audio_embeddings = audio_embeddings.reshape(1, -1)
                        logger.debug(f"Audio embeddings shape: {audio_embeddings.shape}")
            
            if audio_embeddings is None or len(audio_embeddings) == 0:
                logger.warning("No audio embeddings extracted, falling back to transcript")
                # Fallback to transcript-based captions
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            # Extract visual embeddings (FPS is set during VideoProcessor initialization)
            frames_info = self.video_processor.extract_frames(str(video_path))
            
            frames = []
            frame_timestamps = []
            # frames_info is a list of tuples: (frame_path, timestamp)
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
                        
                        if not isinstance(frame_array, np.ndarray):
                            logger.warning(f"preprocess_frame returned non-array type {type(frame_array)}, skipping")
                            continue
                        
                        frames.append(frame_array)
                        frame_timestamps.append(timestamp)
                    except Exception as e:
                        logger.warning(f"Error processing frame {frame_path_str}: {e}, skipping")
                        continue
            
            if not frames:
                logger.warning("No frames extracted, using audio-only")
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            visual_embeddings = self.visual_embeddings.extract_embeddings(frames)
            
            # Validate visual embeddings
            if not isinstance(visual_embeddings, np.ndarray) or len(visual_embeddings) == 0:
                logger.warning("Invalid visual embeddings, using audio-only")
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            logger.debug(f"Visual embeddings shape: {visual_embeddings.shape}")
            
            # Validate embeddings before alignment
            logger.debug(f"Audio embeddings shape: {audio_embeddings.shape}, timestamps: {len(audio_timestamps)}")
            logger.debug(f"Visual embeddings shape: {visual_embeddings.shape}, timestamps: {len(frame_timestamps)}")
            
            # Align audio and visual embeddings temporally
            try:
                aligned_audio, aligned_visual, aligned_times = align_embeddings_with_timestamps(
                    audio_embeddings, audio_timestamps,
                    visual_embeddings, frame_timestamps,
                    method="interpolate"
                )
                logger.debug(f"Aligned audio shape: {aligned_audio.shape}, visual shape: {aligned_visual.shape}")
            except Exception as e:
                logger.error(f"Error aligning embeddings: {e}", exc_info=True)
                # Fallback to audio-only
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            # Fuse embeddings
            try:
                fused_embeddings = self._fuse_embeddings(aligned_audio, aligned_visual)
                logger.debug(f"Fused embeddings shape: {fused_embeddings.shape}")
            except Exception as e:
                logger.error(f"Error fusing embeddings: {e}", exc_info=True)
                # Fallback to audio-only
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            # Generate captions from fused embeddings
            # For now, use transcript segments with fused context
            # In a full implementation, you'd use the fused embeddings directly
            segments = transcription.get("segments", [])
            captions = self.caption_generator.generate_captions_from_transcript(segments)
            
            logger.info(f"Generated {len(captions)} captions using concatenation fusion")
            return captions
            
        except Exception as e:
            logger.error(f"Error generating captions with concatenation fusion: {e}")
            raise


class AdditionFusion(BaseCaptionModel):
    """
    Addition fusion: element-wise addition of audio and visual embeddings.
    Requires embeddings to have the same dimension.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize addition fusion model.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        super().__init__(config_dict)
        self.speech_to_text = SpeechToText()
        self.visual_embeddings = VisualEmbeddings()
        self.frame_extractor = FrameExtractor()
        self.audio_extractor = AudioExtractor()
        self.caption_generator = CaptionGenerator()
        
        # Configuration
        self.target_fps = self.config.get("target_fps", config.VIDEO_FPS)
        
        # Initialize VideoProcessor with target FPS
        self.video_processor = VideoProcessor(fps=self.target_fps)
        self.embedding_dim = self.config.get("embedding_dim", 512)
        
        # Projection layers will be created dynamically based on actual embedding dimensions
        # We'll detect dimensions at runtime and create projections accordingly
        self.audio_projection = None  # Will be created when we know actual audio dim
        self.visual_projection = None  # Will be created when we know actual visual dim
    
    def _fuse_embeddings(self, audio_emb: np.ndarray, visual_emb: np.ndarray) -> np.ndarray:
        """
        Fuse audio and visual embeddings by element-wise addition.
        
        Args:
            audio_emb: Audio embeddings (num_segments, audio_dim)
            visual_emb: Visual embeddings (num_segments, visual_dim)
            
        Returns:
            Fused embeddings (num_segments, embedding_dim)
        """
        # Validate inputs
        if not isinstance(audio_emb, np.ndarray) or not isinstance(visual_emb, np.ndarray):
            raise TypeError("Both embeddings must be numpy arrays")
        
        if len(audio_emb.shape) != 2 or len(visual_emb.shape) != 2:
            raise ValueError(f"Embeddings must be 2D arrays. Got audio: {audio_emb.shape}, visual: {visual_emb.shape}")
        
        # Get actual dimensions
        actual_audio_dim = audio_emb.shape[1]
        actual_visual_dim = visual_emb.shape[1]
        
        # Create projection layers if they don't exist or don't match
        if self.audio_projection is None or self.audio_projection.in_features != actual_audio_dim:
            logger.debug(f"Creating audio projection: {actual_audio_dim} -> {self.embedding_dim}")
            self.audio_projection = nn.Linear(actual_audio_dim, self.embedding_dim)
        
        if self.visual_projection is None or self.visual_projection.in_features != actual_visual_dim:
            logger.debug(f"Creating visual projection: {actual_visual_dim} -> {self.embedding_dim}")
            self.visual_projection = nn.Linear(actual_visual_dim, self.embedding_dim)
        
        # Project to same dimension
        audio_tensor = torch.from_numpy(audio_emb).float()
        visual_tensor = torch.from_numpy(visual_emb).float()
        
        audio_proj = self.audio_projection(audio_tensor).detach().numpy()
        visual_proj = self.visual_projection(visual_tensor).detach().numpy()
        
        # Ensure same number of segments
        min_segments = min(len(audio_proj), len(visual_proj))
        audio_proj = audio_proj[:min_segments]
        visual_proj = visual_proj[:min_segments]
        
        # Element-wise addition
        fused = audio_proj + visual_proj
        
        return fused
    
    def generate_captions(self, video_path: str) -> List[Dict]:
        """
        Generate captions using addition fusion.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of caption dictionaries
        """
        logger.info(f"Generating captions (addition fusion) for {video_path}")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Extract audio and get transcription
            audio_path = self.audio_extractor.extract_audio(str(video_path))
            transcription = self.speech_to_text.transcribe(
                audio_path,
                extract_embeddings=True
            )
            
            audio_embeddings = transcription.get("embeddings")
            audio_timestamps = transcription.get("embedding_timestamps", [])
            
            if audio_embeddings is None or len(audio_embeddings) == 0:
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            # Extract visual embeddings (FPS is set during VideoProcessor initialization)
            frames_info = self.video_processor.extract_frames(str(video_path))
            
            frames = []
            frame_timestamps = []
            # frames_info is a list of tuples: (frame_path, timestamp)
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
                        
                        if not isinstance(frame_array, np.ndarray):
                            logger.warning(f"preprocess_frame returned non-array type {type(frame_array)}, skipping")
                            continue
                        
                        frames.append(frame_array)
                        frame_timestamps.append(timestamp)
                    except Exception as e:
                        logger.warning(f"Error processing frame {frame_path_str}: {e}, skipping")
                        continue
            
            if not frames:
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            visual_embeddings = self.visual_embeddings.extract_embeddings(frames)
            
            # Validate visual embeddings
            if not isinstance(visual_embeddings, np.ndarray) or len(visual_embeddings) == 0:
                logger.warning("Invalid visual embeddings, using audio-only")
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            logger.debug(f"Audio embeddings shape: {audio_embeddings.shape}, timestamps: {len(audio_timestamps)}")
            logger.debug(f"Visual embeddings shape: {visual_embeddings.shape}, timestamps: {len(frame_timestamps)}")
            
            # Align embeddings
            try:
                aligned_audio, aligned_visual, aligned_times = align_embeddings_with_timestamps(
                    audio_embeddings, audio_timestamps,
                    visual_embeddings, frame_timestamps,
                    method="interpolate"
                )
                logger.debug(f"Aligned audio shape: {aligned_audio.shape}, visual shape: {aligned_visual.shape}")
            except Exception as e:
                logger.error(f"Error aligning embeddings: {e}", exc_info=True)
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            # Fuse embeddings
            try:
                fused_embeddings = self._fuse_embeddings(aligned_audio, aligned_visual)
                logger.debug(f"Fused embeddings shape: {fused_embeddings.shape}")
            except Exception as e:
                logger.error(f"Error fusing embeddings: {e}", exc_info=True)
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            # Generate captions
            segments = transcription.get("segments", [])
            captions = self.caption_generator.generate_captions_from_transcript(segments)
            
            logger.info(f"Generated {len(captions)} captions using addition fusion")
            return captions
            
        except Exception as e:
            logger.error(f"Error generating captions with addition fusion: {e}")
            raise


class GatedFusion(BaseCaptionModel):
    """
    Gated fusion: learnable gating mechanism to combine audio and visual embeddings.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize gated fusion model.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        super().__init__(config_dict)
        self.speech_to_text = SpeechToText()
        self.visual_embeddings = VisualEmbeddings()
        self.frame_extractor = FrameExtractor()
        self.audio_extractor = AudioExtractor()
        self.caption_generator = CaptionGenerator()
        
        # Configuration
        self.target_fps = self.config.get("target_fps", config.VIDEO_FPS)
        
        # Initialize VideoProcessor with target FPS
        self.video_processor = VideoProcessor(fps=self.target_fps)
        self.embedding_dim = self.config.get("embedding_dim", 512)
        
        # Projection layers will be created dynamically based on actual embedding dimensions
        self.audio_projection = None
        self.visual_projection = None
        
        # Gating network will be created dynamically
        self.gate_network = None
    
    def _fuse_embeddings(self, audio_emb: np.ndarray, visual_emb: np.ndarray) -> np.ndarray:
        """
        Fuse audio and visual embeddings using gated fusion.
        
        Args:
            audio_emb: Audio embeddings (num_segments, audio_dim)
            visual_emb: Visual embeddings (num_segments, visual_dim)
            
        Returns:
            Fused embeddings (num_segments, embedding_dim)
        """
        # Validate inputs
        if not isinstance(audio_emb, np.ndarray) or not isinstance(visual_emb, np.ndarray):
            raise TypeError("Both embeddings must be numpy arrays")
        
        if len(audio_emb.shape) != 2 or len(visual_emb.shape) != 2:
            raise ValueError(f"Embeddings must be 2D arrays. Got audio: {audio_emb.shape}, visual: {visual_emb.shape}")
        
        # Get actual dimensions
        actual_audio_dim = audio_emb.shape[1]
        actual_visual_dim = visual_emb.shape[1]
        
        # Create projection layers if they don't exist or don't match
        if self.audio_projection is None or self.audio_projection.in_features != actual_audio_dim:
            logger.debug(f"Creating audio projection: {actual_audio_dim} -> {self.embedding_dim}")
            self.audio_projection = nn.Linear(actual_audio_dim, self.embedding_dim)
        
        if self.visual_projection is None or self.visual_projection.in_features != actual_visual_dim:
            logger.debug(f"Creating visual projection: {actual_visual_dim} -> {self.embedding_dim}")
            self.visual_projection = nn.Linear(actual_visual_dim, self.embedding_dim)
        
        # Create gating network if it doesn't exist
        if self.gate_network is None:
            gate_input_dim = self.embedding_dim * 2  # Concatenated audio + visual
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()  # Output gate values between 0 and 1
            )
            logger.debug(f"Created gating network: {gate_input_dim} -> {self.embedding_dim}")
        
        # Project to same dimension
        audio_tensor = torch.from_numpy(audio_emb).float()
        visual_tensor = torch.from_numpy(visual_emb).float()
        
        audio_proj = self.audio_projection(audio_tensor)
        visual_proj = self.visual_projection(visual_tensor)
        
        # Ensure same number of segments
        min_segments = min(len(audio_proj), len(visual_proj))
        audio_proj = audio_proj[:min_segments]
        visual_proj = visual_proj[:min_segments]
        
        # Concatenate for gate input
        gate_input = torch.cat([audio_proj, visual_proj], dim=1)
        
        # Compute gate values
        gate_values = self.gate_network(gate_input)
        
        # Gated fusion: g ⊙ audio + (1-g) ⊙ visual
        fused = gate_values * audio_proj + (1 - gate_values) * visual_proj
        
        return fused.detach().numpy()
    
    def generate_captions(self, video_path: str) -> List[Dict]:
        """
        Generate captions using gated fusion.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of caption dictionaries
        """
        logger.info(f"Generating captions (gated fusion) for {video_path}")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Extract audio and get transcription
            audio_path = self.audio_extractor.extract_audio(str(video_path))
            transcription = self.speech_to_text.transcribe(
                audio_path,
                extract_embeddings=True
            )
            
            audio_embeddings = transcription.get("embeddings")
            audio_timestamps = transcription.get("embedding_timestamps", [])
            
            if audio_embeddings is None or len(audio_embeddings) == 0:
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            # Extract visual embeddings (FPS is set during VideoProcessor initialization)
            frames_info = self.video_processor.extract_frames(str(video_path))
            
            frames = []
            frame_timestamps = []
            # frames_info is a list of tuples: (frame_path, timestamp)
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
                        
                        if not isinstance(frame_array, np.ndarray):
                            logger.warning(f"preprocess_frame returned non-array type {type(frame_array)}, skipping")
                            continue
                        
                        frames.append(frame_array)
                        frame_timestamps.append(timestamp)
                    except Exception as e:
                        logger.warning(f"Error processing frame {frame_path_str}: {e}, skipping")
                        continue
            
            if not frames:
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            visual_embeddings = self.visual_embeddings.extract_embeddings(frames)
            
            # Validate visual embeddings
            if not isinstance(visual_embeddings, np.ndarray) or len(visual_embeddings) == 0:
                logger.warning("Invalid visual embeddings, using audio-only")
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            logger.debug(f"Audio embeddings shape: {audio_embeddings.shape}, timestamps: {len(audio_timestamps)}")
            logger.debug(f"Visual embeddings shape: {visual_embeddings.shape}, timestamps: {len(frame_timestamps)}")
            
            # Align embeddings
            try:
                aligned_audio, aligned_visual, aligned_times = align_embeddings_with_timestamps(
                    audio_embeddings, audio_timestamps,
                    visual_embeddings, frame_timestamps,
                    method="interpolate"
                )
                logger.debug(f"Aligned audio shape: {aligned_audio.shape}, visual shape: {aligned_visual.shape}")
            except Exception as e:
                logger.error(f"Error aligning embeddings: {e}", exc_info=True)
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            # Fuse embeddings
            try:
                fused_embeddings = self._fuse_embeddings(aligned_audio, aligned_visual)
                logger.debug(f"Fused embeddings shape: {fused_embeddings.shape}")
            except Exception as e:
                logger.error(f"Error fusing embeddings: {e}", exc_info=True)
                segments = transcription.get("segments", [])
                return self.caption_generator.generate_captions_from_transcript(segments)
            
            # Generate captions
            segments = transcription.get("segments", [])
            captions = self.caption_generator.generate_captions_from_transcript(segments)
            
            logger.info(f"Generated {len(captions)} captions using gated fusion")
            return captions
            
        except Exception as e:
            logger.error(f"Error generating captions with gated fusion: {e}")
            raise
