"""
Ablated model variants for ablation studies.
These models systematically remove components from base models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path

from src.comparison.base_model import BaseCaptionModel
from src.comparison.baselines.simple_fusion import GatedFusion, SimpleConcatenationFusion
from src.audio.speech_to_text import SpeechToText
from src.visual.visual_embeddings import VisualEmbeddings
from src.visual.frame_extractor import FrameExtractor
from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.audio_extractor import AudioExtractor
from src.generation.caption_generator import CaptionGenerator
from src.fusion.utils import align_timestamps, pool_embeddings
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


class AblatedGatedFusion(BaseCaptionModel):
    """
    Ablated version of GatedFusion with configurable component removal.
    """
    
    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        remove_gating: bool = False,
        remove_audio: bool = False,
        remove_visual: bool = False,
        remove_projection: bool = False
    ):
        """
        Initialize ablated gated fusion model.
        
        Args:
            config_dict: Optional configuration dictionary
            remove_gating: If True, remove gating mechanism (use addition instead)
            remove_audio: If True, remove audio modality
            remove_visual: If True, remove visual modality
            remove_projection: If True, remove projection layers
        """
        super().__init__(config_dict)
        
        # Validate that at least one modality remains
        if remove_audio and remove_visual:
            raise ValueError("Cannot remove both audio and visual modalities")
        
        self.remove_gating = remove_gating
        self.remove_audio = remove_audio
        self.remove_visual = remove_visual
        self.remove_projection = remove_projection
        
        # Initialize components
        if not self.remove_audio:
            self.speech_to_text = SpeechToText()
            self.audio_extractor = AudioExtractor()
        
        if not self.remove_visual:
            self.visual_embeddings = VisualEmbeddings()
            self.frame_extractor = FrameExtractor()
            self.video_processor = VideoProcessor()
        
        self.caption_generator = CaptionGenerator()
        
        # Configuration
        self.target_fps = self.config.get("target_fps", config.VIDEO_FPS)
        self.embedding_dim = self.config.get("embedding_dim", 512)
        
        # Projection layers (only if not removed)
        if not self.remove_projection:
            if not self.remove_audio:
                self.audio_projection = nn.Linear(768, self.embedding_dim)
            if not self.remove_visual:
                self.visual_projection = nn.Linear(512, self.embedding_dim)
        else:
            # Use identity if projection is removed
            if not self.remove_audio:
                self.audio_projection = nn.Identity()
            if not self.remove_visual:
                self.visual_projection = nn.Identity()
        
        # Gating network (only if not removed)
        if not self.remove_gating and not self.remove_audio and not self.remove_visual:
            gate_input_dim = self.embedding_dim * 2
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )
        else:
            self.gate_network = None
    
    def _fuse_embeddings(self, audio_emb: Optional[np.ndarray], 
                        visual_emb: Optional[np.ndarray]) -> np.ndarray:
        """
        Fuse embeddings based on ablation configuration.
        
        Args:
            audio_emb: Audio embeddings (num_segments, audio_dim) or None
            visual_emb: Visual embeddings (num_segments, visual_dim) or None
            
        Returns:
            Fused embeddings
        """
        # Handle single modality cases
        if self.remove_audio and visual_emb is not None:
            visual_tensor = torch.from_numpy(visual_emb).float()
            if not self.remove_projection:
                visual_proj = self.visual_projection(visual_tensor)
            else:
                visual_proj = visual_tensor
            return visual_proj.detach().numpy()
        
        if self.remove_visual and audio_emb is not None:
            audio_tensor = torch.from_numpy(audio_emb).float()
            if not self.remove_projection:
                audio_proj = self.audio_projection(audio_tensor)
            else:
                audio_proj = audio_tensor
            return audio_proj.detach().numpy()
        
        # Both modalities present
        if audio_emb is None or visual_emb is None:
            raise ValueError("Both embeddings required for fusion")
        
        # Project to same dimension
        audio_tensor = torch.from_numpy(audio_emb).float()
        visual_tensor = torch.from_numpy(visual_emb).float()
        
        audio_proj = self.audio_projection(audio_tensor)
        visual_proj = self.visual_projection(visual_tensor)
        
        # Ensure same number of segments
        min_segments = min(len(audio_proj), len(visual_proj))
        audio_proj = audio_proj[:min_segments]
        visual_proj = visual_proj[:min_segments]
        
        # Fuse based on ablation configuration
        if self.remove_gating or self.gate_network is None:
            # Fall back to addition
            fused = audio_proj + visual_proj
        else:
            # Use gating mechanism
            gate_input = torch.cat([audio_proj, visual_proj], dim=1)
            gate_values = self.gate_network(gate_input)
            fused = gate_values * audio_proj + (1 - gate_values) * visual_proj
        
        return fused.detach().numpy()
    
    def generate_captions(self, video_path: str) -> List[Dict]:
        """Generate captions using ablated gated fusion."""
        logger.info(f"Generating captions (ablated gated fusion) for {video_path}")
        logger.info(f"Ablation config: gating={not self.remove_gating}, "
                   f"audio={not self.remove_audio}, visual={not self.remove_visual}, "
                   f"projection={not self.remove_projection}")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            audio_embeddings = None
            audio_timestamps = []
            transcription = None
            
            # Extract audio if not removed
            if not self.remove_audio:
                audio_path = self.audio_extractor.extract_audio(str(video_path))
                transcription = self.speech_to_text.transcribe(
                    audio_path,
                    extract_embeddings=True
                )
                audio_embeddings = transcription.get("embeddings")
                audio_timestamps = transcription.get("embedding_timestamps", [])
            
            visual_embeddings = None
            frame_timestamps = []
            
            # Extract visual if not removed
            if not self.remove_visual:
                frames_info = self.video_processor.extract_frames(
                    str(video_path),
                    target_fps=self.target_fps
                )
                
                frames = []
                for frame_info in frames_info:
                    frame_path = Path(frame_info["frame_path"])
                    if frame_path.exists():
                        frame_array = self.frame_extractor.preprocess_frame(str(frame_path))
                        frames.append(frame_array)
                        frame_timestamps.append(frame_info["timestamp"])
                
                if frames:
                    visual_embeddings = self.visual_embeddings.extract_embeddings(frames)
            
            # Handle fallback cases
            if self.remove_audio and (visual_embeddings is None or len(visual_embeddings) == 0):
                logger.warning("No visual embeddings available, cannot generate captions")
                return []
            
            if self.remove_visual and (audio_embeddings is None or len(audio_embeddings) == 0):
                if transcription:
                    segments = transcription.get("segments", [])
                    return self.caption_generator.generate_captions_from_transcript(segments)
                return []
            
            # Align embeddings if both present
            if audio_embeddings is not None and visual_embeddings is not None:
                aligned_audio, aligned_visual, aligned_times = align_timestamps(
                    audio_embeddings, audio_timestamps,
                    visual_embeddings, frame_timestamps
                )
            else:
                # Single modality
                if audio_embeddings is not None:
                    aligned_audio = audio_embeddings
                    aligned_visual = None
                else:
                    aligned_audio = None
                    aligned_visual = visual_embeddings
            
            # Fuse embeddings
            fused_embeddings = self._fuse_embeddings(aligned_audio, aligned_visual)
            
            # Generate captions
            if transcription:
                segments = transcription.get("segments", [])
                captions = self.caption_generator.generate_captions_from_transcript(segments)
            else:
                # Fallback: generate from embeddings
                captions = self.caption_generator.generate_captions_from_transcript([])
            
            logger.info(f"Generated {len(captions)} captions using ablated gated fusion")
            return captions
            
        except Exception as e:
            logger.error(f"Error generating captions with ablated gated fusion: {e}")
            raise


class AblatedConcatenationFusion(BaseCaptionModel):
    """
    Ablated version of SimpleConcatenationFusion with configurable component removal.
    """
    
    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        remove_projection: bool = False,
        remove_audio: bool = False,
        remove_visual: bool = False
    ):
        """
        Initialize ablated concatenation fusion model.
        
        Args:
            config_dict: Optional configuration dictionary
            remove_projection: If True, remove projection layer
            remove_audio: If True, remove audio modality
            remove_visual: If True, remove visual modality
        """
        super().__init__(config_dict)
        
        # Validate that at least one modality remains
        if remove_audio and remove_visual:
            raise ValueError("Cannot remove both audio and visual modalities")
        
        self.remove_projection = remove_projection
        self.remove_audio = remove_audio
        self.remove_visual = remove_visual
        
        # Initialize components
        if not self.remove_audio:
            self.speech_to_text = SpeechToText()
            self.audio_extractor = AudioExtractor()
        
        if not self.remove_visual:
            self.visual_embeddings = VisualEmbeddings()
            self.frame_extractor = FrameExtractor()
            self.video_processor = VideoProcessor()
        
        self.caption_generator = CaptionGenerator()
        
        # Configuration
        self.target_fps = self.config.get("target_fps", config.VIDEO_FPS)
        self.audio_dim = self.config.get("audio_dim", 768)
        self.visual_dim = self.config.get("visual_dim", 512)
        self.fused_dim = self.audio_dim + self.visual_dim
        
        # Projection layer (only if not removed)
        if not self.remove_projection:
            output_dim = self.config.get("output_dim", 512)
            self.projection = nn.Linear(self.fused_dim, output_dim)
        else:
            self.projection = None
    
    def _fuse_embeddings(self, audio_emb: Optional[np.ndarray], 
                        visual_emb: Optional[np.ndarray]) -> np.ndarray:
        """
        Fuse embeddings by concatenation.
        
        Args:
            audio_emb: Audio embeddings or None
            visual_emb: Visual embeddings or None
            
        Returns:
            Fused embeddings
        """
        # Handle single modality cases
        if self.remove_audio and visual_emb is not None:
            fused = visual_emb
        elif self.remove_visual and audio_emb is not None:
            fused = audio_emb
        else:
            # Both modalities present
            if audio_emb is None or visual_emb is None:
                raise ValueError("Both embeddings required for fusion")
            
            # Ensure same number of segments
            min_segments = min(len(audio_emb), len(visual_emb))
            audio_emb = audio_emb[:min_segments]
            visual_emb = visual_emb[:min_segments]
            
            # Concatenate
            fused = np.concatenate([audio_emb, visual_emb], axis=1)
        
        # Apply projection if configured
        if self.projection is not None:
            fused_tensor = torch.from_numpy(fused).float()
            fused_tensor = self.projection(fused_tensor)
            fused = fused_tensor.detach().numpy()
        
        return fused
    
    def generate_captions(self, video_path: str) -> List[Dict]:
        """Generate captions using ablated concatenation fusion."""
        logger.info(f"Generating captions (ablated concatenation fusion) for {video_path}")
        logger.info(f"Ablation config: projection={not self.remove_projection}, "
                   f"audio={not self.remove_audio}, visual={not self.remove_visual}")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            audio_embeddings = None
            audio_timestamps = []
            transcription = None
            
            # Extract audio if not removed
            if not self.remove_audio:
                audio_path = self.audio_extractor.extract_audio(str(video_path))
                transcription = self.speech_to_text.transcribe(
                    audio_path,
                    extract_embeddings=True
                )
                audio_embeddings = transcription.get("embeddings")
                audio_timestamps = transcription.get("embedding_timestamps", [])
            
            visual_embeddings = None
            frame_timestamps = []
            
            # Extract visual if not removed
            if not self.remove_visual:
                frames_info = self.video_processor.extract_frames(
                    str(video_path),
                    target_fps=self.target_fps
                )
                
                frames = []
                for frame_info in frames_info:
                    frame_path = Path(frame_info["frame_path"])
                    if frame_path.exists():
                        frame_array = self.frame_extractor.preprocess_frame(str(frame_path))
                        frames.append(frame_array)
                        frame_timestamps.append(frame_info["timestamp"])
                
                if frames:
                    visual_embeddings = self.visual_embeddings.extract_embeddings(frames)
            
            # Handle fallback cases
            if self.remove_audio and (visual_embeddings is None or len(visual_embeddings) == 0):
                logger.warning("No visual embeddings available, cannot generate captions")
                return []
            
            if self.remove_visual and (audio_embeddings is None or len(audio_embeddings) == 0):
                if transcription:
                    segments = transcription.get("segments", [])
                    return self.caption_generator.generate_captions_from_transcript(segments)
                return []
            
            # Align embeddings if both present
            if audio_embeddings is not None and visual_embeddings is not None:
                aligned_audio, aligned_visual, aligned_times = align_timestamps(
                    audio_embeddings, audio_timestamps,
                    visual_embeddings, frame_timestamps
                )
            else:
                # Single modality
                if audio_embeddings is not None:
                    aligned_audio = audio_embeddings
                    aligned_visual = None
                else:
                    aligned_audio = None
                    aligned_visual = visual_embeddings
            
            # Fuse embeddings
            fused_embeddings = self._fuse_embeddings(aligned_audio, aligned_visual)
            
            # Generate captions
            if transcription:
                segments = transcription.get("segments", [])
                captions = self.caption_generator.generate_captions_from_transcript(segments)
            else:
                captions = self.caption_generator.generate_captions_from_transcript([])
            
            logger.info(f"Generated {len(captions)} captions using ablated concatenation fusion")
            return captions
            
        except Exception as e:
            logger.error(f"Error generating captions with ablated concatenation fusion: {e}")
            raise


def create_ablated_model(variant_name: str, config_dict: Optional[Dict[str, Any]] = None) -> BaseCaptionModel:
    """
    Factory function to create an ablated model based on variant name.
    
    Args:
        variant_name: Name of the ablation variant (e.g., "gated_no_gating")
        config_dict: Optional configuration dictionary
        
    Returns:
        Ablated model instance
    """
    from .config import ABLATION_VARIANTS
    
    if variant_name not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant_name}")
    
    variant = ABLATION_VARIANTS[variant_name]
    base_model = variant["base_model"]
    ablation_type = variant["ablation_type"]
    
    if base_model == "gated":
        remove_gating = ablation_type == "remove_gating" or "gating" in variant.get("components_removed", [])
        remove_audio = variant.get("modality_removed") == "audio"
        remove_visual = variant.get("modality_removed") == "visual"
        remove_projection = ablation_type == "remove_projection" or "projection" in variant.get("components_removed", [])
        
        return AblatedGatedFusion(
            config_dict=config_dict,
            remove_gating=remove_gating,
            remove_audio=remove_audio,
            remove_visual=remove_visual,
            remove_projection=remove_projection
        )
    
    elif base_model == "concatenation":
        remove_projection = ablation_type == "remove_projection" or "projection" in variant.get("components_removed", [])
        remove_audio = variant.get("modality_removed") == "audio"
        remove_visual = variant.get("modality_removed") == "visual"
        
        return AblatedConcatenationFusion(
            config_dict=config_dict,
            remove_projection=remove_projection,
            remove_audio=remove_audio,
            remove_visual=remove_visual
        )
    
    else:
        raise ValueError(f"Unsupported base model for ablation: {base_model}")
