"""
Fusion Service for Multimodal Audio-Visual Fusion
High-level service that integrates fusion with Phase 2 outputs.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import config
from src.utils.logger import get_logger
from src.utils.file_utils import get_video_embeddings_dir, get_video_metadata_dir, save_json
from src.fusion.fusion_pipeline import create_fusion_pipeline, FusionPipeline

logger = get_logger(__name__)


class FusionService:
    """
    High-level service for multimodal fusion.
    Integrates with Phase 2 outputs and handles errors gracefully.
    """
    
    def __init__(self):
        """Initialize fusion service."""
        self.fusion_pipeline: Optional[FusionPipeline] = None
        self._is_loaded = False
    
    def load_model(self):
        """Lazy load fusion model."""
        if self._is_loaded and self.fusion_pipeline is not None:
            logger.debug("Fusion model already loaded")
            return
        
        if not config.FUSION_ENABLED:
            logger.info("Fusion is disabled in configuration")
            return
        
        try:
            logger.info("Loading fusion model...")
            self.fusion_pipeline = create_fusion_pipeline(
                config={
                    'audio_dim': config.AUDIO_EMBEDDING_DIM,
                    'visual_dim': config.VISUAL_EMBEDDING_DIM,
                    'hidden_dim': config.FUSION_HIDDEN_DIM,
                    'num_layers': config.FUSION_NUM_LAYERS,
                    'num_heads': config.FUSION_NUM_HEADS,
                    'dropout': config.FUSION_DROPOUT
                },
                device=config.FUSION_DEVICE
            )
            self._is_loaded = True
            logger.info("Fusion model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load fusion model: {e}", exc_info=True)
            self.fusion_pipeline = None
            self._is_loaded = False
    
    def fuse_from_phase2_outputs(
        self,
        video_path: str,
        audio_timestamps: Optional[List[float]] = None,
        visual_timestamps: Optional[List[float]] = None
    ) -> Optional[Dict]:
        """
        Load embeddings from Phase 2 outputs and fuse them.
        
        Args:
            video_path: Path to video file
            audio_timestamps: Optional audio segment timestamps
            visual_timestamps: Optional visual frame timestamps
            
        Returns:
            Fusion result dictionary or None if failed
        """
        if not config.FUSION_ENABLED:
            logger.debug("Fusion is disabled, skipping")
            return None
        
        try:
            # Load audio embeddings
            audio_embeddings = self._load_audio_embeddings(video_path)
            if audio_embeddings is None:
                logger.warning("Audio embeddings not found, skipping fusion")
                return None
            
            # Load visual embeddings
            visual_embeddings = self._load_visual_embeddings(video_path)
            if visual_embeddings is None:
                logger.warning("Visual embeddings not found, skipping fusion")
                return None
            
            # Validate embeddings
            if not self._validate_embeddings(audio_embeddings, visual_embeddings):
                logger.warning("Embeddings validation failed, skipping fusion")
                return None
            
            # Perform fusion
            if not self._is_loaded:
                self.load_model()
            
            if self.fusion_pipeline is None:
                logger.warning("Fusion model not available, skipping fusion")
                return None
            
            logger.info(f"Fusing embeddings - Audio: {audio_embeddings.shape}, Visual: {visual_embeddings.shape}")
            
            result = self.fusion_pipeline.fuse(
                audio_embeddings=audio_embeddings,
                visual_embeddings=visual_embeddings,
                timestamps=audio_timestamps,
                align_method=config.FUSION_ALIGNMENT_METHOD,
                return_attention=False
            )
            
            # Save fused embeddings
            self._save_fused_embeddings(video_path, result['fused_embeddings'])
            
            # Save metadata
            self._save_fusion_metadata(video_path, result, audio_timestamps, visual_timestamps)
            
            logger.info(f"Fusion completed successfully - Output shape: {result['fused_embeddings'].shape}")
            return result
            
        except Exception as e:
            logger.error(f"Fusion failed: {e}", exc_info=True)
            return None
    
    def _load_audio_embeddings(self, video_path: str) -> Optional[np.ndarray]:
        """Load audio embeddings from disk."""
        embeddings_dir = get_video_embeddings_dir(video_path)
        npy_path = embeddings_dir / "audio_embeddings.npy"
        
        if not npy_path.exists():
            logger.debug(f"Audio embeddings file not found: {npy_path}")
            return None
        
        try:
            embeddings = np.load(str(npy_path))
            logger.debug(f"Loaded audio embeddings: shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load audio embeddings: {e}")
            return None
    
    def _load_visual_embeddings(self, video_path: str) -> Optional[np.ndarray]:
        """Load visual embeddings from disk."""
        embeddings_dir = get_video_embeddings_dir(video_path)
        npy_path = embeddings_dir / "visual_embeddings.npy"
        
        if not npy_path.exists():
            logger.debug(f"Visual embeddings file not found: {npy_path}")
            return None
        
        try:
            embeddings = np.load(str(npy_path))
            logger.debug(f"Loaded visual embeddings: shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load visual embeddings: {e}")
            return None
    
    def _validate_embeddings(self, audio_emb: np.ndarray, visual_emb: np.ndarray) -> bool:
        """Validate embeddings are compatible for fusion."""
        if audio_emb.size == 0 or visual_emb.size == 0:
            logger.warning("Empty embeddings detected")
            return False
        
        if len(audio_emb.shape) != 2 or len(visual_emb.shape) != 2:
            logger.warning(f"Invalid embedding shapes: audio {audio_emb.shape}, visual {visual_emb.shape}")
            return False
        
        # Check dimensions match expected config
        if audio_emb.shape[1] != config.AUDIO_EMBEDDING_DIM:
            logger.warning(
                f"Audio embedding dim mismatch: expected {config.AUDIO_EMBEDDING_DIM}, "
                f"got {audio_emb.shape[1]}"
            )
            # Don't fail, just warn - model can handle different dims
        
        if visual_emb.shape[1] != config.VISUAL_EMBEDDING_DIM:
            logger.warning(
                f"Visual embedding dim mismatch: expected {config.VISUAL_EMBEDDING_DIM}, "
                f"got {visual_emb.shape[1]}"
            )
            # Don't fail, just warn - model can handle different dims
        
        return True
    
    def _save_fused_embeddings(self, video_path: str, fused_embeddings: np.ndarray):
        """Save fused embeddings to disk."""
        embeddings_dir = get_video_embeddings_dir(video_path)
        npy_path = embeddings_dir / "fused_embeddings.npy"
        
        np.save(str(npy_path), fused_embeddings)
        logger.info(f"Saved fused embeddings to {npy_path}")
    
    def _save_fusion_metadata(
        self,
        video_path: str,
        result: Dict,
        audio_timestamps: Optional[List[float]],
        visual_timestamps: Optional[List[float]]
    ):
        """Save fusion metadata to JSON."""
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "fusion_metadata.json"
        
        metadata = {
            "video_path": video_path,
            "fused_embeddings_shape": list(result['fused_embeddings'].shape),
            "fusion_dim": result['fusion_dim'],
            "alignment_method": config.FUSION_ALIGNMENT_METHOD,
            "audio_timestamps_count": len(audio_timestamps) if audio_timestamps else 0,
            "visual_timestamps_count": len(visual_timestamps) if visual_timestamps else 0,
            "model_config": {
                "hidden_dim": config.FUSION_HIDDEN_DIM,
                "num_layers": config.FUSION_NUM_LAYERS,
                "num_heads": config.FUSION_NUM_HEADS,
                "dropout": config.FUSION_DROPOUT,
                "audio_dim": config.AUDIO_EMBEDDING_DIM,
                "visual_dim": config.VISUAL_EMBEDDING_DIM
            }
        }
        
        save_json(metadata, str(json_path))
        logger.info(f"Saved fusion metadata to {json_path}")

