"""
Main Fusion Pipeline for Multimodal Audio-Visual Fusion.

This module provides a high-level interface for fusing audio and visual embeddings
using the multimodal transformer architecture. It handles embedding preparation,
sequence alignment, fusion, and provides utilities for batch processing and
temporal pooling.

Key Features:
- Automatic sequence alignment (interpolation, padding, truncation)
- Batch processing for multiple video segments
- Temporal pooling for sequence length reduction
- Attention weight visualization
- Model saving/loading for pre-trained weights

Example Usage:
    >>> from src.fusion.fusion_pipeline import FusionPipeline
    >>> 
    >>> # Initialize pipeline
    >>> pipeline = FusionPipeline(
    ...     audio_dim=768,
    ...     visual_dim=512,
    ...     hidden_dim=512
    ... )
    >>> 
    >>> # Fuse embeddings
    >>> audio_emb = np.random.randn(100, 768)  # [seq_len, dim]
    >>> visual_emb = np.random.randn(100, 512)
    >>> result = pipeline.fuse(audio_emb, visual_emb)
    >>> 
    >>> # Access fused embeddings
    >>> fused = result['fused_embeddings']  # [seq_len, hidden_dim]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .multimodal_transformer import (
    MultimodalTransformer,
    create_multimodal_transformer,
    align_sequences
)

logger = logging.getLogger(__name__)


class FusionPipeline:
    """
    Complete pipeline for multimodal audio-visual fusion.
    
    This class provides an end-to-end interface for fusing audio and visual embeddings
    using a multimodal transformer. It handles all aspects of the fusion process:
    - Embedding preparation and tensor conversion
    - Sequence alignment (handling different temporal resolutions)
    - Multimodal transformer forward pass
    - Result formatting and extraction
    
    The pipeline can work with:
    - Single video segments
    - Batch processing of multiple segments
    - Pre-trained model weights
    - Different alignment strategies
    
    Args:
        audio_dim: Dimension of input audio embeddings (default: 768 for Whisper)
        visual_dim: Dimension of input visual embeddings (default: 512 for CLIP)
        hidden_dim: Hidden dimension for transformer (default: 512)
        num_layers: Number of transformer layers (default: 4)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate for regularization (default: 0.1)
        device: Computing device - "cuda" or "cpu" (auto-detected)
        model_path: Optional path to pre-trained model checkpoint
    
    Example:
        >>> pipeline = FusionPipeline(audio_dim=768, visual_dim=512)
        >>> result = pipeline.fuse(audio_emb, visual_emb)
        >>> fused_features = result['fused_embeddings']
    """
    
    def __init__(
        self,
        audio_dim: int = 768,
        visual_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: Optional[str] = None
    ):
        """
        Initialize fusion pipeline with multimodal transformer.
        
        Args:
            audio_dim: Input dimension for audio embeddings (typically 768)
            visual_dim: Input dimension for visual embeddings (typically 512)
            hidden_dim: Hidden dimension throughout the transformer
            num_layers: Number of transformer layers (more layers = more capacity)
            num_heads: Number of parallel attention heads
            dropout: Dropout probability for regularization
            device: Device to run computations on ("cuda" or "cpu")
            model_path: Optional path to pre-trained model weights file
        
        Note:
            If model_path is provided and file exists, weights are loaded.
            Otherwise, model is initialized with random weights.
        """
        self.device = torch.device(device)
        logger.info(f"Initializing FusionPipeline on {self.device}")
        
        # Create multimodal transformer
        self.model = create_multimodal_transformer(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        ).to(self.device)
        
        # Load pre-trained weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.info("No pre-trained model loaded. Using random initialization.")
        
        self.model.eval()  # Set to evaluation mode
        
    def load_model(self, model_path: str):
        """
        Load pre-trained model weights from checkpoint file.
        
        Loads the model state dictionary from a saved checkpoint. The checkpoint
        should contain 'model_state_dict' key with the model weights.
        
        Args:
            model_path: Path to the checkpoint file (.pt or .pth)
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint format is invalid or incompatible
        
        Note:
            Model is automatically moved to the correct device (CPU/GPU).
        """
        try:
            # Load checkpoint and map to current device
            checkpoint = torch.load(model_path, map_location=self.device)
            # Load state dictionary into model
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def save_model(self, save_path: str):
        """
        Save model weights and configuration to checkpoint file.
        
        Saves the current model state along with configuration information
        needed to recreate the model architecture.
        
        Args:
            save_path: Path where to save the checkpoint file
        
        Raises:
            IOError: If file cannot be written
        
        Note:
            Checkpoint includes both model weights and configuration for
            easy model reconstruction later.
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),  # Model parameters
                'config': {  # Architecture configuration
                    'hidden_dim': self.model.hidden_dim,
                    'num_layers': self.model.num_layers
                }
            }
            torch.save(checkpoint, save_path)
            logger.info(f"Saved model to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def prepare_embeddings(
        self,
        audio_embeddings: np.ndarray,
        visual_embeddings: np.ndarray,
        timestamps: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare embeddings for fusion.
        
        Args:
            audio_embeddings: Audio embeddings [audio_len, audio_dim]
            visual_embeddings: Visual embeddings [visual_len, visual_dim]
            timestamps: Optional timestamps for alignment
            
        Returns:
            audio_tensor: [1, audio_len, audio_dim]
            visual_tensor: [1, visual_len, visual_dim]
        """
        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_embeddings).float()
        visual_tensor = torch.from_numpy(visual_embeddings).float()
        
        # Add batch dimension
        audio_tensor = audio_tensor.unsqueeze(0)
        visual_tensor = visual_tensor.unsqueeze(0)
        
        # Move to device
        audio_tensor = audio_tensor.to(self.device)
        visual_tensor = visual_tensor.to(self.device)
        
        return audio_tensor, visual_tensor
    
    def fuse(
        self,
        audio_embeddings: np.ndarray,
        visual_embeddings: np.ndarray,
        timestamps: Optional[List[float]] = None,
        align_method: str = "interpolate",
        return_attention: bool = False
    ) -> Dict:
        """
        Fuse audio and visual embeddings.
        
        Args:
            audio_embeddings: Audio embeddings [audio_len, audio_dim] or list of arrays
            visual_embeddings: Visual embeddings [visual_len, visual_dim] or list of arrays
            timestamps: Optional timestamps for alignment
            align_method: Method to align sequences ("interpolate", "pad", "truncate")
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - fused_embeddings: Fused features [seq_len, hidden_dim]
                - attention_weights: Optional attention weights
                - aligned_audio: Aligned audio embeddings
                - aligned_visual: Aligned visual embeddings
        """
        logger.info("Starting multimodal fusion...")
        
        # Use no_grad for inference (faster, less memory)
        with torch.no_grad():
            # Step 1: Prepare embeddings (convert numpy to torch, add batch dim, move to device)
            audio_tensor, visual_tensor = self.prepare_embeddings(
                audio_embeddings,
                visual_embeddings,
                timestamps
            )
            
            logger.info(f"Audio shape: {audio_tensor.shape}, Visual shape: {visual_tensor.shape}")
            
            # Step 2: Align sequences if they have different temporal lengths
            # Audio and visual may have different frame rates (e.g., 30fps vs 1fps)
            if audio_tensor.shape[1] != visual_tensor.shape[1]:
                logger.info(f"Aligning sequences using method: {align_method}")
                audio_tensor, visual_tensor = align_sequences(
                    audio_tensor,
                    visual_tensor,
                    method=align_method  # "interpolate", "pad", or "truncate"
                )
                logger.info(f"Aligned shapes - Audio: {audio_tensor.shape}, Visual: {visual_tensor.shape}")
            
            # Step 3: Forward pass through multimodal transformer
            # The transformer applies cross-attention and self-attention layers
            fused_embeddings, attention_weights = self.model(
                audio_tensor,
                visual_tensor,
                return_attention=return_attention  # Whether to return attention weights for analysis
            )
            
            # Step 4: Convert back to numpy and remove batch dimension
            # [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
            fused_embeddings = fused_embeddings.squeeze(0).cpu().numpy()
            
            # Step 5: Package results
            result = {
                'fused_embeddings': fused_embeddings,  # Main output: fused multimodal features
                'aligned_audio': audio_tensor.squeeze(0).cpu().numpy(),  # Aligned audio (for reference)
                'aligned_visual': visual_tensor.squeeze(0).cpu().numpy(),  # Aligned visual (for reference)
                'fusion_dim': fused_embeddings.shape[-1]  # Output dimension
            }
            
            # Include attention weights if requested (useful for visualization/analysis)
            if return_attention:
                result['attention_weights'] = attention_weights
            
            logger.info(f"Fusion complete. Output shape: {fused_embeddings.shape}")
            
            return result
    
    def fuse_segments(
        self,
        audio_segments: List[np.ndarray],
        visual_segments: List[np.ndarray],
        segment_timestamps: List[Tuple[float, float]],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Fuse multiple segments in batches.
        
        Args:
            audio_segments: List of audio embedding arrays
            visual_segments: List of visual embedding arrays
            segment_timestamps: List of (start, end) timestamps
            batch_size: Batch size for processing
            
        Returns:
            List of fusion results for each segment
        """
        logger.info(f"Fusing {len(audio_segments)} segments in batches of {batch_size}")
        
        results = []
        
        for i in range(0, len(audio_segments), batch_size):
            batch_audio = audio_segments[i:i+batch_size]
            batch_visual = visual_segments[i:i+batch_size]
            batch_timestamps = segment_timestamps[i:i+batch_size]
            
            batch_results = []
            for audio, visual, (start, end) in zip(batch_audio, batch_visual, batch_timestamps):
                result = self.fuse(audio, visual)
                result['start_time'] = start
                result['end_time'] = end
                batch_results.append(result)
            
            results.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(audio_segments) + batch_size - 1)//batch_size}")
        
        return results
    
    def temporal_pooling(
        self,
        embeddings: np.ndarray,
        method: str = "mean",
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply temporal pooling to reduce sequence length.
        
        Args:
            embeddings: Input embeddings [seq_len, dim]
            method: Pooling method ("mean", "max", "attention")
            window_size: Optional window size for local pooling
            
        Returns:
            pooled: Pooled embeddings
        """
        if window_size is None:
            # Global pooling
            if method == "mean":
                return np.mean(embeddings, axis=0, keepdims=True)
            elif method == "max":
                return np.max(embeddings, axis=0, keepdims=True)
            else:
                raise ValueError(f"Unknown pooling method: {method}")
        else:
            # Local pooling with sliding window
            seq_len = embeddings.shape[0]
            pooled = []
            
            for i in range(0, seq_len, window_size):
                window = embeddings[i:i+window_size]
                if method == "mean":
                    pooled.append(np.mean(window, axis=0))
                elif method == "max":
                    pooled.append(np.max(window, axis=0))
            
            return np.stack(pooled)
    
    def get_segment_representations(
        self,
        fused_embeddings: np.ndarray,
        segment_boundaries: List[Tuple[int, int]]
    ) -> List[np.ndarray]:
        """
        Extract segment-level representations from fused embeddings.
        
        Args:
            fused_embeddings: Fused embeddings [seq_len, dim]
            segment_boundaries: List of (start_idx, end_idx) for each segment
            
        Returns:
            List of segment representations
        """
        segments = []
        for start, end in segment_boundaries:
            segment = fused_embeddings[start:end]
            # Mean pooling over segment
            segment_repr = np.mean(segment, axis=0)
            segments.append(segment_repr)
        
        return segments


def create_fusion_pipeline(
    config: Optional[Dict] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> FusionPipeline:
    """
    Factory function to create fusion pipeline with optional config.
    
    Args:
        config: Optional configuration dictionary
        device: Device to run on
        
    Returns:
        Initialized FusionPipeline
    """
    default_config = {
        'audio_dim': 768,
        'visual_dim': 512,
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return FusionPipeline(
        audio_dim=default_config['audio_dim'],
        visual_dim=default_config['visual_dim'],
        hidden_dim=default_config['hidden_dim'],
        num_layers=default_config['num_layers'],
        num_heads=default_config['num_heads'],
        dropout=default_config['dropout'],
        device=device
    )


# Utility function for visualization
def visualize_attention(
    attention_weights: Dict,
    layer_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize attention weights.
    
    Args:
        attention_weights: Dictionary of attention weights from model
        layer_idx: Which layer to visualize
        save_path: Optional path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        layer_attn = attention_weights[layer_idx]
        
        # Get cross-attention weights (audio to visual)
        audio_to_visual = layer_attn['cross_attention']['audio_to_visual']
        
        # Take first head, first batch
        attn_map = audio_to_visual[0, 0].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_map, cmap='viridis', cbar=True)
        plt.title(f'Cross-Attention (Audio→Visual) - Layer {layer_idx}')
        plt.xlabel('Visual Tokens')
        plt.ylabel('Audio Tokens')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib/seaborn not available for visualization")