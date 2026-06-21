"""
Script to visualize attention weights from fusion model.

Usage:
    python scripts/visualize_attention.py <video_name> [--output_dir <dir>] [--layer <idx>] [--head <idx>]
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.fusion_pipeline import FusionPipeline
from src.fusion.attention_visualization import AttentionVisualizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_embeddings(video_name: str, data_dir: str = "data"):
    """Load audio and visual embeddings for a video."""
    data_path = Path(data_dir) / video_name / "embeddings"
    
    audio_path = data_path / "audio_embeddings.npy"
    visual_path = data_path / "visual_embeddings.npy"
    
    if not audio_path.exists() or not visual_path.exists():
        raise FileNotFoundError(f"Embeddings not found for {video_name}")
    
    audio_emb = np.load(audio_path)
    visual_emb = np.load(visual_path)
    
    logger.info(f"Loaded embeddings: audio {audio_emb.shape}, visual {visual_emb.shape}")
    
    return audio_emb, visual_emb


def visualize_attention_for_video(
    video_name: str,
    output_dir: str = "visualizations/attention",
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    data_dir: str = "data",
    model_path: Optional[str] = None
):
    """
    Visualize attention for a specific video.
    
    Args:
        video_name: Name of video to visualize
        output_dir: Directory to save visualizations
        layer_idx: Layer index to visualize
        head_idx: Head index to visualize (None for average)
        data_dir: Directory containing video data
        model_path: Optional path to trained model
    """
    # Load embeddings
    audio_emb, visual_emb = load_embeddings(video_name, data_dir)
    
    # Convert to tensors
    audio_tensor = torch.from_numpy(audio_emb).float().unsqueeze(0)  # [1, seq_len, dim]
    visual_tensor = torch.from_numpy(visual_emb).float().unsqueeze(0)  # [1, seq_len, dim]
    
    # Create fusion pipeline
    pipeline = FusionPipeline(model_path=model_path)
    
    # Forward pass with attention
    with torch.no_grad():
        # We need to modify the pipeline to return attention
        # For now, use the model directly
        from src.fusion.multimodal_transformer import MultimodalTransformer
        
        model = pipeline.model
        model.eval()
        
        # Get attention weights
        fused, all_attention = model(
            audio_tensor,
            visual_tensor,
            return_attention=True
        )
    
    # Create visualizer
    visualizer = AttentionVisualizer(output_dir=output_dir)
    
    # Visualize
    if all_attention and len(all_attention) > layer_idx:
        layer_attention = all_attention[layer_idx]
        
        # Cross-attention visualization
        if 'cross_attention' in layer_attention:
            cross_attn = layer_attention['cross_attention']
            
            save_path = Path(output_dir) / f"{video_name}_cross_attention_layer_{layer_idx}.png"
            visualizer.visualize_cross_attention(
                cross_attn,
                audio_length=audio_emb.shape[0],
                visual_length=visual_emb.shape[0],
                layer_idx=layer_idx,
                head_idx=head_idx,
                save_path=str(save_path),
                title=f"Cross-Attention: {video_name} (Layer {layer_idx})"
            )
        
        # Self-attention visualization
        if 'self_attention' in layer_attention:
            self_attn = layer_attention['self_attention']
            
            # Handle tuple from MultiheadAttention
            if isinstance(self_attn, tuple):
                self_attn = self_attn[1]  # Get attention weights
            
            save_path = Path(output_dir) / f"{video_name}_self_attention_layer_{layer_idx}.png"
            visualizer.visualize_self_attention(
                self_attn,
                sequence_length=min(audio_emb.shape[0], visual_emb.shape[0]),
                layer_idx=layer_idx,
                head_idx=head_idx,
                save_path=str(save_path),
                title=f"Self-Attention: {video_name} (Layer {layer_idx})"
            )
        
        # Interactive map
        if 'cross_attention' in layer_attention:
            cross_attn = layer_attention['cross_attention']
            html_path = Path(output_dir) / f"{video_name}_interactive_attention.html"
            visualizer.create_interactive_attention_map(
                cross_attn,
                audio_length=audio_emb.shape[0],
                visual_length=visual_emb.shape[0],
                save_path=str(html_path)
            )
            logger.info(f"Interactive map saved to {html_path}")
    
    # Multi-layer visualization
    if all_attention:
        save_path = Path(output_dir) / f"{video_name}_multi_layer_attention.png"
        visualizer.visualize_multi_layer_attention(
            all_attention,
            audio_length=audio_emb.shape[0],
            visual_length=visual_emb.shape[0],
            save_path=str(save_path)
        )
    
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize attention weights from fusion model")
    parser.add_argument("video_name", help="Name of video to visualize")
    parser.add_argument("--output_dir", default="visualizations/attention", help="Output directory")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--model_path", default=None, help="Path to trained model")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to visualize")
    parser.add_argument("--head", type=int, default=None, help="Head index to visualize (None for average)")
    
    args = parser.parse_args()
    
    visualize_attention_for_video(
        args.video_name,
        args.output_dir,
        args.layer,
        args.head,
        args.data_dir,
        args.model_path
    )


if __name__ == "__main__":
    main()
