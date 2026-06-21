"""
Attention Visualization Tools for Fusion Model.

This module provides tools to visualize attention weights from the fusion model,
showing which parts of audio and visual features are being attended to.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class AttentionVisualizer:
    """
    Visualizes attention weights from fusion model.
    
    Can create:
    - Heatmaps of attention weights
    - Attention flow diagrams
    - Interactive attention maps (HTML)
    """
    
    def __init__(self, output_dir: str = "visualizations/attention"):
        """
        Initialize attention visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_cross_attention(
        self,
        attention_weights: Dict[str, torch.Tensor],
        audio_length: int,
        visual_length: int,
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        save_path: Optional[str] = None,
        title: str = "Cross-Attention Weights"
    ):
        """
        Visualize cross-attention weights between audio and visual.
        
        Args:
            attention_weights: Dictionary with 'audio_to_visual' and 'visual_to_audio' weights
            audio_length: Length of audio sequence
            visual_length: Length of visual sequence
            layer_idx: Layer index (if multiple layers)
            head_idx: Head index (if None, average across heads)
            save_path: Path to save visualization
            title: Title for the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Audio to Visual attention
        if 'audio_to_visual' in attention_weights:
            attn_a2v = attention_weights['audio_to_visual']
            
            # Handle multi-head: [batch, heads, audio_len, visual_len]
            if attn_a2v.dim() == 4:
                if head_idx is not None:
                    attn_a2v = attn_a2v[0, head_idx, :audio_length, :visual_length]
                else:
                    # Average across heads
                    attn_a2v = attn_a2v[0, :, :audio_length, :visual_length].mean(dim=0)
            elif attn_a2v.dim() == 3:
                attn_a2v = attn_a2v[0, :audio_length, :visual_length]
            else:
                attn_a2v = attn_a2v[:audio_length, :visual_length]
            
            # Convert to numpy
            if isinstance(attn_a2v, torch.Tensor):
                attn_a2v = attn_a2v.cpu().numpy()
            
            # Plot heatmap
            sns.heatmap(
                attn_a2v,
                ax=axes[0],
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=False,
                yticklabels=False
            )
            axes[0].set_title('Audio → Visual Attention', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Visual Frames', fontsize=12)
            axes[0].set_ylabel('Audio Segments', fontsize=12)
        
        # Visual to Audio attention
        if 'visual_to_audio' in attention_weights:
            attn_v2a = attention_weights['visual_to_audio']
            
            # Handle multi-head
            if attn_v2a.dim() == 4:
                if head_idx is not None:
                    attn_v2a = attn_v2a[0, head_idx, :visual_length, :audio_length]
                else:
                    attn_v2a = attn_v2a[0, :, :visual_length, :audio_length].mean(dim=0)
            elif attn_v2a.dim() == 3:
                attn_v2a = attn_v2a[0, :visual_length, :audio_length]
            else:
                attn_v2a = attn_v2a[:visual_length, :audio_length]
            
            # Convert to numpy
            if isinstance(attn_v2a, torch.Tensor):
                attn_v2a = attn_v2a.cpu().numpy()
            
            # Plot heatmap
            sns.heatmap(
                attn_v2a,
                ax=axes[1],
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=False,
                yticklabels=False
            )
            axes[1].set_title('Visual → Audio Attention', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Audio Segments', fontsize=12)
            axes[1].set_ylabel('Visual Frames', fontsize=12)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention visualization to {save_path}")
        else:
            save_path = self.output_dir / f"cross_attention_layer_{layer_idx}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention visualization to {save_path}")
        
        plt.close()
        return save_path
    
    def visualize_self_attention(
        self,
        attention_weights: torch.Tensor,
        sequence_length: int,
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        save_path: Optional[str] = None,
        title: str = "Self-Attention Weights"
    ):
        """
        Visualize self-attention weights.
        
        Args:
            attention_weights: Self-attention weights [batch, heads, seq_len, seq_len]
            sequence_length: Length of sequence to visualize
            layer_idx: Layer index
            head_idx: Head index (if None, average across heads)
            save_path: Path to save visualization
            title: Title for the plot
        """
        # Handle multi-head
        if attention_weights.dim() == 4:
            if head_idx is not None:
                attn = attention_weights[0, head_idx, :sequence_length, :sequence_length]
            else:
                attn = attention_weights[0, :, :sequence_length, :sequence_length].mean(dim=0)
        elif attention_weights.dim() == 3:
            attn = attention_weights[0, :sequence_length, :sequence_length]
        else:
            attn = attention_weights[:sequence_length, :sequence_length]
        
        # Convert to numpy
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().numpy()
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attn,
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            square=True,
            xticklabels=False,
            yticklabels=False
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Key Position', fontsize=12)
        plt.ylabel('Query Position', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f"self_attention_layer_{layer_idx}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Saved self-attention visualization to {save_path}")
        return save_path
    
    def visualize_multi_layer_attention(
        self,
        all_attention_weights: List[Dict],
        audio_length: int,
        visual_length: int,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention across multiple layers.
        
        Args:
            all_attention_weights: List of attention weight dictionaries for each layer
            audio_length: Length of audio sequence
            visual_length: Length of visual sequence
            save_path: Path to save visualization
        """
        num_layers = len(all_attention_weights)
        fig, axes = plt.subplots(num_layers, 2, figsize=(16, 4 * num_layers))
        
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx, attn_weights in enumerate(all_attention_weights):
            # Audio to Visual
            if 'audio_to_visual' in attn_weights:
                attn_a2v = attn_weights['audio_to_visual']
                if attn_a2v.dim() == 4:
                    attn_a2v = attn_a2v[0, :, :audio_length, :visual_length].mean(dim=0)
                elif attn_a2v.dim() == 3:
                    attn_a2v = attn_a2v[0, :audio_length, :visual_length]
                
                if isinstance(attn_a2v, torch.Tensor):
                    attn_a2v = attn_a2v.cpu().numpy()
                
                sns.heatmap(
                    attn_a2v,
                    ax=axes[layer_idx, 0],
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Attention'},
                    xticklabels=False,
                    yticklabels=False
                )
                axes[layer_idx, 0].set_title(f'Layer {layer_idx}: Audio → Visual', fontsize=12)
            
            # Visual to Audio
            if 'visual_to_audio' in attn_weights:
                attn_v2a = attn_weights['visual_to_audio']
                if attn_v2a.dim() == 4:
                    attn_v2a = attn_v2a[0, :, :visual_length, :audio_length].mean(dim=0)
                elif attn_v2a.dim() == 3:
                    attn_v2a = attn_v2a[0, :visual_length, :audio_length]
                
                if isinstance(attn_v2a, torch.Tensor):
                    attn_v2a = attn_v2a.cpu().numpy()
                
                sns.heatmap(
                    attn_v2a,
                    ax=axes[layer_idx, 1],
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Attention'},
                    xticklabels=False,
                    yticklabels=False
                )
                axes[layer_idx, 1].set_title(f'Layer {layer_idx}: Visual → Audio', fontsize=12)
        
        plt.suptitle('Attention Weights Across Layers', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / "multi_layer_attention.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Saved multi-layer attention visualization to {save_path}")
        return save_path
    
    def create_interactive_attention_map(
        self,
        attention_weights: Dict[str, torch.Tensor],
        audio_length: int,
        visual_length: int,
        audio_timestamps: Optional[List[float]] = None,
        visual_timestamps: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create interactive HTML attention map.
        
        Args:
            attention_weights: Dictionary with attention weights
            audio_length: Length of audio sequence
            visual_length: Length of visual sequence
            audio_timestamps: Optional timestamps for audio segments
            visual_timestamps: Optional timestamps for visual frames
            save_path: Path to save HTML file
            
        Returns:
            Path to saved HTML file
        """
        # Prepare data
        if 'audio_to_visual' in attention_weights:
            attn_a2v = attention_weights['audio_to_visual']
            if attn_a2v.dim() == 4:
                attn_a2v = attn_a2v[0, :, :audio_length, :visual_length].mean(dim=0)
            elif attn_a2v.dim() == 3:
                attn_a2v = attn_a2v[0, :audio_length, :visual_length]
            
            if isinstance(attn_a2v, torch.Tensor):
                attn_a2v = attn_a2v.cpu().numpy()
        else:
            attn_a2v = None
        
        if 'visual_to_audio' in attention_weights:
            attn_v2a = attention_weights['visual_to_audio']
            if attn_v2a.dim() == 4:
                attn_v2a = attn_v2a[0, :, :visual_length, :audio_length].mean(dim=0)
            elif attn_v2a.dim() == 3:
                attn_v2a = attn_v2a[0, :visual_length, :audio_length]
            
            if isinstance(attn_v2a, torch.Tensor):
                attn_v2a = attn_v2a.cpu().numpy()
        else:
            attn_v2a = None
        
        # Generate timestamps if not provided
        if audio_timestamps is None:
            audio_timestamps = list(range(audio_length))
        if visual_timestamps is None:
            visual_timestamps = list(range(visual_length))
        
        # Create HTML
        html_content = self._generate_interactive_html(
            attn_a2v, attn_v2a,
            audio_timestamps, visual_timestamps
        )
        
        if save_path is None:
            save_path = self.output_dir / "interactive_attention_map.html"
        else:
            save_path = Path(save_path)
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Saved interactive attention map to {save_path}")
        return str(save_path)
    
    def _generate_interactive_html(
        self,
        attn_a2v: Optional[np.ndarray],
        attn_v2a: Optional[np.ndarray],
        audio_timestamps: List[float],
        visual_timestamps: List[float]
    ) -> str:
        """Generate interactive HTML content."""
        # Convert to JSON-serializable format
        if attn_a2v is not None:
            attn_a2v_data = attn_a2v.tolist()
        else:
            attn_a2v_data = None
        
        if attn_v2a is not None:
            attn_v2a_data = attn_v2a.tolist()
        else:
            attn_v2a_data = None
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Attention Map</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .plot-container {{
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Attention Map</h1>
        
        <div class="plot-container">
            <h2>Audio → Visual Attention</h2>
            <div id="plot-a2v"></div>
        </div>
        
        <div class="plot-container">
            <h2>Visual → Audio Attention</h2>
            <div id="plot-v2a"></div>
        </div>
    </div>
    
    <script>
        // Audio to Visual
        var attnA2V = {json.dumps(attn_a2v_data)};
        var audioTimestamps = {json.dumps(audio_timestamps)};
        var visualTimestamps = {json.dumps(visual_timestamps)};
        
        var traceA2V = {{
            z: attnA2V,
            type: 'heatmap',
            colorscale: 'YlOrRd',
            colorbar: {{ title: 'Attention Weight' }},
            hovertemplate: 'Audio: %{{y}}<br>Visual: %{{x}}<br>Attention: %{{z}}<extra></extra>'
        }};
        
        var layoutA2V = {{
            title: 'Audio → Visual Attention',
            xaxis: {{ title: 'Visual Frames' }},
            yaxis: {{ title: 'Audio Segments' }},
            width: 800,
            height: 600
        }};
        
        Plotly.newPlot('plot-a2v', [traceA2V], layoutA2V);
        
        // Visual to Audio
        var attnV2A = {json.dumps(attn_v2a_data)};
        
        var traceV2A = {{
            z: attnV2A,
            type: 'heatmap',
            colorscale: 'YlOrRd',
            colorbar: {{ title: 'Attention Weight' }},
            hovertemplate: 'Visual: %{{y}}<br>Audio: %{{x}}<br>Attention: %{{z}}<extra></extra>'
        }};
        
        var layoutV2A = {{
            title: 'Visual → Audio Attention',
            xaxis: {{ title: 'Audio Segments' }},
            yaxis: {{ title: 'Visual Frames' }},
            width: 800,
            height: 600
        }};
        
        Plotly.newPlot('plot-v2a', [traceV2A], layoutV2A);
    </script>
</body>
</html>
"""
        return html
    
    def visualize_attention_head_comparison(
        self,
        attention_weights: torch.Tensor,
        audio_length: int,
        visual_length: int,
        num_heads: int,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention weights for each head separately.
        
        Args:
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            audio_length: Length of audio sequence
            visual_length: Length of visual sequence
            num_heads: Number of attention heads
            save_path: Path to save visualization
        """
        # Determine grid size
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for head_idx in range(num_heads):
            row = head_idx // cols
            col = head_idx % cols
            
            # Extract head attention
            if attention_weights.dim() == 4:
                attn = attention_weights[0, head_idx, :audio_length, :visual_length]
            else:
                attn = attention_weights[head_idx, :audio_length, :visual_length]
            
            if isinstance(attn, torch.Tensor):
                attn = attn.cpu().numpy()
            
            # Plot
            sns.heatmap(
                attn,
                ax=axes[row, col],
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention'},
                xticklabels=False,
                yticklabels=False
            )
            axes[row, col].set_title(f'Head {head_idx}', fontsize=10)
        
        # Hide unused subplots
        for idx in range(num_heads, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Attention Weights by Head', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / "attention_heads_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Saved attention head comparison to {save_path}")
        return save_path
