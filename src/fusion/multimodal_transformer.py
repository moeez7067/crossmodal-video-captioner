"""
Multimodal Transformer for Audio-Visual Fusion.

This module implements a complete transformer architecture for fusing audio and visual
embeddings. It processes multimodal inputs through multiple transformer layers with
cross-modal attention, enabling rich joint representations.

Key Components:
- PositionalEncoding: Adds temporal position information to embeddings
- FeedForward: Position-wise feed-forward network (standard transformer component)
- MultimodalTransformerLayer: Single transformer layer with cross-attention
- MultimodalTransformer: Complete transformer stack for multimodal fusion

Architecture Flow:
    1. Input projection: Audio/visual embeddings → hidden_dim
    2. Positional encoding: Add temporal position information
    3. Transformer layers: Cross-attention + self-attention + feed-forward
    4. Output projection: Final fused representations

Example Usage:
    >>> # Create transformer
    >>> transformer = MultimodalTransformer(
    ...     audio_input_dim=768,
    ...     visual_input_dim=512,
    ...     hidden_dim=512,
    ...     num_layers=4,
    ...     num_heads=8
    ... )
    >>> 
    >>> # Forward pass
    >>> audio = torch.randn(1, 100, 768)  # [batch, seq_len, dim]
    >>> visual = torch.randn(1, 100, 512)
    >>> fused, attn_weights = transformer(audio, visual, return_attention=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from .cross_attention import BidirectionalCrossAttention, create_attention_mask


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    
    Adds temporal position information to embeddings using sinusoidal functions.
    This allows the model to understand the order and relative positions of
    tokens in the sequence, which is crucial for temporal understanding.
    
    The encoding uses different frequencies for different dimensions:
    - Even dimensions: sin(position / 10000^(2i/d_model))
    - Odd dimensions: cos(position / 10000^(2i/d_model))
    
    Args:
        d_model: Model dimension (embedding size)
        max_len: Maximum sequence length to support (default: 5000)
        dropout: Dropout probability (default: 0.1)
    
    Note:
        Positional encodings are pre-computed and stored as buffers (not trainable).
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length to pre-compute encodings for
            dropout: Dropout rate applied after adding positional encoding
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Pre-compute positional encoding matrix for efficiency
        # Shape: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        
        # Compute division term for frequency scaling
        # Different frequencies for different dimensions prevent similar encodings
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sinusoidal functions: sin for even indices, cos for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        pe = pe.unsqueeze(0)  # Add batch dimension: [1, max_len, d_model]
        
        # Register as buffer (not a trainable parameter, but saved with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Standard transformer component that applies the same MLP to each position
    independently. This allows the model to transform representations non-linearly.
    
    Architecture:
        Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model)
        + Residual connection + Layer normalization
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension of feed-forward network (typically 4x d_model)
        dropout: Dropout probability
    
    Note:
        The feed-forward network is applied independently to each position,
        unlike attention which considers all positions together.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension (input and output size)
            d_ff: Feed-forward hidden dimension (typically 4 * d_model)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # Expansion
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # Compression
        self.norm = nn.LayerNorm(d_model)  # Normalization
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor [batch, seq_len, d_model] with same shape as input
        """
        # Save residual for skip connection
        residual = x
        
        # Two-layer MLP: expand → activate → compress
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.dropout(x)
        
        # Residual connection + layer norm (enables deep networks)
        return self.norm(residual + x)


class MultimodalTransformerLayer(nn.Module):
    """
    Single layer of multimodal transformer with cross-attention.
    
    A complete transformer layer that processes multimodal inputs through:
    1. Cross-modal attention: Audio and visual attend to each other
    2. Self-attention: Fused features attend to themselves (captures temporal dependencies)
    3. Feed-forward: Non-linear transformation
    
    This architecture enables the model to:
    - Integrate information across modalities (cross-attention)
    - Model temporal relationships within sequences (self-attention)
    - Apply non-linear transformations (feed-forward)
    
    Args:
        audio_dim: Dimension of audio embeddings
        visual_dim: Dimension of visual embeddings
        hidden_dim: Hidden dimension throughout the layer
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension (typically 4 * hidden_dim)
        dropout: Dropout probability
        fusion_method: Method for fusing modalities ("concat", "add", "gated")
    """
    
    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        fusion_method: str = "concat"
    ):
        """
        Initialize multimodal transformer layer.
        
        Args:
            audio_dim: Input dimension for audio embeddings
            visual_dim: Input dimension for visual embeddings
            hidden_dim: Hidden dimension for all operations in this layer
            num_heads: Number of parallel attention heads
            ff_dim: Feed-forward network hidden dimension
            dropout: Dropout rate for regularization
            fusion_method: How to fuse audio and visual ("concat", "add", "gated")
        """
        super().__init__()
        
        # Cross-modal attention: enables audio-visual interaction
        self.cross_attention = BidirectionalCrossAttention(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            fusion_method=fusion_method
        )
        
        # Self-attention: captures temporal dependencies in fused features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network: non-linear transformation
        self.feed_forward = FeedForward(hidden_dim, ff_dim, dropout)
        
        # Layer normalization for stable training
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through transformer layer.
        
        Args:
            audio_features: [batch, audio_len, audio_dim]
            visual_features: [batch, visual_len, visual_dim]
            audio_mask: Optional mask for audio
            visual_mask: Optional mask for visual
            
        Returns:
            output: Fused features [batch, seq_len, hidden_dim]
            attention_weights: Dictionary of attention weights
        """
        # Step 1: Cross-modal attention
        # Audio and visual attend to each other, creating enriched multimodal features
        fused, cross_attn_weights = self.cross_attention(
            audio_features,
            visual_features,
            audio_mask,
            visual_mask
        )
        
        # Step 2: Self-attention on fused features
        # Captures temporal relationships within the fused sequence
        # Each position can attend to all other positions in the sequence
        residual = fused
        attn_out, self_attn_weights = self.self_attention(
            fused, fused, fused,  # Query, key, value all from fused features
            need_weights=True
        )
        # Residual connection + layer norm (standard transformer pattern)
        fused = self.norm(residual + self.dropout(attn_out))
        
        # Step 3: Feed-forward network
        # Applies non-linear transformation to each position independently
        output = self.feed_forward(fused)
        
        # Collect attention weights
        attention_weights = {
            'cross_attention': cross_attn_weights,
            'self_attention': self_attn_weights
        }
        
        return output, attention_weights


class MultimodalTransformer(nn.Module):
    """
    Complete multimodal transformer for audio-visual fusion.
    Processes audio and visual embeddings through multiple transformer layers.
    """
    
    def __init__(
        self,
        audio_input_dim: int,
        visual_input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        fusion_method: str = "concat"
    ):
        """
        Args:
            audio_input_dim: Dimension of input audio embeddings
            visual_input_dim: Dimension of input visual embeddings
            hidden_dim: Hidden dimension throughout the transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            fusion_method: Method for fusing modalities ("concat", "add", "gated")
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection layers
        self.audio_projection = nn.Linear(audio_input_dim, hidden_dim)
        self.visual_projection = nn.Linear(visual_input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MultimodalTransformerLayer(
                audio_dim=hidden_dim,
                visual_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                fusion_method=fusion_method
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def encode_audio(self, audio_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode audio embeddings.
        
        Args:
            audio_embeddings: [batch, audio_len, audio_input_dim]
            
        Returns:
            encoded: [batch, audio_len, hidden_dim]
        """
        # Project to hidden dimension
        encoded = self.audio_projection(audio_embeddings)
        
        # Add positional encoding
        encoded = self.positional_encoding(encoded)
        
        return encoded
    
    def encode_visual(self, visual_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode visual embeddings.
        
        Args:
            visual_embeddings: [batch, visual_len, visual_input_dim]
            
        Returns:
            encoded: [batch, visual_len, hidden_dim]
        """
        # Project to hidden dimension
        encoded = self.visual_projection(visual_embeddings)
        
        # Add positional encoding
        encoded = self.positional_encoding(encoded)
        
        return encoded
    
    def forward(
        self,
        audio_embeddings: torch.Tensor,
        visual_embeddings: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through multimodal transformer.
        
        Args:
            audio_embeddings: [batch, audio_len, audio_input_dim]
            visual_embeddings: [batch, visual_len, visual_input_dim]
            audio_mask: Optional mask for audio
            visual_mask: Optional mask for visual
            return_attention: Whether to return attention weights
            
        Returns:
            output: Fused multimodal embeddings [batch, seq_len, hidden_dim]
            attention_weights: Optional dict of attention weights from all layers
        """
        # Encode inputs
        audio_encoded = self.encode_audio(audio_embeddings)
        visual_encoded = self.encode_visual(visual_embeddings)
        
        # Store attention weights if requested
        all_attention_weights = [] if return_attention else None
        
        # Pass through transformer layers sequentially
        fused = None
        for layer_idx, layer in enumerate(self.layers):
            # First layer: use original encoded inputs (separate audio and visual)
            # Subsequent layers: use fused output as both inputs (refines fusion)
            if fused is None:
                # Initial layer processes separate modalities
                layer_input_audio = audio_encoded
                layer_input_visual = visual_encoded
            else:
                # Later layers process already-fused features
                # This allows progressive refinement of the multimodal representation
                # Both inputs are the same fused tensor, enabling self-attention to
                # further refine the joint representation
                layer_input_audio = fused
                layer_input_visual = fused
            
            # Process through layer (cross-attention + self-attention + feed-forward)
            fused, attn_weights = layer(
                layer_input_audio,
                layer_input_visual,
                audio_mask,
                visual_mask
            )
            
            # Store attention weights if requested (for visualization/analysis)
            if return_attention:
                all_attention_weights.append(attn_weights)
        
        # Final output projection and normalization
        output = self.output_projection(fused)
        output = self.output_norm(output)
        
        if return_attention:
            return output, all_attention_weights
        else:
            return output, None
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the transformer."""
        return self.hidden_dim


# Example usage and helper functions
def create_multimodal_transformer(
    audio_dim: int = 768,  # e.g., from audio embeddings
    visual_dim: int = 512,  # e.g., from CLIP
    hidden_dim: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1
) -> MultimodalTransformer:
    """
    Factory function to create a multimodal transformer with default settings.
    
    Args:
        audio_dim: Dimension of audio embeddings
        visual_dim: Dimension of visual embeddings
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        
    Returns:
        Initialized MultimodalTransformer
    """
    model = MultimodalTransformer(
        audio_input_dim=audio_dim,
        visual_input_dim=visual_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=hidden_dim * 4,  # Common practice: ff_dim = 4 * hidden_dim
        dropout=dropout,
        fusion_method="concat"
    )
    
    return model


def align_sequences(
    audio_embeddings: torch.Tensor,
    visual_embeddings: torch.Tensor,
    method: str = "interpolate"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align audio and visual sequences to same length.
    
    Args:
        audio_embeddings: [batch, audio_len, audio_dim]
        visual_embeddings: [batch, visual_len, visual_dim]
        method: Alignment method ("interpolate", "pad", "truncate")
        
    Returns:
        aligned_audio: [batch, target_len, audio_dim]
        aligned_visual: [batch, target_len, visual_dim]
    """
    batch_size = audio_embeddings.shape[0]
    audio_len = audio_embeddings.shape[1]
    visual_len = visual_embeddings.shape[1]
    
    if audio_len == visual_len:
        return audio_embeddings, visual_embeddings
    
    if method == "interpolate":
        # Use the longer sequence as target length
        target_len = max(audio_len, visual_len)
        
        # Interpolate shorter sequence
        if audio_len < target_len:
            audio_embeddings = F.interpolate(
                audio_embeddings.transpose(1, 2),
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        if visual_len < target_len:
            visual_embeddings = F.interpolate(
                visual_embeddings.transpose(1, 2),
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
    elif method == "pad":
        # Pad shorter sequence with zeros
        target_len = max(audio_len, visual_len)
        
        if audio_len < target_len:
            padding = torch.zeros(
                batch_size,
                target_len - audio_len,
                audio_embeddings.shape[2],
                device=audio_embeddings.device
            )
            audio_embeddings = torch.cat([audio_embeddings, padding], dim=1)
        
        if visual_len < target_len:
            padding = torch.zeros(
                batch_size,
                target_len - visual_len,
                visual_embeddings.shape[2],
                device=visual_embeddings.device
            )
            visual_embeddings = torch.cat([visual_embeddings, padding], dim=1)
            
    elif method == "truncate":
        # Truncate to shorter sequence length
        target_len = min(audio_len, visual_len)
        audio_embeddings = audio_embeddings[:, :target_len]
        visual_embeddings = visual_embeddings[:, :target_len]
    
    return audio_embeddings, visual_embeddings