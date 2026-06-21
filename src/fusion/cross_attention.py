"""
Cross-Attention Module for Multimodal Fusion.

This module implements cross-attention mechanisms that allow audio and visual features
to attend to each other, enabling rich multimodal representations. The implementation
includes both unidirectional and bidirectional cross-attention.

Key Components:
- CrossAttention: Single-direction cross-attention (query modality attends to key-value modality)
- BidirectionalCrossAttention: Both modalities attend to each other simultaneously

Example Usage:
    >>> # Create bidirectional cross-attention
    >>> cross_attn = BidirectionalCrossAttention(
    ...     audio_dim=768,
    ...     visual_dim=512,
    ...     hidden_dim=512,
    ...     num_heads=8
    ... )
    >>> 
    >>> # Forward pass
    >>> audio_features = torch.randn(1, 100, 768)  # [batch, seq_len, dim]
    >>> visual_features = torch.randn(1, 100, 512)
    >>> fused, attn_weights = cross_attn(audio_features, visual_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CrossAttention(nn.Module):
    """
    Single direction cross-attention module.
    
    Implements scaled dot-product attention where the query modality attends to
    the key-value modality. This allows one modality (e.g., audio) to incorporate
    information from another modality (e.g., visual) through attention mechanisms.
    
    Architecture:
        1. Project query, key, value to hidden_dim
        2. Multi-head attention with scaled dot-product
        3. Apply attention weights to values
        4. Output projection with residual connection and layer norm
    
    Args:
        query_dim: Dimension of query embeddings (e.g., 768 for audio)
        key_dim: Dimension of key/value embeddings (e.g., 512 for visual)
        hidden_dim: Hidden dimension for attention projections (must be divisible by num_heads)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability for regularization (default: 0.1)
    
    Example:
        >>> # Audio attending to visual
        >>> cross_attn = CrossAttention(
        ...     query_dim=768,  # Audio dimension
        ...     key_dim=512,    # Visual dimension
        ...     hidden_dim=512,
        ...     num_heads=8
        ... )
        >>> audio_query = torch.randn(1, 100, 768)
        >>> visual_kv = torch.randn(1, 100, 512)
        >>> output, attn_weights = cross_attn(audio_query, visual_kv, visual_kv)
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention module.
        
        Args:
            query_dim: Dimension of query embeddings (input modality that queries)
            key_dim: Dimension of key/value embeddings (modality being queried)
            hidden_dim: Hidden dimension for attention (must be divisible by num_heads)
            num_heads: Number of parallel attention heads (default: 8)
            dropout: Dropout probability for regularization (default: 0.1)
        
        Raises:
            AssertionError: If hidden_dim is not divisible by num_heads
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Projection layers
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, query_dim)
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with cross-attention.
        
        Args:
            query: Query tensor [batch, query_len, query_dim]
            key: Key tensor [batch, key_len, key_dim]
            value: Value tensor [batch, key_len, key_dim]
            mask: Optional mask [batch, query_len, key_len]
            
        Returns:
            output: Attended output [batch, query_len, query_dim]
            attention_weights: Attention weights [batch, num_heads, query_len, key_len]
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        
        # Save residual connection for later (enables gradient flow)
        residual = query
        
        # Step 1: Project inputs to Q, K, V space
        # Linear projections transform inputs to common hidden dimension
        Q = self.query_proj(query)  # [batch, query_len, hidden_dim]
        K = self.key_proj(key)      # [batch, key_len, hidden_dim]
        V = self.value_proj(value)  # [batch, key_len, hidden_dim]
        
        # Step 2: Reshape for multi-head attention
        # Split hidden_dim into num_heads separate attention computations
        # [batch, seq_len, hidden_dim] -> [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Compute scaled dot-product attention scores
        # Q @ K^T gives similarity scores between query and key positions
        # Scaling by 1/sqrt(head_dim) prevents softmax saturation
        # [batch, num_heads, query_len, head_dim] @ [batch, num_heads, head_dim, key_len]
        # -> [batch, num_heads, query_len, key_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Step 4: Apply attention mask if provided (for padding/valid positions)
        if mask is not None:
            # Expand mask to match attention scores shape (add head dimension)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, query_len, key_len]
            # Mask out invalid positions by setting to -inf (becomes 0 after softmax)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 5: Apply softmax to get attention weights (probabilities)
        # Each query position gets a probability distribution over key positions
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)  # Regularization
        
        # Step 6: Apply attention weights to values
        # Weighted sum of values based on attention scores
        # [batch, num_heads, query_len, key_len] @ [batch, num_heads, key_len, head_dim]
        # -> [batch, num_heads, query_len, head_dim]
        attended = torch.matmul(attention_weights, V)
        
        # Step 7: Reshape back to original format
        # Concatenate all heads back together
        # [batch, num_heads, query_len, head_dim] -> [batch, query_len, hidden_dim]
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.hidden_dim
        )
        
        # Step 8: Output projection and residual connection
        # Project back to query_dim and add residual (helps with training)
        output = self.out_proj(attended)
        output = self.dropout(output)
        
        # Residual connection + layer norm (standard transformer architecture)
        # Enables deep networks to learn effectively
        output = self.layer_norm(residual + output)
        
        return output, attention_weights


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between two modalities.
    
    Implements mutual attention where both audio and visual modalities attend to
    each other simultaneously. This creates rich bidirectional information flow:
    - Audio features are enhanced with visual context
    - Visual features are enhanced with audio context
    - The enhanced features are then fused using the specified method
    
    Architecture:
        1. Audio → Visual: Audio queries visual features
        2. Visual → Audio: Visual queries audio features
        3. Fusion: Combine enhanced features (concat/add/gated)
    
    Args:
        audio_dim: Dimension of audio embeddings (typically 768 for Whisper)
        visual_dim: Dimension of visual embeddings (typically 512 for CLIP)
        hidden_dim: Hidden dimension for attention projections
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        fusion_method: Method to fuse attended features - "concat", "add", or "gated" (default: "concat")
    
    Example:
        >>> # Create bidirectional cross-attention
        >>> cross_attn = BidirectionalCrossAttention(
        ...     audio_dim=768,
        ...     visual_dim=512,
        ...     hidden_dim=512,
        ...     fusion_method="concat"
        ... )
        >>> audio = torch.randn(1, 100, 768)
        >>> visual = torch.randn(1, 100, 512)
        >>> fused, attn_weights = cross_attn(audio, visual)
    """
    
    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_method: str = "concat"  # "concat", "add", "gated"
    ):
        """
        Initialize bidirectional cross-attention module.
        
        Args:
            audio_dim: Dimension of input audio embeddings
            visual_dim: Dimension of input visual embeddings
            hidden_dim: Hidden dimension for attention mechanisms
            num_heads: Number of parallel attention heads
            dropout: Dropout rate for regularization
            fusion_method: How to combine the two attended modalities:
                - "concat": Concatenate and project (most common)
                - "add": Element-wise addition (requires same dimensions)
                - "gated": Learnable gating mechanism for adaptive fusion
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Audio attends to visual
        self.audio_to_visual = CrossAttention(
            query_dim=audio_dim,
            key_dim=visual_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Visual attends to audio
        self.visual_to_audio = CrossAttention(
            query_dim=visual_dim,
            key_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Fusion layers based on method
        if fusion_method == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(audio_dim + visual_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif fusion_method == "gated":
            # Gated fusion - learn what to take from each modality
            self.gate_audio = nn.Linear(audio_dim, audio_dim)
            self.gate_visual = nn.Linear(visual_dim, visual_dim)
            self.fusion_layer = nn.Sequential(
                nn.Linear(audio_dim + visual_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Bidirectional cross-attention between audio and visual.
        
        Args:
            audio_features: [batch, audio_len, audio_dim]
            visual_features: [batch, visual_len, visual_dim]
            audio_mask: Optional mask for audio [batch, audio_len, visual_len]
            visual_mask: Optional mask for visual [batch, visual_len, audio_len]
            
        Returns:
            fused_features: Combined multimodal features [batch, seq_len, hidden_dim]
            attention_weights: Dictionary with attention weights
        """
        # Step 1: Audio attends to visual (audio features enhanced with visual context)
        # Audio queries use visual features as keys/values, allowing audio to "see" visual info
        audio_attended, audio_attn_weights = self.audio_to_visual(
            query=audio_features,
            key=visual_features,
            value=visual_features,
            mask=audio_mask
        )
        
        # Step 2: Visual attends to audio (visual features enhanced with audio context)
        # Visual queries use audio features as keys/values, allowing visual to "hear" audio info
        visual_attended, visual_attn_weights = self.visual_to_audio(
            query=visual_features,
            key=audio_features,
            value=audio_features,
            mask=visual_mask
        )
        
        # Step 3: Align sequence lengths if they differ
        # Audio and visual may have different temporal resolutions (different frame rates)
        # We interpolate the shorter sequence to match the longer one
        if audio_attended.shape[1] != visual_attended.shape[1]:
            # Use audio length as reference (can be changed to visual or max)
            target_len = audio_attended.shape[1]
            # Linear interpolation along temporal dimension
            visual_attended = F.interpolate(
                visual_attended.transpose(1, 2),  # [batch, dim, seq_len] for interpolation
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # Back to [batch, seq_len, dim]
        
        # Step 4: Fuse the two enhanced modalities using specified method
        if self.fusion_method == "concat":
            # Concatenate along feature dimension and project to hidden_dim
            # Preserves information from both modalities
            combined = torch.cat([audio_attended, visual_attended], dim=-1)
            fused_features = self.fusion_layer(combined)
            
        elif self.fusion_method == "add":
            # Element-wise addition (simpler but requires matching dimensions)
            # Assumes both modalities contribute equally
            if audio_attended.shape[-1] == visual_attended.shape[-1]:
                fused_features = audio_attended + visual_attended
            else:
                raise ValueError("Audio and visual dims must match for 'add' fusion")
                
        elif self.fusion_method == "gated":
            # Gated fusion - learnable mechanism to weight each modality
            # Allows model to adaptively decide how much to use from each modality
            gate_a = torch.sigmoid(self.gate_audio(audio_attended))  # [0, 1] weights for audio
            gate_v = torch.sigmoid(self.gate_visual(visual_attended))  # [0, 1] weights for visual
            
            # Apply learned gates (element-wise multiplication)
            gated_audio = gate_a * audio_attended
            gated_visual = gate_v * visual_attended
            
            # Concatenate gated features and project
            combined = torch.cat([gated_audio, gated_visual], dim=-1)
            fused_features = self.fusion_layer(combined)
        
        # Store attention weights for visualization
        attention_weights = {
            'audio_to_visual': audio_attn_weights,
            'visual_to_audio': visual_attn_weights
        }
        
        return fused_features, attention_weights


# Helper function to create attention masks
def create_attention_mask(
    query_len: int,
    key_len: int,
    device: torch.device,
    causal: bool = False
) -> torch.Tensor:
    """
    Create attention mask.
    
    Args:
        query_len: Length of query sequence
        key_len: Length of key sequence
        device: Device to create mask on
        causal: If True, create causal (autoregressive) mask
        
    Returns:
        mask: [query_len, key_len] mask tensor
    """
    if causal:
        # Causal mask - can only attend to past
        mask = torch.tril(torch.ones(query_len, key_len, device=device))
    else:
        # Full attention
        mask = torch.ones(query_len, key_len, device=device)
    
    return mask