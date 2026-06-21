"""
Utility functions for multimodal fusion operations.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import interpolate
from src.utils.logger import get_logger

logger = get_logger(__name__)


def align_timestamps(
    audio_timestamps: List[float],
    visual_timestamps: List[float],
    method: str = "interpolate"
) -> Dict:
    """
    Align audio and visual timestamps.
    
    Args:
        audio_timestamps: List of audio segment timestamps
        visual_timestamps: List of visual frame timestamps
        method: Alignment method ("interpolate", "nearest", "pool")
        
    Returns:
        Dictionary with alignment mapping:
        {
            'audio_indices': List[int],  # Indices of audio timestamps
            'visual_indices': List[int],  # Indices of visual timestamps
            'alignment_map': Dict[int, int]  # audio_idx -> visual_idx mapping
        }
    """
    if not audio_timestamps or not visual_timestamps:
        return {
            'audio_indices': [],
            'visual_indices': [],
            'alignment_map': {}
        }
    
    audio_timestamps = np.array(audio_timestamps)
    visual_timestamps = np.array(visual_timestamps)
    
    alignment_map = {}
    audio_indices = []
    visual_indices = []
    
    if method == "nearest":
        # Find nearest visual timestamp for each audio timestamp
        for i, audio_ts in enumerate(audio_timestamps):
            nearest_idx = np.argmin(np.abs(visual_timestamps - audio_ts))
            alignment_map[i] = int(nearest_idx)
            audio_indices.append(i)
            visual_indices.append(nearest_idx)
    
    elif method == "interpolate":
        # Use interpolation to find corresponding visual frames
        for i, audio_ts in enumerate(audio_timestamps):
            # Find the two nearest visual timestamps
            diffs = np.abs(visual_timestamps - audio_ts)
            nearest_idx = np.argmin(diffs)
            alignment_map[i] = int(nearest_idx)
            audio_indices.append(i)
            visual_indices.append(nearest_idx)
    
    elif method == "pool":
        # Pool visual frames for each audio segment
        for i, audio_ts in enumerate(audio_timestamps):
            # Find all visual frames within this audio segment
            if i < len(audio_timestamps) - 1:
                next_audio_ts = audio_timestamps[i + 1]
            else:
                next_audio_ts = audio_ts + (audio_ts - audio_timestamps[i - 1] if i > 0 else 1.0)
            
            # Find visual frames in range
            in_range = (visual_timestamps >= audio_ts) & (visual_timestamps < next_audio_ts)
            if np.any(in_range):
                # Use mean index of frames in range
                visual_idx = int(np.mean(np.where(in_range)[0]))
            else:
                # Fallback to nearest
                visual_idx = int(np.argmin(np.abs(visual_timestamps - audio_ts)))
            
            alignment_map[i] = visual_idx
            audio_indices.append(i)
            visual_indices.append(visual_idx)
    
    return {
        'audio_indices': audio_indices,
        'visual_indices': visual_indices,
        'alignment_map': alignment_map
    }


def align_embeddings_with_timestamps(
    audio_embeddings: np.ndarray,
    audio_timestamps: List[float],
    visual_embeddings: np.ndarray,
    visual_timestamps: List[float],
    method: str = "interpolate"
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Align audio and visual embeddings temporally.
    
    This function aligns embeddings from different modalities (audio and visual)
    that have different temporal sampling rates. It ensures both embeddings
    are aligned to the same temporal points for fusion.
    
    Args:
        audio_embeddings: Audio embeddings array [num_audio_segments, audio_dim]
        audio_timestamps: List of timestamps for audio embeddings
        visual_embeddings: Visual embeddings array [num_frames, visual_dim]
        visual_timestamps: List of timestamps for visual embeddings
        method: Alignment method ("interpolate", "nearest", "pool")
        
    Returns:
        Tuple of (aligned_audio, aligned_visual, aligned_times):
        - aligned_audio: Audio embeddings aligned to common timestamps [num_aligned, audio_dim]
        - aligned_visual: Visual embeddings aligned to common timestamps [num_aligned, visual_dim]
        - aligned_times: Common timestamps used for alignment
    """
    # Validate inputs - ensure they are numpy arrays
    if not isinstance(audio_embeddings, np.ndarray):
        raise TypeError(
            f"audio_embeddings must be numpy array, got {type(audio_embeddings)}. "
            f"Value preview: {str(audio_embeddings)[:100] if audio_embeddings is not None else None}"
        )
    if not isinstance(visual_embeddings, np.ndarray):
        raise TypeError(
            f"visual_embeddings must be numpy array, got {type(visual_embeddings)}. "
            f"Value preview: {str(visual_embeddings)[:100] if visual_embeddings is not None else None}"
        )
    
    # Ensure 2D arrays
    if len(audio_embeddings.shape) != 2:
        raise ValueError(
            f"audio_embeddings must be 2D array [num_segments, dim], got shape {audio_embeddings.shape}"
        )
    if len(visual_embeddings.shape) != 2:
        raise ValueError(
            f"visual_embeddings must be 2D array [num_frames, dim], got shape {visual_embeddings.shape}"
        )
    
    if len(audio_embeddings) == 0 or len(visual_embeddings) == 0:
        logger.warning("Empty embeddings provided for alignment")
        return np.array([]), np.array([]), []
    
    if len(audio_embeddings) != len(audio_timestamps):
        raise ValueError(
            f"Audio embeddings ({len(audio_embeddings)}) and timestamps ({len(audio_timestamps)}) mismatch"
        )
    
    if len(visual_embeddings) != len(visual_timestamps):
        raise ValueError(
            f"Visual embeddings ({len(visual_embeddings)}) and timestamps ({len(visual_timestamps)}) mismatch"
        )
    
    # Determine common timestamps (use audio timestamps as reference)
    aligned_times = audio_timestamps.copy()
    
    if method == "interpolate":
        # Interpolate visual embeddings to match audio timestamps
        aligned_visual = interpolate_embeddings(
            visual_embeddings,
            visual_timestamps,
            aligned_times,
            method="linear"
        )
        aligned_audio = audio_embeddings  # Already at correct timestamps
    
    elif method == "nearest":
        # Find nearest visual embedding for each audio timestamp
        aligned_visual = []
        visual_timestamps_arr = np.array(visual_timestamps)
        
        for audio_ts in aligned_times:
            nearest_idx = np.argmin(np.abs(visual_timestamps_arr - audio_ts))
            aligned_visual.append(visual_embeddings[nearest_idx])
        
        aligned_visual = np.array(aligned_visual)
        aligned_audio = audio_embeddings
    
    elif method == "pool":
        # Pool visual embeddings for each audio segment
        aligned_visual = pool_embeddings(
            visual_embeddings,
            visual_timestamps,
            aligned_times,
            method="mean"
        )
        aligned_audio = audio_embeddings
    
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    logger.debug(
        f"Aligned embeddings: audio {aligned_audio.shape}, visual {aligned_visual.shape}, "
        f"times {len(aligned_times)}"
    )
    
    return aligned_audio, aligned_visual, aligned_times


def interpolate_embeddings(
    embeddings: np.ndarray,
    source_timestamps: List[float],
    target_timestamps: List[float],
    method: str = "linear"
) -> np.ndarray:
    """
    Interpolate embeddings to match target timestamps.
    
    Args:
        embeddings: Source embeddings [num_source, embedding_dim]
        source_timestamps: Timestamps for source embeddings
        target_timestamps: Target timestamps to interpolate to
        method: Interpolation method ("linear", "cubic")
        
    Returns:
        Interpolated embeddings [num_target, embedding_dim]
    """
    if len(embeddings) == 0:
        return np.array([])
    
    if len(source_timestamps) != len(embeddings):
        raise ValueError(
            f"Timestamp count ({len(source_timestamps)}) doesn't match "
            f"embedding count ({len(embeddings)})"
        )
    
    source_timestamps = np.array(source_timestamps)
    target_timestamps = np.array(target_timestamps)
    embedding_dim = embeddings.shape[1]
    
    # Interpolate each dimension separately
    interpolated = np.zeros((len(target_timestamps), embedding_dim))
    
    for dim in range(embedding_dim):
        if method == "linear":
            interp_func = interpolate.interp1d(
                source_timestamps,
                embeddings[:, dim],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
        elif method == "cubic":
            if len(source_timestamps) < 4:
                # Fallback to linear if not enough points
                interp_func = interpolate.interp1d(
                    source_timestamps,
                    embeddings[:, dim],
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
            else:
                interp_func = interpolate.interp1d(
                    source_timestamps,
                    embeddings[:, dim],
                    kind='cubic',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        interpolated[:, dim] = interp_func(target_timestamps)
    
    return interpolated


def pool_embeddings(
    embeddings: np.ndarray,
    source_timestamps: List[float],
    target_timestamps: List[float],
    method: str = "mean"
) -> np.ndarray:
    """
    Pool embeddings to match target timestamps.
    
    Args:
        embeddings: Source embeddings [num_source, embedding_dim]
        source_timestamps: Timestamps for source embeddings
        target_timestamps: Target timestamps (segment boundaries)
        method: Pooling method ("mean", "max", "attention")
        
    Returns:
        Pooled embeddings [num_target, embedding_dim]
    """
    if len(embeddings) == 0:
        return np.array([])
    
    if len(source_timestamps) != len(embeddings):
        raise ValueError(
            f"Timestamp count ({len(source_timestamps)}) doesn't match "
            f"embedding count ({len(embeddings)})"
        )
    
    source_timestamps = np.array(source_timestamps)
    target_timestamps = np.array(target_timestamps)
    embedding_dim = embeddings.shape[1]
    
    pooled = []
    
    for i in range(len(target_timestamps)):
        start_time = target_timestamps[i]
        end_time = target_timestamps[i + 1] if i < len(target_timestamps) - 1 else source_timestamps[-1]
        
        # Find embeddings in this time range
        in_range = (source_timestamps >= start_time) & (source_timestamps < end_time)
        
        if not np.any(in_range):
            # No embeddings in range, use nearest
            nearest_idx = np.argmin(np.abs(source_timestamps - start_time))
            pooled.append(embeddings[nearest_idx])
        else:
            # Pool embeddings in range
            range_embeddings = embeddings[in_range]
            
            if method == "mean":
                pooled_emb = np.mean(range_embeddings, axis=0)
            elif method == "max":
                pooled_emb = np.max(range_embeddings, axis=0)
            elif method == "attention":
                # Simple attention: weighted mean (could be enhanced)
                pooled_emb = np.mean(range_embeddings, axis=0)
            else:
                raise ValueError(f"Unknown pooling method: {method}")
            
            pooled.append(pooled_emb)
    
    return np.array(pooled)


def validate_embeddings(audio_emb: np.ndarray, visual_emb: np.ndarray) -> bool:
    """
    Validate embeddings are compatible for fusion.
    
    Args:
        audio_emb: Audio embeddings array
        visual_emb: Visual embeddings array
        
    Returns:
        True if embeddings are valid for fusion
    """
    if audio_emb.size == 0 or visual_emb.size == 0:
        logger.warning("Empty embeddings detected")
        return False
    
    if len(audio_emb.shape) != 2 or len(visual_emb.shape) != 2:
        logger.warning(
            f"Invalid embedding shapes: audio {audio_emb.shape}, visual {visual_emb.shape}"
        )
        return False
    
    # Check for NaN or Inf values
    if np.any(np.isnan(audio_emb)) or np.any(np.isinf(audio_emb)):
        logger.warning("Audio embeddings contain NaN or Inf values")
        return False
    
    if np.any(np.isnan(visual_emb)) or np.any(np.isinf(visual_emb)):
        logger.warning("Visual embeddings contain NaN or Inf values")
        return False
    
    return True


def normalize_embeddings(embeddings: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize embeddings to unit length.
    
    Args:
        embeddings: Embeddings array [num_embeddings, embedding_dim]
        method: Normalization method ("l2", "minmax", "zscore")
        
    Returns:
        Normalized embeddings
    """
    if embeddings.size == 0:
        return embeddings
    
    if method == "l2":
        # L2 normalization (unit length)
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
        return embeddings / norms
    
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_vals = np.min(embeddings, axis=0, keepdims=True)
        max_vals = np.max(embeddings, axis=0, keepdims=True)
        ranges = max_vals - min_vals
        ranges = np.where(ranges == 0, 1.0, ranges)  # Avoid division by zero
        return (embeddings - min_vals) / ranges
    
    elif method == "zscore":
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(embeddings, axis=0, keepdims=True)
        std = np.std(embeddings, axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
        return (embeddings - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def pad_sequences(
    sequences: List[np.ndarray],
    max_length: Optional[int] = None,
    pad_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of embedding sequences
        max_length: Maximum length (if None, use longest sequence)
        pad_value: Value to use for padding
        
    Returns:
        Tuple of (padded_sequences, mask)
        - padded_sequences: [batch_size, max_length, embedding_dim]
        - mask: [batch_size, max_length] boolean mask (True for real, False for padding)
    """
    if not sequences:
        return np.array([]), np.array([])
    
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    embedding_dim = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
    batch_size = len(sequences)
    
    padded = np.full((batch_size, max_length, embedding_dim), pad_value, dtype=np.float32)
    mask = np.zeros((batch_size, max_length), dtype=bool)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len > max_length:
            # Truncate if too long
            padded[i] = seq[:max_length]
            mask[i] = True
        else:
            padded[i, :seq_len] = seq
            mask[i, :seq_len] = True
    
    return padded, mask


def create_attention_mask_from_lengths(lengths: List[int], max_length: Optional[int] = None) -> torch.Tensor:
    """
    Create attention mask from sequence lengths.
    
    Args:
        lengths: List of sequence lengths
        max_length: Maximum length (if None, use max of lengths)
        
    Returns:
        Attention mask tensor [batch_size, max_length]
        True for valid positions, False for padding
    """
    if max_length is None:
        max_length = max(lengths) if lengths else 0
    
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask


def check_compatibility(audio_dim: int, visual_dim: int, expected_audio_dim: int, expected_visual_dim: int) -> bool:
    """
    Check if embedding dimensions are compatible with expected dimensions.
    
    Args:
        audio_dim: Actual audio embedding dimension
        visual_dim: Actual visual embedding dimension
        expected_audio_dim: Expected audio embedding dimension
        expected_visual_dim: Expected visual embedding dimension
        
    Returns:
        True if compatible (warnings logged if not)
    """
    compatible = True
    
    if audio_dim != expected_audio_dim:
        logger.warning(
            f"Audio dimension mismatch: expected {expected_audio_dim}, got {audio_dim}. "
            "Model will project to expected dimension."
        )
        # Not a failure, model can handle it
    
    if visual_dim != expected_visual_dim:
        logger.warning(
            f"Visual dimension mismatch: expected {expected_visual_dim}, got {visual_dim}. "
            "Model will project to expected dimension."
        )
        # Not a failure, model can handle it
    
    return compatible

