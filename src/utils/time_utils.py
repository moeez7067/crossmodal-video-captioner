"""
Timestamp conversion and formatting utilities.
"""

from typing import Union
from datetime import timedelta


def seconds_to_timestamp(seconds: float, format_type: str = "srt") -> str:
    """
    Convert seconds to timestamp string.
    
    Args:
        seconds: Time in seconds
        format_type: Format type ('srt' or 'vtt')
        
    Returns:
        Formatted timestamp string
    """
    if format_type.lower() == "srt":
        return seconds_to_srt_timestamp(seconds)
    elif format_type.lower() == "vtt":
        return seconds_to_vtt_timestamp(seconds)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def seconds_to_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        SRT formatted timestamp (e.g., "00:01:23,456")
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """
    Convert seconds to VTT timestamp format (HH:MM:SS.mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        VTT formatted timestamp (e.g., "00:01:23.456")
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert timestamp string to seconds.
    Supports both SRT (HH:MM:SS,mmm) and VTT (HH:MM:SS.mmm) formats.
    
    Args:
        timestamp: Timestamp string
        
    Returns:
        Time in seconds
    """
    # Handle both comma and dot separators
    if "," in timestamp:
        timestamp = timestamp.replace(",", ".")
    
    parts = timestamp.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split(".")
    seconds = int(seconds_parts[0])
    milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
    
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total_seconds


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    td = timedelta(seconds=int(seconds))
    hours = int(td.total_seconds()) // 3600
    minutes = (int(td.total_seconds()) % 3600) // 60
    secs = int(td.total_seconds()) % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def align_timestamps(segment_timestamps: list, target_timestamps: list) -> dict:
    """
    Align segment timestamps with target timestamps.
    
    Args:
        segment_timestamps: List of (start, end) tuples for segments
        target_timestamps: List of target timestamps
        
    Returns:
        Dictionary mapping target timestamps to segment indices
    """
    alignment = {}
    segment_idx = 0
    
    for target_time in target_timestamps:
        # Find segment that contains this timestamp
        while segment_idx < len(segment_timestamps):
            start, end = segment_timestamps[segment_idx]
            if start <= target_time <= end:
                alignment[target_time] = segment_idx
                break
            elif target_time < start:
                # Target is before current segment
                alignment[target_time] = segment_idx if segment_idx > 0 else 0
                break
            segment_idx += 1
        else:
            # No segment found, use last segment
            alignment[target_time] = len(segment_timestamps) - 1
    
    return alignment


def merge_overlapping_segments(segments: list, threshold: float = 0.1) -> list:
    """
    Merge segments that overlap or are very close together.
    
    Args:
        segments: List of segments with (start, end, ...) format
        threshold: Time threshold in seconds for merging
        
    Returns:
        Merged segments list
    """
    if not segments:
        return []
    
    merged = [segments[0]]
    
    for current in segments[1:]:
        last = merged[-1]
        last_end = last[1] if isinstance(last, (list, tuple)) else getattr(last, 'end_time', 0)
        current_start = current[0] if isinstance(current, (list, tuple)) else getattr(current, 'start_time', 0)
        
        if current_start - last_end <= threshold:
            # Merge segments
            current_end = current[1] if isinstance(current, (list, tuple)) else getattr(current, 'end_time', 0)
            if isinstance(merged[-1], (list, tuple)):
                merged[-1] = (merged[-1][0], max(merged[-1][1], current_end), *merged[-1][2:])
            else:
                merged[-1].end_time = max(last_end, current_end)
        else:
            merged.append(current)
    
    return merged

