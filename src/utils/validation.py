"""
Input validation functions for video processing.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from src.utils.file_utils import is_video_file, validate_file_size, get_file_size
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_video_file(file_path: str, check_size: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate video file for processing.
    
    Args:
        file_path: Path to video file
        check_size: Whether to check file size
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    # Check if it's a file (not directory)
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    
    # Check file extension
    if not is_video_file(file_path):
        return False, f"Unsupported video format. Supported formats: {', '.join(['.mp4', '.mkv', '.mov', '.avi', '.webm'])}"
    
    # Check file size
    if check_size:
        if not validate_file_size(file_path):
            file_size_mb = get_file_size(file_path) / (1024 * 1024)
            max_size_mb = 1024  # 1GB
            return False, f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
    
    # Check if file is readable
    try:
        with open(file_path, 'rb') as f:
            f.read(1)
    except IOError as e:
        return False, f"Cannot read file: {str(e)}"
    
    return True, None


def validate_output_path(output_path: str, create_dir: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate output path for writing.
    
    Args:
        output_path: Path for output file or directory
        create_dir: Whether to create parent directory if it doesn't exist
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(output_path)
    
    # Check if parent directory exists or can be created
    parent_dir = path.parent
    if not parent_dir.exists():
        if create_dir:
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return False, f"Cannot create output directory: {str(e)}"
        else:
            return False, f"Output directory does not exist: {parent_dir}"
    
    # Check if parent directory is writable
    if not os.access(parent_dir, os.W_OK):
        return False, f"Output directory is not writable: {parent_dir}"
    
    return True, None


def validate_timestamp(timestamp: float, min_value: float = 0.0, max_value: Optional[float] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate timestamp value.
    
    Args:
        timestamp: Timestamp in seconds
        min_value: Minimum allowed value
        max_value: Maximum allowed value (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(timestamp, (int, float)):
        return False, "Timestamp must be a number"
    
    if timestamp < min_value:
        return False, f"Timestamp ({timestamp}) is less than minimum ({min_value})"
    
    if max_value is not None and timestamp > max_value:
        return False, f"Timestamp ({timestamp}) exceeds maximum ({max_value})"
    
    return True, None


def validate_fps(fps: float) -> Tuple[bool, Optional[str]]:
    """
    Validate frames per second value.
    
    Args:
        fps: Frames per second
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(fps, (int, float)):
        return False, "FPS must be a number"
    
    if fps <= 0:
        return False, "FPS must be greater than 0"
    
    if fps > 60:
        return False, "FPS should not exceed 60 (unrealistic for video processing)"
    
    return True, None


def validate_model_name(model_name: str, allowed_models: list) -> Tuple[bool, Optional[str]]:
    """
    Validate model name.
    
    Args:
        model_name: Model name to validate
        allowed_models: List of allowed model names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(model_name, str):
        return False, "Model name must be a string"
    
    if model_name not in allowed_models:
        return False, f"Model '{model_name}' not in allowed models: {', '.join(allowed_models)}"
    
    return True, None


def validate_job_id(job_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate job ID format.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(job_id, str):
        return False, "Job ID must be a string"
    
    if not job_id:
        return False, "Job ID cannot be empty"
    
    # Allow alphanumeric, hyphens, and underscores
    if not all(c.isalnum() or c in ['-', '_'] for c in job_id):
        return False, "Job ID can only contain alphanumeric characters, hyphens, and underscores"
    
    if len(job_id) > 100:
        return False, "Job ID is too long (max 100 characters)"
    
    return True, None


def validate_config() -> Tuple[bool, Optional[str]]:
    """
    Validate application configuration.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    from config import (
        DATA_DIR, MODELS_DIR, OUTPUT_DIR, TEMP_DIR,
        WHISPER_MODEL, CLIP_MODEL, T5_MODEL
    )
    
    # Check required directories
    directories = [DATA_DIR, MODELS_DIR, OUTPUT_DIR, TEMP_DIR]
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"Cannot create required directory {directory}: {str(e)}"
    
    # Validate model names (basic check)
    whisper_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    if WHISPER_MODEL not in whisper_models:
        logger.warning(f"Whisper model '{WHISPER_MODEL}' may not be valid")
    
    return True, None

