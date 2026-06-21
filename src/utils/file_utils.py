"""
File handling utilities for video processing.
"""

import os
import shutil
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Any, Dict
from config import TEMP_DIR, OUTPUT_DIR, DATA_DIR, SUPPORTED_VIDEO_FORMATS, MAX_UPLOAD_SIZE


def get_file_extension(file_path: str) -> str:
    """
    Get file extension from file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension (e.g., '.mp4')
    """
    return Path(file_path).suffix.lower()


def is_video_file(file_path: str) -> bool:
    """
    Check if file is a supported video format.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is a supported video format
    """
    extension = get_file_extension(file_path)
    return extension in SUPPORTED_VIDEO_FORMATS


def validate_file_size(file_path: str, max_size: Optional[int] = None) -> bool:
    """
    Validate file size is within limits.
    
    Args:
        file_path: Path to file
        max_size: Maximum file size in bytes (defaults to MAX_UPLOAD_SIZE)
        
    Returns:
        True if file size is valid
    """
    if max_size is None:
        max_size = MAX_UPLOAD_SIZE
    
    file_size = os.path.getsize(file_path)
    return file_size <= max_size


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    return get_file_size(file_path) / (1024 * 1024)


def create_temp_file(prefix: str = "video_", suffix: str = ".mp4") -> Path:
    """
    Create a temporary file path.
    
    Args:
        prefix: File prefix
        suffix: File suffix
        
    Returns:
        Path to temporary file
    """
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=suffix,
        dir=TEMP_DIR,
        delete=False
    )
    temp_file.close()
    return Path(temp_file.name)


def create_output_dir(job_id: str) -> Path:
    """
    Create output directory for a processing job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Path to output directory
    """
    output_path = OUTPUT_DIR / job_id
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def cleanup_temp_files(file_paths: List[Path]) -> None:
    """
    Clean up temporary files.
    
    Args:
        file_paths: List of file paths to delete
    """
    for file_path in file_paths:
        try:
            if file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
        except Exception as e:
            # Log error but don't raise
            print(f"Error cleaning up {file_path}: {e}")


def get_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def ensure_directory(directory: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_files(directory: Path, pattern: str = "*") -> List[Path]:
    """
    List files in directory matching pattern.
    
    Args:
        directory: Directory path
        pattern: Glob pattern (e.g., "*.mp4")
        
    Returns:
        List of file paths
    """
    if not directory.exists():
        return []
    return list(directory.glob(pattern))


def copy_file(source: Path, destination: Path) -> Path:
    """
    Copy file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        Path to copied file
    """
    ensure_directory(destination.parent)
    shutil.copy2(source, destination)
    return destination


def move_file(source: Path, destination: Path) -> Path:
    """
    Move file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        Path to moved file
    """
    ensure_directory(destination.parent)
    shutil.move(str(source), str(destination))
    return destination


def get_video_output_dir(video_path: str) -> Path:
    """
    Get base output directory for a video file and create all subdirectories.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to base output directory (data/{video_name}/)
    """
    video_name = Path(video_path).stem
    base_dir = DATA_DIR / video_name
    
    # Create all subdirectories
    subdirs = [
        base_dir / "metadata",
        base_dir / "embeddings",
        base_dir / "outputs",
        base_dir / "frames",
        base_dir / "audio" / "chunks"
    ]
    for subdir in subdirs:
        ensure_directory(subdir)
    
    return base_dir


def get_video_metadata_dir(video_path: str) -> Path:
    """
    Get metadata directory for a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to metadata directory (data/{video_name}/metadata/)
    """
    base_dir = get_video_output_dir(video_path)
    metadata_dir = base_dir / "metadata"
    ensure_directory(metadata_dir)
    return metadata_dir


def get_video_embeddings_dir(video_path: str) -> Path:
    """
    Get embeddings directory for a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to embeddings directory (data/{video_name}/embeddings/)
    """
    base_dir = get_video_output_dir(video_path)
    embeddings_dir = base_dir / "embeddings"
    ensure_directory(embeddings_dir)
    return embeddings_dir


def get_video_outputs_dir(video_path: str) -> Path:
    """
    Get outputs directory for a video file (frontend outputs).
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to outputs directory (data/{video_name}/outputs/)
    """
    base_dir = get_video_output_dir(video_path)
    outputs_dir = base_dir / "outputs"
    ensure_directory(outputs_dir)
    return outputs_dir


def get_video_frames_dir(video_path: str) -> Path:
    """
    Get frames directory for a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to frames directory (data/{video_name}/frames/)
    """
    base_dir = get_video_output_dir(video_path)
    frames_dir = base_dir / "frames"
    ensure_directory(frames_dir)
    return frames_dir


def get_video_audio_dir(video_path: str) -> Path:
    """
    Get audio directory for a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to audio directory (data/{video_name}/audio/)
    """
    base_dir = get_video_output_dir(video_path)
    audio_dir = base_dir / "audio"
    ensure_directory(audio_dir)
    return audio_dir


def save_json(data: Any, file_path: str, indent: int = 2) -> Path:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (dict, list, etc.)
        file_path: Path to save JSON file
        indent: JSON indentation (default: 2)
        
    Returns:
        Path to saved JSON file
    """
    file_path_obj = Path(file_path)
    ensure_directory(file_path_obj.parent)
    
    # Convert numpy arrays and other non-serializable types
    serializable_data = _make_json_serializable(data)
    
    with open(file_path_obj, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=indent, ensure_ascii=False)
    
    return file_path_obj


def load_json(file_path: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data (dict, list, etc.)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.
    Handles numpy arrays, Path objects, etc.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    import numpy as np
    
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        # Try to convert to string as fallback
        return str(obj)

