"""
Configuration file for Multimodal Video Captioning & Summarization System.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Helper function to get env var with proper default handling
def get_env(key: str, default: str) -> str:
    """Get environment variable, using default if not set or empty."""
    value = os.getenv(key, default)
    return default if value == "" else value

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEST_DATA_DIR = BASE_DIR / "tests" / "test_set"  # Test dataset location
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = DATA_DIR / "output"
TEMP_DIR = DATA_DIR / "temp"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Model Configuration
# ============================================================================

# Whisper Model Configuration
WHISPER_MODEL = get_env("WHISPER_MODEL", "base")  # tiny, base, small, medium, large-v2, large-v3
WHISPER_DEVICE = get_env("WHISPER_DEVICE", "auto")  # auto, cpu, cuda

# CLIP Model Configuration
CLIP_MODEL = get_env("CLIP_MODEL", "ViT-B/32")  # ViT-B/32, ViT-L/14, etc.
CLIP_DEVICE = get_env("CLIP_DEVICE", "auto")  # auto, cpu, cuda

# T5 Model Configuration
T5_MODEL = get_env("T5_MODEL", "t5-base")  # t5-small, t5-base, t5-large
T5_DEVICE = get_env("T5_DEVICE", "auto")  # auto, cpu, cuda

# Speaker Diarization Model
DIARIZATION_MODEL = get_env("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

# ============================================================================
# Processing Parameters
# ============================================================================

# Video Processing
VIDEO_FPS = float(get_env("VIDEO_FPS", "1.0"))  # Frames per second to extract
VIDEO_TARGET_SIZE = tuple(map(int, get_env("VIDEO_TARGET_SIZE", "224,224").split(",")))  # (width, height)
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

# Audio Processing
AUDIO_SAMPLE_RATE = int(get_env("AUDIO_SAMPLE_RATE", "16000"))  # 16kHz recommended for Whisper
AUDIO_CHUNK_DURATION = float(get_env("AUDIO_CHUNK_DURATION", "30.0"))  # seconds
AUDIO_FORMAT = get_env("AUDIO_FORMAT", "wav")

# Frame Processing
FRAME_PREPROCESS_SIZE = tuple(map(int, get_env("FRAME_PREPROCESS_SIZE", "224,224").split(",")))
FRAME_TEMPORAL_POOLING = get_env("FRAME_TEMPORAL_POOLING", "mean")  # mean, max, attention

# ============================================================================
# Multimodal Fusion Configuration
# ============================================================================

# Fusion Model Configuration
FUSION_ENABLED = get_env("FUSION_ENABLED", "True").lower() == "true"
FUSION_MODEL_PATH = get_env("FUSION_MODEL_PATH", "")  # Optional: path to pre-trained model
FUSION_DEVICE = get_env("FUSION_DEVICE", "auto")  # auto, cpu, cuda

# Transformer Architecture
FUSION_HIDDEN_DIM = int(get_env("FUSION_HIDDEN_DIM", "768"))
FUSION_NUM_LAYERS = int(get_env("FUSION_NUM_LAYERS", "6"))
FUSION_NUM_HEADS = int(get_env("FUSION_NUM_HEADS", "12"))
FUSION_DROPOUT = float(get_env("FUSION_DROPOUT", "0.1"))

# Alignment Configuration
FUSION_ALIGNMENT_METHOD = get_env("FUSION_ALIGNMENT_METHOD", "interpolate")  # interpolate, pooling, nearest
FUSION_BATCH_SIZE = int(get_env("FUSION_BATCH_SIZE", "32"))
FUSION_MAX_SEQUENCE_LENGTH = int(get_env("FUSION_MAX_SEQUENCE_LENGTH", "512"))
FUSION_COMBINATION_METHOD = get_env("FUSION_COMBINATION_METHOD", "concat")  # concat, add, gated

# Embedding Dimensions
AUDIO_EMBEDDING_DIM = int(get_env("AUDIO_EMBEDDING_DIM", "512"))  # Whisper base: 512, large: 1280
VISUAL_EMBEDDING_DIM = int(get_env("VISUAL_EMBEDDING_DIM", "512"))  # CLIP ViT-B/32: 512

# Audio Embeddings Configuration
AUDIO_EMBEDDING_EXTRACT = get_env("AUDIO_EMBEDDING_EXTRACT", "True").lower() == "true"

# ============================================================================
# Generation Configuration
# ============================================================================

# Caption Generation
CAPTION_MAX_LENGTH = int(get_env("CAPTION_MAX_LENGTH", "128"))
CAPTION_MIN_LENGTH = int(get_env("CAPTION_MIN_LENGTH", "10"))
CAPTION_MAX_DURATION = float(get_env("CAPTION_MAX_DURATION", "5.0"))  # seconds per caption

# Summarization
SUMMARY_MAX_LENGTH = int(get_env("SUMMARY_MAX_LENGTH", "512"))
SUMMARY_MIN_LENGTH = int(get_env("SUMMARY_MIN_LENGTH", "50"))
SUMMARY_NUM_KEY_POINTS = int(get_env("SUMMARY_NUM_KEY_POINTS", "5"))

# ============================================================================
# API Configuration
# ============================================================================

API_HOST = get_env("API_HOST", "0.0.0.0")
API_PORT = int(get_env("API_PORT", "8000"))
API_DEBUG = get_env("API_DEBUG", "False").lower() == "true"
API_TITLE = get_env("API_TITLE", "Multimodal Video Captioning API")
API_VERSION = get_env("API_VERSION", "1.0.0")

# CORS Configuration
CORS_ORIGINS = get_env("CORS_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = get_env("CORS_ALLOW_CREDENTIALS", "True").lower() == "true"

# File Upload Configuration
MAX_UPLOAD_SIZE = int(get_env("MAX_UPLOAD_SIZE", "1073741824"))  # 1GB in bytes
ALLOWED_EXTENSIONS = SUPPORTED_VIDEO_FORMATS

# ============================================================================
# Device Configuration
# ============================================================================

# Auto-detect device if set to "auto"
def get_device(device_setting: str) -> str:
    """Get device string (cpu or cuda) based on setting and availability."""
    if device_setting == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting

# Set actual devices
WHISPER_DEVICE = get_device(WHISPER_DEVICE)
CLIP_DEVICE = get_device(CLIP_DEVICE)
T5_DEVICE = get_device(T5_DEVICE)
FUSION_DEVICE = get_device(FUSION_DEVICE)
FUSION_DEVICE = get_device(FUSION_DEVICE)

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = get_env("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = get_env("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = get_env("LOG_FILE", str(BASE_DIR / "logs" / "app.log"))
LOG_DIR = Path(LOG_FILE).parent
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# External Service Configuration
# ============================================================================

# Hugging Face
HUGGING_FACE_TOKEN = get_env("HUGGING_FACE_TOKEN", "")

# Database (if using)
DATABASE_URL = get_env("DATABASE_URL", "")
DATABASE_NAME = get_env("DATABASE_NAME", "video_processing")

# Redis (for task queue, if using)
REDIS_HOST = get_env("REDIS_HOST", "localhost")
REDIS_PORT = int(get_env("REDIS_PORT", "6379"))
REDIS_DB = int(get_env("REDIS_DB", "0"))

# ============================================================================
# Security Configuration
# ============================================================================

SECRET_KEY = get_env("SECRET_KEY", "")
API_KEY = get_env("API_KEY", "")

# ============================================================================
# Performance Configuration
# ============================================================================

# Batch Processing
BATCH_SIZE = int(get_env("BATCH_SIZE", "8"))
NUM_WORKERS = int(get_env("NUM_WORKERS", "4"))

# Caching
ENABLE_CACHE = get_env("ENABLE_CACHE", "True").lower() == "true"
CACHE_TTL = int(get_env("CACHE_TTL", "3600"))  # seconds

# ============================================================================
# Output Format Configuration
# ============================================================================

# Supported output formats
OUTPUT_FORMATS = {
    "captions": [".srt", ".vtt"],
    "transcript": [".txt", ".docx"],
    "summary": [".txt", ".pdf"]
}

# Default output formats
DEFAULT_CAPTION_FORMAT = get_env("DEFAULT_CAPTION_FORMAT", "srt")
DEFAULT_TRANSCRIPT_FORMAT = get_env("DEFAULT_TRANSCRIPT_FORMAT", "txt")
DEFAULT_SUMMARY_FORMAT = get_env("DEFAULT_SUMMARY_FORMAT", "txt")

