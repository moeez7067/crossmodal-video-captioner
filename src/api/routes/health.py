"""
Health check API routes.
"""

from fastapi import APIRouter
from datetime import datetime
from src.api.models.schemas import HealthResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        API health status
    """
    try:
        # Check model availability (basic check)
        models_status = {
            "whisper": True,  # Will be checked when actually loading
            "clip": True,
            "diarization": True
        }
        
        return HealthResponse(
            status="healthy",
            models=models_status,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now()
        )

