"""
Pydantic models for API request/response schemas.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class VideoUploadResponse(BaseModel):
    """Response model for video upload."""
    job_id: str = Field(..., description="Unique job identifier")
    id: Optional[str] = Field(None, description="Alternative ID field (for compatibility)")
    message: Optional[str] = Field(None, description="Upload success message")


class ProcessingRequest(BaseModel):
    """Request model for starting video processing."""
    job_id: str = Field(..., description="Job ID from upload")


class ProcessingStatus(BaseModel):
    """Response model for processing status."""
    status: str = Field(..., description="Status: pending, processing, completed, failed")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage (0-100)")
    stage: Optional[str] = Field(None, description="Current processing stage")
    estimated_time_remaining: Optional[float] = Field(None, description="Estimated seconds remaining")
    error: Optional[str] = Field(None, description="Error message if failed")


class Caption(BaseModel):
    """Caption model with timestamps."""
    text: str = Field(..., description="Caption text")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")


class Results(BaseModel):
    """Response model for processing results."""
    job_id: str = Field(..., description="Job identifier")
    captions: Optional[List[Caption]] = Field(None, description="Time-synced captions")
    captions_text: Optional[str] = Field(None, description="Captions as text (SRT format)")
    transcript_text: Optional[str] = Field(None, description="Full transcript with speaker attribution")
    summary_text: Optional[str] = Field(None, description="Abstractive summary")
    key_points: Optional[List[str]] = Field(None, description="Key discussion points from summary")
    video_url: Optional[str] = Field(None, description="URL to processed video (if available)")


class ErrorResponse(BaseModel):
    """Error response model."""
    message: str = Field(..., description="Error message")
    status: int = Field(..., description="HTTP status code")
    detail: Optional[str] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="API status")
    models: Optional[Dict[str, bool]] = Field(None, description="Model availability status")
    timestamp: Optional[datetime] = Field(None, description="Response timestamp")

