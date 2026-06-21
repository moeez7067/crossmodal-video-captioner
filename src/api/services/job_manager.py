"""
Job status manager for tracking video processing jobs.
"""

from typing import Dict, Optional
from datetime import datetime
from enum import Enum
import uuid
from src.utils.logger import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    """Manages job status and results."""
    
    def __init__(self):
        """Initialize job manager."""
        self._jobs: Dict[str, Dict] = {}
    
    def create_job(self, video_path: str) -> str:
        """
        Create a new job.
        
        Args:
            video_path: Path to uploaded video file
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "video_path": video_path,
            "status": JobStatus.PENDING,
            "progress": 0.0,
            "stage": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "results": None,
            "error": None
        }
        logger.info(f"Created job {job_id} for video {video_path}")
        return job_id
    
    def update_status(self, job_id: str, status: JobStatus, 
                     progress: Optional[float] = None,
                     stage: Optional[str] = None,
                     error: Optional[str] = None):
        """
        Update job status.
        
        Args:
            job_id: Job identifier
            status: New status
            progress: Progress percentage (0-100)
            stage: Current processing stage
            error: Error message if failed
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        
        self._jobs[job_id]["status"] = status
        self._jobs[job_id]["updated_at"] = datetime.now()
        
        if progress is not None:
            self._jobs[job_id]["progress"] = progress
        if stage is not None:
            self._jobs[job_id]["stage"] = stage
        if error is not None:
            self._jobs[job_id]["error"] = error
        
        logger.debug(f"Updated job {job_id}: {status} ({progress}%) - {stage}")
    
    def set_results(self, job_id: str, results: Dict):
        """
        Set job results.
        
        Args:
            job_id: Job identifier
            results: Processing results dictionary
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        
        self._jobs[job_id]["results"] = results
        self._jobs[job_id]["status"] = JobStatus.COMPLETED
        self._jobs[job_id]["progress"] = 100.0
        self._jobs[job_id]["updated_at"] = datetime.now()
        logger.info(f"Set results for job {job_id}")
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """
        Get job information.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job dictionary or None if not found
        """
        return self._jobs.get(job_id)
    
    def get_status(self, job_id: str) -> Optional[Dict]:
        """
        Get job status.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Status dictionary or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None
        
        return {
            "status": job["status"],
            "progress": job["progress"],
            "stage": job["stage"],
            "error": job.get("error"),
            "estimated_time_remaining": None  # Can be calculated if needed
        }
    
    def get_results(self, job_id: str) -> Optional[Dict]:
        """
        Get job results.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Results dictionary or None if not found
        """
        job = self.get_job(job_id)
        if not job or job["status"] != JobStatus.COMPLETED:
            return None
        
        return job.get("results")
    
    def delete_job(self, job_id: str):
        """
        Delete a job (cleanup).
        
        Args:
            job_id: Job identifier
        """
        if job_id in self._jobs:
            del self._jobs[job_id]
            logger.info(f"Deleted job {job_id}")


# Global job manager instance
job_manager = JobManager()

