"""
Video processing API routes.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from redis import Redis
from rq import Queue
import config as config_module
from fastapi.responses import FileResponse, StreamingResponse
from src.api.models.schemas import (
    VideoUploadResponse, ProcessingRequest, ProcessingStatus, Results, ErrorResponse
)
from src.api.services.processing_service import ProcessingService, process_video_job
from src.api.services.job_manager import job_manager, JobStatus
from src.utils.validation import is_video_file
from src.utils.file_utils import ensure_directory, get_video_metadata_dir, move_file
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)

router = APIRouter(prefix="/video", tags=["video"])

# Initialize processing service
processing_service = ProcessingService()

# Temporary upload directory (for initial upload, then moved to metadata folder)
UPLOAD_DIR = config.TEMP_DIR / "uploads"
ensure_directory(UPLOAD_DIR)


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for processing.
    
    Args:
        file: Video file to upload
        
    Returns:
        Job ID for tracking processing
    """
    try:
        # Validate file
        if not is_video_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(config.SUPPORTED_VIDEO_FORMATS)}"
            )
        
        # Validate file size
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > config.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {config.MAX_UPLOAD_SIZE / (1024*1024):.1f} MB"
            )
        
        # Save uploaded file temporarily
        ensure_directory(UPLOAD_DIR)
        original_filename = Path(file.filename).stem
        original_suffix = Path(file.filename).suffix
        unique_id = os.urandom(4).hex()
        video_filename = f"{original_filename}_{unique_id}{original_suffix}"
        temp_video_path = UPLOAD_DIR / video_filename
        
        with open(temp_video_path, "wb") as f:
            f.write(file_content)
        
        # Move video to metadata folder in organized structure
        # Create directory structure based on final video path
        # The directory name will be the video filename (stem) without extension
        final_video_name = f"{original_filename}_{unique_id}"
        # Create a path reference to determine directory structure
        # This path doesn't need to exist, just used to determine directory name
        reference_path = UPLOAD_DIR / f"{final_video_name}{original_suffix}"
        metadata_dir = get_video_metadata_dir(str(reference_path))
        final_video_path = metadata_dir / video_filename
        
        # Move file to final location
        move_file(temp_video_path, final_video_path)
        video_path = str(final_video_path)
        
        # Create job with final video path
        job_id = job_manager.create_job(video_path)
        
        logger.info(f"Video uploaded: {video_filename} -> Job {job_id}")
        
        return VideoUploadResponse(
            job_id=job_id,
            id=job_id,  # For compatibility
            message=f"Video uploaded successfully. Job ID: {job_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/process", response_model=ProcessingStatus)
async def process_video(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """
    Start processing a video.
    
    Args:
        request: Processing request with job_id
        background_tasks: FastAPI background tasks
        
    Returns:
        Processing status
    """
    try:
        job = job_manager.get_job(request.job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {request.job_id} not found")
        
        if job["status"] == JobStatus.PROCESSING:
            return ProcessingStatus(
                status=job["status"],
                progress=job["progress"],
                stage=job["stage"]
            )
        
        if job["status"] == JobStatus.COMPLETED:
            return ProcessingStatus(
                status=job["status"],
                progress=100.0,
                stage="Completed"
            )
        
        # Enqueue processing job via RQ so a worker picks it up
        video_path = job["video_path"]
        try:
            redis_conn = Redis(
                host=config_module.REDIS_HOST,
                port=config_module.REDIS_PORT,
                socket_connect_timeout=1,
            )
            redis_conn.ping()  # verify Redis is actually reachable
            q = Queue("default", connection=redis_conn)
            # Enqueue by import path so the worker (rq) can resolve the callable
            q.enqueue("src.api.services.processing_service.process_video_job", request.job_id, str(video_path))
            logger.info(f"Enqueued job {request.job_id} to Redis/RQ worker")
        except Exception as e:
            # No Redis available (typical for local/non-Docker runs): run the job
            # in-process via FastAPI BackgroundTasks so the app still works without
            # a Redis server or RQ worker. The Docker path (with Redis) is unchanged.
            logger.warning(
                f"Redis unavailable ({e}); running job {request.job_id} in-process via BackgroundTasks"
            )
            job_manager.update_status(
                request.job_id, JobStatus.PROCESSING, progress=0.0, stage="Starting processing (local mode)"
            )
            background_tasks.add_task(process_video_job, request.job_id, str(video_path))
        
        return ProcessingStatus(
            status="processing",
            progress=0.0,
            stage="Starting processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


@router.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_status(job_id: str):
    """
    Get processing status for a job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Processing status
    """
    status = job_manager.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return ProcessingStatus(**status)


@router.get("/results/{job_id}", response_model=Results)
async def get_results(job_id: str):
    """
    Get processing results for a completed job.
    Can load from in-memory job manager or from disk for previously processed videos.
    
    Args:
        job_id: Job identifier (or video directory name for historical videos)
        
    Returns:
        Processing results
    """
    # First, try to get from in-memory job manager
    job = job_manager.get_job(job_id)
    if job:
        if job["status"] != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not completed. Status: {job['status']}"
            )
        
        results = job_manager.get_results(job_id)
        if results:
            return Results(**results)
    
    # If not in memory, try to load from disk (for previously processed videos)
    try:
        results = _load_results_from_disk(job_id)
        if results:
            return Results(**results)
    except Exception as e:
        logger.debug(f"Could not load results from disk for {job_id}: {e}")
    
    raise HTTPException(status_code=404, detail=f"Results not found for job {job_id}")


def _load_results_from_disk(job_id: str) -> Optional[Dict]:
    """
    Load processing results from disk for a previously processed video.
    
    Args:
        job_id: Video directory name (used as identifier)
        
    Returns:
        Results dictionary or None if not found
    """
    try:
        from src.utils.file_utils import get_video_outputs_dir, get_video_metadata_dir, DATA_DIR
        import json
        from pathlib import Path
        
        # Try to find the video directory
        video_dir = DATA_DIR / job_id
        if not video_dir.exists() or not video_dir.is_dir():
            return None
        
        outputs_dir = video_dir / "outputs"
        metadata_dir = video_dir / "metadata"
        
        if not outputs_dir.exists():
            return None
        
        # Load captions
        captions = []
        captions_srt = ""
        if (outputs_dir / "captions.srt").exists():
            captions_srt = (outputs_dir / "captions.srt").read_text(encoding='utf-8')
            # Parse SRT to get captions list
            captions = _parse_srt_to_captions(captions_srt)
        
        # Load transcript
        transcript_text = ""
        if (outputs_dir / "transcript.txt").exists():
            transcript_text = (outputs_dir / "transcript.txt").read_text(encoding='utf-8')
        
        # Load summary
        summary_text = ""
        key_points = []
        if (outputs_dir / "summary.txt").exists():
            summary_text = (outputs_dir / "summary.txt").read_text(encoding='utf-8')
        
        # Try to load key points from summary JSON if available
        summary_json_path = metadata_dir / "summary.json"
        if summary_json_path.exists():
            try:
                with open(summary_json_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    key_points = summary_data.get("key_points", [])
            except Exception:
                pass
        
        # Load enhanced segments for metadata
        enhanced_segments = []
        enhanced_transcript_path = metadata_dir / "enhanced_transcript.json"
        if enhanced_transcript_path.exists():
            try:
                with open(enhanced_transcript_path, 'r', encoding='utf-8') as f:
                    enhanced_data = json.load(f)
                    enhanced_segments = enhanced_data.get("segments", [])
            except Exception:
                pass
        
        # Load video metadata
        video_metadata = {}
        video_metadata_path = metadata_dir / "video_metadata.json"
        if video_metadata_path.exists():
            try:
                with open(video_metadata_path, 'r', encoding='utf-8') as f:
                    video_metadata = json.load(f)
            except Exception:
                pass
        
        # Find video file
        video_file = None
        for ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm']:
            video_files = list(metadata_dir.glob(f'*{ext}'))
            if video_files:
                video_file = video_files[0]
                break
        
        results = {
            "job_id": job_id,
            "captions": captions,
            "captions_text": captions_srt,
            "transcript_text": transcript_text,
            "summary_text": summary_text,
            "key_points": key_points,
            "video_url": f"/api/video/stream/{job_id}" if video_file else None,
            "metadata": {
                "video": video_metadata,
                "enhanced_segments": enhanced_segments
            }
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading results from disk for {job_id}: {e}", exc_info=True)
        return None


def _parse_srt_to_captions(srt_content: str) -> List[Dict]:
    """
    Parse SRT content to list of caption dictionaries.
    
    Args:
        srt_content: SRT file content
        
    Returns:
        List of caption dicts with text, start_time, end_time
    """
    import re
    
    captions = []
    # SRT format: number, timestamp, text, blank line
    pattern = r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})\s+(.+?)(?=\n\d+\s+\d{2}:\d{2}:\d{2}|\Z)'
    
    matches = re.finditer(pattern, srt_content, re.DOTALL)
    
    for match in matches:
        start_str = match.group(2).replace(',', '.')
        end_str = match.group(3).replace(',', '.')
        text = match.group(4).strip().replace('\n', ' ')
        
        # Convert timestamp to seconds
        def timestamp_to_seconds(ts: str) -> float:
            parts = ts.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            secs = float(parts[2])
            return hours * 3600 + minutes * 60 + secs
        
        captions.append({
            "text": text,
            "start_time": timestamp_to_seconds(start_str),
            "end_time": timestamp_to_seconds(end_str)
        })
    
    return captions


@router.get("/download/{job_id}/{format}")
async def download_file(
    job_id: str,
    format: str,
    type: str = Query("captions", description="File type: captions, transcript, or summary")
):
    """
    Download generated file in specified format.
    
    Args:
        job_id: Job identifier
        format: File format (srt, vtt, txt, docx, pdf)
        type: File type (captions, transcript, summary)
        
    Returns:
        File download
    """
    try:
        job = job_manager.get_job(job_id)
        if not job or job["status"] != JobStatus.COMPLETED:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
        
        results = job_manager.get_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail=f"Results not found for job {job_id}")
        
        # Get outputs directory (organized structure)
        video_path = Path(job["video_path"])
        from src.utils.file_utils import get_video_outputs_dir
        
        outputs_dir = get_video_outputs_dir(str(video_path))
        
        # Determine file path based on type and format
        if type == "captions":
            if format == "srt":
                filename = "captions.srt"
                file_path = outputs_dir / filename
            elif format == "vtt":
                filename = "captions.vtt"
                file_path = outputs_dir / filename
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format for captions: {format}")
        
        elif type == "transcript":
            if format == "txt":
                filename = "transcript.txt"
                file_path = outputs_dir / filename
            elif format == "docx":
                filename = "transcript.docx"
                file_path = outputs_dir / filename
                # Generate DOCX file if it doesn't exist
                if not file_path.exists():
                    from src.output.transcript_formatter import TranscriptFormatter
                    formatter = TranscriptFormatter()
                    # Get enhanced segments from results metadata
                    enhanced_segments = results.get("metadata", {}).get("enhanced_segments")
                    
                    if not enhanced_segments:
                        # Fallback: try to get from captions (which have timestamps)
                        captions = results.get("captions", [])
                        if captions:
                            enhanced_segments = [
                                {
                                    "text": cap.get("text", ""),
                                    "start_time": cap.get("start_time", 0.0),
                                    "end_time": cap.get("end_time", 0.0)
                                }
                                for cap in captions
                            ]
                    
                    if enhanced_segments:
                        try:
                            formatter.format_to_docx(enhanced_segments, str(file_path))
                            logger.info(f"Generated DOCX file: {file_path}")
                        except Exception as e:
                            logger.error(f"Error generating DOCX: {e}", exc_info=True)
                            raise HTTPException(status_code=500, detail=f"Failed to generate DOCX: {str(e)}")
                    else:
                        raise HTTPException(status_code=500, detail="Could not generate DOCX: transcript segments not available")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format for transcript: {format}")
        
        elif type == "summary":
            if format == "txt":
                filename = "summary.txt"
                file_path = outputs_dir / filename
            elif format == "pdf":
                filename = "summary.pdf"
                file_path = outputs_dir / filename
                # Generate PDF file if it doesn't exist
                if not file_path.exists():
                    from src.output.transcript_formatter import TranscriptFormatter
                    formatter = TranscriptFormatter()
                    summary_text = results.get("summary_text", "")
                    key_points = results.get("key_points", [])
                    if summary_text:
                        formatter.format_summary_to_pdf(summary_text, str(file_path), key_points)
                    else:
                        raise HTTPException(status_code=500, detail="Could not generate PDF: summary not available")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format for summary: {format}")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {type}")
        
        # Check if file exists in outputs folder, otherwise generate on-the-fly
        if file_path.exists():
            # Determine media type based on format
            media_types = {
                "txt": "text/plain",
                "srt": "text/plain",
                "vtt": "text/vtt",
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "pdf": "application/pdf"
            }
            media_type = media_types.get(format, "application/octet-stream")
            
            # Return file from disk
            return FileResponse(
                path=str(file_path),
                filename=filename,
                media_type=media_type
            )
        else:
            # Fallback: generate on-the-fly from results
            logger.warning(f"Output file not found at {file_path}, generating on-the-fly")
            if type == "captions":
                if format == "srt":
                    content = results.get("captions_text", "")
                elif format == "vtt":
                    captions = results.get("captions", [])
                    if captions:
                        content = processing_service.caption_formatter.format_vtt(captions)
                    else:
                        content = results.get("captions_text", "").replace(",", ".")
            elif type == "transcript":
                content = results.get("transcript_text", "")
            elif type == "summary":
                content = results.get("summary_text", "")
            else:
                content = ""
            
            return StreamingResponse(
                iter([content.encode('utf-8')]),
                media_type="text/plain",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/stream/{job_id}")
async def stream_video(job_id: str):
    """
    Stream video file for playback in frontend.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Video file stream
    """
    try:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        video_path = Path(job["video_path"])
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found for job {job_id}")
        
        # Return video file with appropriate media type
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Type": "video/mp4"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video streaming failed: {str(e)}")


@router.get("/history")
async def get_processed_videos():
    """
    Get list of all processed videos from the data folder.
    
    Returns:
        List of processed video information
    """
    try:
        from src.utils.file_utils import get_video_outputs_dir, get_video_metadata_dir
        from src.utils.file_utils import DATA_DIR
        import json
        from datetime import datetime
        
        processed_videos = []
        
        # Scan data directory for processed videos
        if not DATA_DIR.exists():
            return {"videos": []}
        
        for video_dir in DATA_DIR.iterdir():
            if not video_dir.is_dir():
                continue
            
            # Check if this directory has outputs (indicating processing is complete)
            outputs_dir = video_dir / "outputs"
            metadata_dir = video_dir / "metadata"
            
            if not outputs_dir.exists() or not metadata_dir.exists():
                continue
            
            # Try to find video file in metadata directory
            video_file = None
            for ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm']:
                video_files = list(metadata_dir.glob(f'*{ext}'))
                if video_files:
                    video_file = video_files[0]
                    break
            
            if not video_file:
                continue
            
            # Get video metadata if available
            video_metadata_path = metadata_dir / "video_metadata.json"
            video_name = video_dir.name
            processed_date = None
            duration = None
            
            if video_metadata_path.exists():
                try:
                    with open(video_metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        duration = metadata.get('duration', 0)
                        # Get file modification time as processed date
                        processed_date = datetime.fromtimestamp(
                            video_metadata_path.stat().st_mtime
                        ).isoformat()
                except Exception as e:
                    logger.warning(f"Error reading metadata for {video_name}: {e}")
            
            # Check if results files exist
            has_captions = (outputs_dir / "captions.srt").exists()
            has_transcript = (outputs_dir / "transcript.txt").exists()
            has_summary = (outputs_dir / "summary.txt").exists()
            
            # Try to get job_id from the video name or create a reference
            # The video name format is typically: {original_name}_{unique_id}
            job_id = video_name  # Use directory name as job identifier
            
            processed_videos.append({
                "job_id": job_id,
                "video_name": video_name,
                "video_path": str(video_file),
                "processed_date": processed_date,
                "duration": duration,
                "has_captions": has_captions,
                "has_transcript": has_transcript,
                "has_summary": has_summary,
                "outputs_dir": str(outputs_dir)
            })
        
        # Sort by processed date (most recent first)
        processed_videos.sort(
            key=lambda x: x.get("processed_date", ""), 
            reverse=True
        )
        
        return {"videos": processed_videos}
        
    except Exception as e:
        logger.error(f"Error getting processed videos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get processed videos: {str(e)}")

