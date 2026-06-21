# Phase 5: Backend API - Implementation Guide

## Overview

Phase 5 implements the FastAPI backend that connects the frontend to the processing pipeline. The API orchestrates all Phase 2 modules to process videos and return results.

## Architecture

```
Frontend (React) → FastAPI Backend → Processing Service → Phase 2 Modules
     ↓                    ↓                  ↓                    ↓
  Port 3000          Port 8000        Job Manager        Video/Audio/Visual
```

## API Endpoints

### 1. Upload Video
**POST** `/api/video/upload`

Upload a video file for processing.

**Request:**
- `file`: Video file (multipart/form-data)

**Response:**
```json
{
  "job_id": "uuid-string",
  "id": "uuid-string",
  "message": "Video uploaded successfully. Job ID: uuid-string"
}
```

### 2. Start Processing
**POST** `/api/video/process`

Start processing an uploaded video.

**Request:**
```json
{
  "job_id": "uuid-string"
}
```

**Response:**
```json
{
  "status": "processing",
  "progress": 0.0,
  "stage": "Starting processing"
}
```

### 3. Get Status
**GET** `/api/video/status/{job_id}`

Get processing status for a job.

**Response:**
```json
{
  "status": "processing",
  "progress": 45.0,
  "stage": "Transcribing audio",
  "estimated_time_remaining": null,
  "error": null
}
```

**Status values:**
- `pending`: Job created but not started
- `processing`: Currently processing
- `completed`: Processing finished successfully
- `failed`: Processing failed

### 4. Get Results
**GET** `/api/video/results/{job_id}`

Get processing results for a completed job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "captions": [
    {
      "text": "Hello, welcome to this presentation.",
      "start_time": 0.0,
      "end_time": 3.5
    }
  ],
  "captions_text": "1\n00:00:00,000 --> 00:00:03,500\nHello, welcome...",
  "transcript_text": "[SPEAKER_00] [00:00:00] Hello, welcome...",
  "summary_text": "This presentation covers...",
  "video_url": null
}
```

### 5. Download File
**GET** `/api/video/download/{job_id}/{format}?type={type}`

Download generated files in various formats.

**Parameters:**
- `job_id`: Job identifier
- `format`: File format (`srt`, `vtt`, `txt`, `docx`, `pdf`)
- `type`: File type (`captions`, `transcript`, `summary`)

**Examples:**
- `/api/video/download/{job_id}/srt?type=captions` - Download captions as SRT
- `/api/video/download/{job_id}/txt?type=transcript` - Download transcript as TXT
- `/api/video/download/{job_id}/txt?type=summary` - Download summary as TXT

### 6. Health Check
**GET** `/api/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "whisper": true,
    "clip": true,
    "diarization": true
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## Running the API

### 1. Install Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install FastAPI dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Start the Server

**Option 1: Using run.py**
```bash
python run.py
```

**Option 2: Using uvicorn directly**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will start on `http://localhost:8000`

### 3. Access API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Processing Pipeline

When a video is processed, the following steps occur:

1. **Video Preprocessing** (10% progress)
   - Extract frames
   - Extract audio
   - Get metadata

2. **Audio Extraction** (20% progress)
   - Convert to WAV format
   - Chunk audio for processing

3. **Speech-to-Text** (40% progress)
   - Transcribe audio with Whisper
   - Get segments with timestamps

4. **Speaker Diarization** (55% progress)
   - Identify speakers
   - Assign speakers to segments

5. **Visual Processing** (75% progress)
   - Extract visual embeddings
   - Detect slide transitions

6. **Format Outputs** (90% progress)
   - Format captions (SRT/VTT)
   - Format transcript with speakers
   - Generate summary

7. **Complete** (100% progress)
   - Save all results
   - Make available for download

## Job Management

Jobs are tracked in-memory using the `JobManager` class. Each job has:

- `job_id`: Unique identifier
- `video_path`: Path to uploaded video
- `status`: Current status (pending, processing, completed, failed)
- `progress`: Progress percentage (0-100)
- `stage`: Current processing stage
- `results`: Processing results (when completed)
- `error`: Error message (if failed)

## File Structure

```
src/api/
├── __init__.py
├── main.py                 # FastAPI application
├── routes/
│   ├── __init__.py
│   ├── video.py           # Video processing routes
│   └── health.py          # Health check route
├── models/
│   ├── __init__.py
│   └── schemas.py         # Pydantic models
└── services/
    ├── __init__.py
    ├── job_manager.py     # Job status tracking
    └── processing_service.py  # Processing orchestration
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input (file format, size, etc.)
- **404 Not Found**: Job not found
- **422 Validation Error**: Request validation failed
- **500 Internal Server Error**: Processing errors

All errors return JSON responses:
```json
{
  "message": "Error message",
  "status": 400,
  "detail": "Additional details"
}
```

## CORS Configuration

CORS is configured to allow requests from the frontend. Default settings:

- **Origins**: `*` (all origins) - Change in `.env` for production
- **Methods**: All methods
- **Headers**: All headers

To configure CORS, set `CORS_ORIGINS` in `.env`:
```env
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

## Testing the API

### Using curl

```bash
# Upload video
curl -X POST "http://localhost:8000/api/video/upload" \
  -F "file=@video.mp4"

# Start processing
curl -X POST "http://localhost:8000/api/video/process" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "your-job-id"}'

# Get status
curl "http://localhost:8000/api/video/status/your-job-id"

# Get results
curl "http://localhost:8000/api/video/results/your-job-id"
```

### Using Python

```python
import requests

# Upload video
with open("video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/video/upload",
        files={"file": f}
    )
    job_id = response.json()["job_id"]

# Start processing
requests.post(
    "http://localhost:8000/api/video/process",
    json={"job_id": job_id}
)

# Poll status
while True:
    status = requests.get(
        f"http://localhost:8000/api/video/status/{job_id}"
    ).json()
    print(f"Status: {status['status']}, Progress: {status['progress']}%")
    if status["status"] in ["completed", "failed"]:
        break

# Get results
results = requests.get(
    f"http://localhost:8000/api/video/results/{job_id}"
).json()
print(results)
```

## Integration with Frontend

The frontend is already configured to connect to this API. The API base URL is set in `frontend/src/services/api.ts`:

```typescript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
```

To run both frontend and backend:

**Terminal 1 - Backend:**
```bash
python run.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

Then access the frontend at `http://localhost:3000` and it will connect to the backend API.

## Output Files

All processing outputs are saved to JSON files in:
```
data/{video_name}/
├── video_metadata.json
├── frames_info.json
├── audio_metadata.json
├── audio_chunks_info.json
├── transcription.json
├── speaker_diarization.json
├── enhanced_transcript.json
├── speaker_statistics.json
├── slide_detection.json
└── visual_embeddings_info.json
```

## Next Steps

1. **Test the complete pipeline**: Upload a video through the frontend
2. **Monitor processing**: Check status endpoint during processing
3. **Download results**: Test download endpoints for different formats
4. **Error handling**: Test with invalid files, large files, etc.

## Troubleshooting

### API won't start
- Check if port 8000 is available
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check logs for errors

### Processing fails
- Check if FFmpeg is installed and in PATH
- Verify video file is valid
- Check logs for detailed error messages
- Ensure sufficient disk space for outputs

### Frontend can't connect
- Verify API is running on port 8000
- Check CORS configuration
- Check browser console for errors
- Verify `REACT_APP_API_URL` in frontend `.env` (if set)

## Production Considerations

For production deployment:

1. **Use a proper job queue** (Celery, RQ) instead of in-memory job manager
2. **Add authentication** (JWT tokens, API keys)
3. **Use a database** for job persistence
4. **Add rate limiting** to prevent abuse
5. **Configure proper CORS** origins
6. **Add monitoring** and logging
7. **Use a reverse proxy** (Nginx) for production
8. **Add file cleanup** for old jobs

