# Multimodal Video Captioning & Summarization

*Video captioning & summarization via bidirectional cross-attention fusion of audio (Whisper) and visual (CLIP) embeddings.*

A transformer-based AI system that generates time-synchronized captions and abstractive summaries from recorded videos by integrating audio and visual modalities.

## Project Overview

This project implements a **multimodal transformer-based video captioning system** that generates time-synchronized captions and abstractive summaries by integrating audio and visual modalities. The system uses advanced multimodal learning techniques including cross-attention mechanisms and feature fusion to provide richer interpretations than audio-only models.

### Core Capabilities

- **Multimodal Video Captioning**: Generates time-synced captions (.srt/.vtt) using both audio and visual information
- **Speaker-Attributed Transcripts**: Produces full transcripts with speaker identification (.txt/.docx)
- **Abstractive Summarization**: Creates concise, human-like summaries with key points (.txt/.pdf)
- **Multimodal Fusion**: Combines audio and visual embeddings using cross-attention and transformer architectures

### Technical Approach

The system leverages:
- **Multimodal Embeddings**: CLIP for visual embeddings and Whisper encoder for audio embeddings
- **Cross-Modal Contrastive Learning**: CLIP-based visual understanding and alignment
- **Cross-Attention Mechanisms**: Bidirectional attention between audio and visual features
- **Fusion Strategies**: Multiple fusion methods (concatenation, addition, gated fusion)
- **Transformer-Based Generation**: T5 models for caption and summary generation
- **Temporal Alignment**: Synchronizes audio and visual features across time

The pipeline processes both audio (speech) and visual (frames) content simultaneously, enabling the system to understand context that would be missed by audio-only or visual-only approaches.

## Current Status

**System is Fully Functional with Multimodal Fusion & AI Generation!**

- ✅ Phase 1: Backend Foundation
- ✅ Phase 2: Core Processing Modules
- ✅ Phase 3: Multimodal Fusion (Cross-attention & Transformer)
- ✅ Phase 4: Generation Pipeline (T5-based)
- ✅ Phase 5: Backend API Development
- ✅ Phase 6: Frontend Development

The complete pipeline works end-to-end with multimodal fusion and AI-generated captions and summaries! You can upload videos through the web interface and get high-quality captions, transcripts, summaries, and key points generated using fused audio-visual embeddings.

## Quick Start Guide

### Prerequisites

**Choose one setup method:**

**Option A: Local Setup**
- Python 3.8+ with virtual environment
- Node.js 16+ and npm
- FFmpeg installed and in PATH
- (Optional) GPU support for faster processing

**Option B: Docker Setup** (Recommended for easier deployment)
- Docker Desktop
- 15+ GB free disk space
- Hugging Face account with token

### Step 1: Install Dependencies

**Backend:**
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### Step 2: Configure Environment

Create a `.env` file in the project root (or copy from `.env.example`):

```env
# Hugging Face Token (for pyannote.audio)
HUGGING_FACE_TOKEN=your_token_here

# Model Settings (optional, defaults in config.py)
WHISPER_MODEL=base
CLIP_MODEL=ViT-B/32

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

### Step 3: Start Backend API

**Terminal 1:**
```bash
# Make sure you're in the project root
python run.py
```

The API will start on `http://localhost:8000`

You can verify it's running by visiting:
- **API Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/health`

### Step 4: Start Frontend

**Terminal 2:**
```bash
cd frontend
npm start
```

The frontend will start on `http://localhost:3000`

### Step 5: Process a Video!

1. Open `http://localhost:3000` in your browser
2. Upload a video file (drag & drop or click to select)
3. Click "Start Processing"
4. Watch the progress bar as the video is processed
5. View results: captions, transcript, and summary
6. Download files in various formats (SRT, VTT, TXT)

## Docker Deployment (Alternative Setup)

### Prerequisites for Docker

- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop))
- Hugging Face account with access token ([Get token here](https://huggingface.co/settings/tokens))

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd MultimodalVideoIntelligence
```

### Step 2: Create Environment File
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Hugging Face token
# HUGGING_FACE_TOKEN=hf_your_token_here
```

**Important**: Accept the required model licenses:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Step 3: Configure Docker Resources

Open Docker Desktop → Settings → Resources:
- **Memory**: 6-8 GB minimum
- **CPUs**: 4 cores recommended
- **Disk**: Ensure 15+ GB free space

### Step 4: Build Docker Images
```bash
docker-compose build
```

⏱️ **First build takes 15-30 minutes** (downloads Python packages, PyTorch, Whisper, etc.)

### Step 5: Start Containers
```bash
# Start in background
docker-compose up -d

# Or start with logs visible
docker-compose up
```

### Step 6: Access Application

- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

### Docker Management Commands
```bash
# View running containers
docker ps

# View logs
docker-compose logs -f

# Stop containers
docker-compose down

# Restart containers
docker-compose restart

# Rebuild after code changes
docker-compose build --no-cache
docker-compose up -d
```

### Troubleshooting Docker

**Port already in use:**
```bash
docker-compose down
# Change ports in docker-compose.yml if needed
docker-compose up -d
```

**Out of disk space:**
```bash
docker system prune -a
```

**Containers won't start:**
```bash
docker-compose logs backend
docker-compose logs frontend
```

---

## What Happens During Processing

1. **Video Upload** → Video saved to `data/temp/uploads/`
2. **Frame Extraction** → Frames saved to `data/temp/frames/`
3. **Audio Extraction** → Audio saved to `data/temp/audio/`
4. **Transcription** → Whisper transcribes the audio
5. **Speaker Diarization** → Speakers identified
6. **Visual Processing** → CLIP extracts visual features
7. **Format Outputs** → Captions, transcript, summary generated
8. **Results Available** → All outputs saved to `data/{video_name}/`

## Output Files

All processing outputs are organized in a structured directory per video:

```
data/{video_name}/
├── metadata/                    # Metadata and original video
│   ├── {video_name}.mp4        # Original video file
│   ├── video_metadata.json      # Video properties (duration, FPS, resolution, etc.)
│   ├── frames_info.json         # Frame extraction information
│   ├── audio_metadata.json      # Audio properties (sample rate, duration, etc.)
│   ├── audio_chunks_info.json   # Audio chunking information
│   ├── transcription.json       # Whisper transcription results
│   ├── speaker_diarization.json # Speaker identification results
│   ├── enhanced_transcript.json # Transcript with speaker attribution
│   ├── speaker_statistics.json  # Speaking time statistics
│   ├── slide_detection.json     # Slide transition detections
│   ├── visual_embeddings_info.json # Visual embeddings metadata
│   ├── generated_captions.json  # T5-generated captions
│   └── generated_summary.json   # T5-generated summary with key points
│
├── embeddings/                  # Embedding files (.npy)
│   ├── visual_embeddings.npy   # CLIP visual embeddings
│   ├── audio_embeddings.npy     # Whisper audio embeddings
│   └── fused_embeddings.npy     # Fused audio-visual embeddings
│
├── outputs/                     # Frontend-ready output files
│   ├── captions.srt            # SubRip subtitle format
│   ├── captions.vtt            # WebVTT subtitle format
│   ├── transcript.txt          # Full transcript with speaker attribution
│   └── summary.txt             # Generated summary text
│
├── frames/                      # Extracted video frames
│   └── frame_*.jpg             # Individual frame images
│
└── audio/                       # Extracted audio
    ├── {video_name}.wav         # Full audio track (16kHz, mono)
    └── chunks/                  # Audio chunks
        └── chunk_*.wav          # Processed audio chunks
```

See `Documentation/FOLDER_STRUCTURE.md` for detailed information about the directory structure.

## API Endpoints

- **Upload**: `POST /api/video/upload`
- **Process**: `POST /api/video/process`
- **Status**: `GET /api/video/status/{job_id}`
- **Results**: `GET /api/video/results/{job_id}`
- **Download**: `GET /api/video/download/{job_id}/{format}?type={type}`
- **Health**: `GET /api/health`

See `Documentation/PHASE5_API_GUIDE.md` for detailed API documentation.

## Project Structure

```
genAI-project/
├── src/
│   ├── preprocessing/      # Video and audio preprocessing
│   ├── audio/              # Speech-to-text and speaker diarization
│   ├── visual/             # Frame extraction and visual embeddings
│   ├── fusion/             # Multimodal transformer and cross-attention
│   ├── generation/         # Caption and summary generation
│   ├── output/             # Formatting for various output formats
│   ├── api/                # FastAPI backend
│   └── utils/               # Utility functions
├── frontend/               # React + TypeScript frontend
├── models/                 # Saved model checkpoints
├── data/                   # Input/output data
├── tests/                  # Unit and integration tests
├── notebooks/              # Jupyter notebooks for experiments, training, finetuning, etc
├── Documentation/          # Project documentation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Docker Configuration Files
```
genAI-project/
├── Dockerfile                  # Backend container configuration
├── docker-compose.yml          # Multi-container orchestration
├── .dockerignore              # Files to exclude from Docker build
├── .env                       # Environment variables template
└── frontend/
    ├── Dockerfile             # Frontend container configuration
    └── nginx.conf             # Nginx web server configuration
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd genAI-project
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Additional setup:
   - Install FFmpeg for video/audio processing
   - For pyannote.audio, you may need to accept Hugging Face model licenses
   - For OCR, install Tesseract (if using pytesseract)

## Features

- **Multimodal Processing**: Jointly processes audio and visual content using transformer architectures
- **Cross-Attention Fusion**: Bidirectional cross-attention between audio and visual embeddings
- **Multimodal Embeddings**: CLIP visual embeddings and Whisper audio embeddings with temporal alignment
- **Time-Synchronized Captions**: Generates accurate captions with timestamps using fused features
- **Speaker Diarization**: Identifies and attributes different speakers in multi-speaker videos
- **Abstractive Summarization**: Creates human-like summaries with improved grammar and key points
- **Multiple Output Formats**: Supports .srt, .vtt, .txt, .docx, and .pdf
- **Web Interface**: Modern React + TypeScript frontend with real-time progress tracking
- **REST API**: Complete FastAPI backend with automatic documentation
- **Structured Outputs**: All processing results saved as structured JSON files and organized directories

## Technology Stack

### Backend (Python)
- **API Framework**: FastAPI 0.104.0+, Uvicorn
- **Video Processing**: OpenCV, FFmpeg/FFprobe
- **Audio Processing**: OpenAI Whisper (speech-to-text & embeddings), pyannote.audio (speaker diarization)
- **Visual Processing**: CLIP (OpenAI) for visual embeddings, EasyOCR/pytesseract for OCR
- **Multimodal Fusion**: Custom PyTorch transformer with cross-attention mechanisms
- **Generation**: T5 transformers (Hugging Face) for caption and summary generation
- **ML Frameworks**: PyTorch
- **Data Validation**: Pydantic 2.0.0+

### Frontend (TypeScript + React)
- **Framework**: React 18.2.0
- **Language**: TypeScript 5.3.3
- **Routing**: React Router DOM 6.20.0
- **HTTP Client**: Axios 1.6.2
- **File Upload**: React Dropzone 14.2.3

## Troubleshooting

### Backend won't start
- Check if port 8000 is available
- Verify dependencies: `pip install -r requirements.txt`
- Check if FFmpeg is installed

### Frontend can't connect
- Verify backend is running on port 8000
- Check browser console for errors
- Verify CORS is configured (default allows all origins)

### Processing fails
- Check if video file is valid
- Verify FFmpeg is installed and in PATH
- Check backend logs for detailed errors
- Ensure sufficient disk space

## Documentation

- **`IMPLEMENTATION_GUIDE.md`** - Complete step-by-step implementation guide
- **`PROGRESS_SUMMARY.md`** - Current project status and progress
- **`Documentation/FOLDER_STRUCTURE.md`** - Complete output directory structure guide
- **`Documentation/PHASE2_DOCUMENTATION.md`** - Phase 2 modules documentation
- **`Documentation/PHASE2_JSON_OUTPUTS.md`** - JSON output file guide
- **`Documentation/PHASE5_API_GUIDE.md`** - Complete API documentation

## Development Status
Core functionality is complete and working!

**Completed:**
- ✅ Backend foundation and configuration
- ✅ Core processing modules (video, audio, visual)
- ✅ Multimodal fusion with cross-attention and transformer architecture
- ✅ Audio and visual embeddings extraction and fusion
- ✅ T5-based caption and summary generation
- ✅ Backend API with all endpoints
- ✅ Frontend application with full UI (React + TypeScript)
- ✅ End-to-end pipeline with multimodal fusion

## Next Steps

The system is fully functional! You can:

1. **Test with different videos** - Try various video formats and lengths
2. **Monitor processing** - Watch the progress in real-time
3. **Download results** - Get captions, transcripts, and summaries
4. **View JSON outputs** - Check detailed outputs in `data/{video_name}/`

For optional enhancements, see `IMPLEMENTATION_GUIDE.md` for Phases 3-4.

## License

*License information to be added.*

## Authors

Hafsa Imtiaz, Areen Zainab, and Areeba Riaz
