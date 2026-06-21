# Video Processing Folder Structure

## Overview

All data for each video is now organized in a single, structured folder. This makes it easy to manage, archive, and understand what files belong to which video.

## Directory Structure

```
data/
└── {video_name}/                    ← Single folder per video
    ├── metadata/                    ← Metadata and original video
    │   ├── {video_name}.mp4        ← Original video file
    │   ├── video_metadata.json
    │   ├── frames_info.json
    │   ├── audio_metadata.json
    │   ├── audio_chunks_info.json
    │   ├── transcription.json
    │   ├── speaker_diarization.json
    │   ├── enhanced_transcript.json
    │   ├── speaker_statistics.json
    │   ├── slide_detection.json
    │   ├── visual_embeddings_info.json
    │   ├── generated_captions.json
    │   └── generated_summary.json
    │
    ├── embeddings/                  ← Embedding files (.npy)
    │   ├── visual_embeddings.npy   ← CLIP visual embeddings
    │   ├── audio_embeddings.npy    ← Whisper audio embeddings (Phase 3)
    │   └── fused_embeddings.npy    ← Fused embeddings (Phase 3)
    │
    ├── outputs/                     ← Frontend-ready output files
    │   ├── captions.srt
    │   ├── captions.vtt
    │   ├── transcript.txt
    │   └── summary.txt
    │
    ├── frames/                      ← Extracted video frames
    │   └── frame_*.jpg
    │
    └── audio/                       ← Extracted audio
        ├── {video_name}.wav         ← Full audio track
        └── chunks/                  ← Audio chunks
            └── chunk_*.wav
```

## Folder Descriptions

### `metadata/`
**Purpose:** Stores the original video file and all JSON metadata files.

**Contents:**
- Original video file (moved from temp folder on upload)
- All JSON metadata files from processing steps
- Processing results and statistics

**Files:**
- `{video_name}.mp4` - Original uploaded video
- `video_metadata.json` - Video properties (duration, FPS, resolution, etc.)
- `frames_info.json` - Frame extraction information
- `audio_metadata.json` - Audio properties (sample rate, duration, etc.)
- `audio_chunks_info.json` - Audio chunking information
- `transcription.json` - Whisper transcription results
- `speaker_diarization.json` - Speaker identification results
- `enhanced_transcript.json` - Transcript with speaker attribution
- `speaker_statistics.json` - Speaking time statistics
- `slide_detection.json` - Slide transition detections
- `visual_embeddings_info.json` - Visual embeddings metadata
- `generated_captions.json` - T5-generated captions
- `generated_summary.json` - T5-generated summary with key points

---

### `embeddings/`
**Purpose:** Stores embedding arrays in NumPy format (.npy files).

**Contents:**
- Visual embeddings from CLIP
- Audio embeddings from Whisper (Phase 3)
- Fused multimodal embeddings (Phase 3)

**Files:**
- `visual_embeddings.npy` - CLIP visual embeddings (shape: num_frames, 512)
- `audio_embeddings.npy` - Whisper encoder embeddings (Phase 3)
- `fused_embeddings.npy` - Fused audio-visual embeddings (Phase 3)

**Usage:**
```python
import numpy as np
embeddings = np.load("data/{video_name}/embeddings/visual_embeddings.npy")
```

---

### `outputs/`
**Purpose:** Stores formatted output files ready for frontend download.

**Contents:**
- Caption files (SRT, VTT)
- Transcript files (TXT)
- Summary files (TXT)

**Files:**
- `captions.srt` - SubRip subtitle format
- `captions.vtt` - WebVTT subtitle format
- `transcript.txt` - Full transcript with speaker attribution
- `summary.txt` - Generated summary text

**Note:** These files are automatically generated during processing and can be downloaded directly by the frontend.

---

### `frames/`
**Purpose:** Stores extracted video frames as JPEG images.

**Contents:**
- Individual frame images extracted at specified FPS

**Files:**
- `frame_000001_1.000s.jpg` - Frame at 1.0 seconds
- `frame_000002_2.000s.jpg` - Frame at 2.0 seconds
- ... (one file per extracted frame)

**Naming Convention:**
- Format: `frame_{index:06d}_{timestamp:.3f}s.jpg`
- Index: Sequential frame number (zero-padded to 6 digits)
- Timestamp: Time in video (seconds, 3 decimal places)

---

### `audio/`
**Purpose:** Stores extracted audio and audio chunks.

**Contents:**
- Full audio track extracted from video
- Audio chunks for processing

**Files:**
- `{video_name}.wav` - Full audio track (16kHz, mono, WAV format)
- `chunks/chunk_0000_0.00s_30.00s.wav` - Audio chunk from 0-30 seconds
- `chunks/chunk_0001_30.00s_60.00s.wav` - Audio chunk from 30-60 seconds
- ... (one file per chunk)

**Naming Convention:**
- Format: `chunk_{index:04d}_{start_time:.2f}s_{end_time:.2f}s.wav`
- Index: Sequential chunk number (zero-padded to 4 digits)
- Start/End: Chunk time range in seconds

---

## Implementation Details

### Directory Creation
All directories are automatically created when needed using helper functions:
- `get_video_output_dir()` - Creates base directory and all subdirectories
- `get_video_metadata_dir()` - Returns metadata directory path
- `get_video_embeddings_dir()` - Returns embeddings directory path
- `get_video_outputs_dir()` - Returns outputs directory path
- `get_video_frames_dir()` - Returns frames directory path
- `get_video_audio_dir()` - Returns audio directory path

### File Paths
- All paths are relative to the video's base directory
- Frame paths in `frames_info.json` reference the `frames/` folder
- Chunk paths in `audio_chunks_info.json` reference the `audio/chunks/` folder
- Embedding file paths in metadata JSON reference the `embeddings/` folder

### Migration
- New videos automatically use the new structure
- Old videos in the previous structure remain accessible
- No automatic migration needed (can reprocess if desired)

---

## Benefits

1. **Organization:** All video data in one place
2. **Portability:** Easy to archive/backup per video
3. **Clarity:** Clear separation of data types
4. **Efficiency:** Easier to find and manage files
5. **Scalability:** Better for multiple videos
6. **Maintainability:** Easier to understand and debug

---

## Example

For a video named `lecture_abc123.mp4`, the structure would be:

```
data/
└── lecture_abc123/
    ├── metadata/
    │   ├── lecture_abc123.mp4
    │   ├── video_metadata.json
    │   ├── transcription.json
    │   └── ... (other JSON files)
    ├── embeddings/
    │   └── visual_embeddings.npy
    ├── outputs/
    │   ├── captions.srt
    │   ├── transcript.txt
    │   └── summary.txt
    ├── frames/
    │   ├── frame_000001_1.000s.jpg
    │   ├── frame_000002_2.000s.jpg
    │   └── ... (more frames)
    └── audio/
        ├── lecture_abc123.wav
        └── chunks/
            ├── chunk_0000_0.00s_30.00s.wav
            └── ... (more chunks)
```

---

**Last Updated:** [Current Date]
**Status:** ✅ Fully Implemented
