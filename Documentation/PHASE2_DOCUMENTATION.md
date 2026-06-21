# Phase 2: Core Processing Modules - Documentation

## Overview

Phase 2 implements the core processing modules that extract and process audio and visual information from video files. This phase consists of three main components:

1. **Video Preprocessing Module** - Extracts frames and audio from videos
2. **Audio Processing Module** - Transcribes speech and identifies speakers
3. **Visual Processing Module** - Extracts visual features and detects on-screen content

---

## Module 1: Video Preprocessing

### Location
- `src/preprocessing/video_processor.py`
- `src/preprocessing/audio_extractor.py`

### What It Does
- Extracts video frames at specified intervals
- Extracts audio tracks from video files
- Splits audio into chunks for processing
- Extracts metadata from video and audio files

### Outputs

#### Video Processing Outputs:
1. **Extracted Frames** (`extract_frames()`)
   - **Location**: `data/temp/frames/{video_name}/`
   - **Format**: JPEG images
   - **Naming**: `frame_{index:06d}_{timestamp:.3f}s.jpg`
   - **Example**: `frame_000001_5.234s.jpg`
   - **Metadata**: List of tuples `(frame_path, timestamp)`

2. **Video Metadata** (`get_video_metadata()`)
   - **Format**: Python dictionary
   - **Contains**:
     ```python
     {
         "file_path": "path/to/video.mp4",
         "file_name": "video.mp4",
         "file_size": 104857600,  # bytes
         "file_size_mb": 100.0,
         "duration": 3600.0,  # seconds
         "duration_formatted": "1h 0m 0s",
         "fps": 30.0,
         "frame_count": 108000,
         "width": 1920,
         "height": 1080,
         "resolution": "1920x1080",
         "codec": "avc1",
         "aspect_ratio": 1.777
     }
     ```

#### Audio Processing Outputs:
1. **Extracted Audio** (`extract_audio()`)
   - **Location**: `data/temp/audio/{video_name}.wav`
   - **Format**: WAV (16kHz, mono)
   - **Example**: `video_lecture.wav`

2. **Audio Chunks** (`chunk_audio()`)
   - **Location**: `data/temp/audio/chunks/{video_name}/`
   - **Format**: WAV files
   - **Naming**: `chunk_{index:04d}_{start_time:.2f}s_{end_time:.2f}s.wav`
   - **Example**: `chunk_0000_0.00s_30.00s.wav`
   - **Metadata**: List of tuples `(chunk_path, start_time)`

3. **Audio Metadata** (`get_audio_metadata()`)
   - **Format**: Python dictionary
   - **Contains**:
     ```python
     {
         "file_path": "path/to/audio.wav",
         "file_name": "audio.wav",
         "file_size": 5760000,  # bytes
         "file_size_mb": 5.49,
         "duration": 360.0,  # seconds
         "duration_formatted": "6m 0s",
         "sample_rate": 16000,
         "channels": 1,
         "codec": "pcm_s16le",
         "bit_rate": 256000,
         "bit_rate_kbps": 256.0
     }
     ```

### How to Use

```python
from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.audio_extractor import AudioExtractor

# Initialize processors
video_processor = VideoProcessor(fps=1.0)  # Extract 1 frame per second
audio_extractor = AudioExtractor()

# Process video
video_path = "data/input/video.mp4"

# Extract frames
frames = video_processor.extract_frames(video_path)
# Output: [("data/temp/frames/video/frame_000001_1.000s.jpg", 1.0), ...]

# Get video metadata
metadata = video_processor.get_video_metadata(video_path)
print(f"Video duration: {metadata['duration_formatted']}")
print(f"Resolution: {metadata['resolution']}")

# Extract audio
audio_path = audio_extractor.extract_audio(video_path)
# Output: "data/temp/audio/video.wav"

# Chunk audio
chunks = audio_extractor.chunk_audio(audio_path, chunk_duration=30.0)
# Output: [("data/temp/audio/chunks/video/chunk_0000_0.00s_30.00s.wav", 0.0), ...]

# Get audio metadata
audio_metadata = audio_extractor.get_audio_metadata(audio_path)
print(f"Audio duration: {audio_metadata['duration_formatted']}")
print(f"Sample rate: {audio_metadata['sample_rate']} Hz")
```

### How to View Outputs

1. **View Extracted Frames**:
   ```bash
   # Navigate to frames directory
   cd data/temp/frames/video_name/
   
   # List all frames
   ls -la
   
   # View a specific frame (on Linux/Mac)
   xdg-open frame_000001_1.000s.jpg
   
   # Or use any image viewer
   ```

2. **Listen to Audio Chunks**:
   ```bash
   # Navigate to chunks directory
   cd data/temp/audio/chunks/video_name/
   
   # Play a chunk (on Linux)
   aplay chunk_0000_0.00s_30.00s.wav
   
   # Or use any audio player
   ```

---

## Module 2: Audio Processing

### Location
- `src/audio/speech_to_text.py`
- `src/audio/speaker_diarization.py`

### What It Does
- Transcribes audio to text using Whisper
- Identifies different speakers in the audio
- Assigns speaker labels to transcript segments
- Provides word-level timestamps

### Outputs

#### Speech-to-Text Outputs:
1. **Full Transcription** (`transcribe()`)
   - **Format**: Python dictionary
   - **Contains**:
     ```python
     {
         "text": "Full transcript text here...",
         "language": "en",
         "language_probability": 0.99,
         "segments": [
             {
                 "id": 0,
                 "text": "Hello, welcome to this lecture.",
                 "start": 0.0,
                 "end": 3.5,
                 "no_speech_prob": 0.01,
                 "words": [
                     {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.99},
                     ...
                 ]
             },
             ...
         ],
         "num_segments": 45,
         "duration": 360.0
     }
     ```

2. **Chunk Transcription** (`transcribe_chunk()`)
   - **Format**: Python dictionary
   - **Contains**:
     ```python
     {
         "chunk_path": "path/to/chunk.wav",
         "chunk_start_time": 30.0,
         "text": "Transcribed text from chunk...",
         "segments": [
             {
                 "id": 0,
                 "text": "Segment text",
                 "start": 30.5,  # Adjusted timestamp
                 "end": 33.2,     # Adjusted timestamp
                 "words": [...]
             }
         ],
         "language": "en"
     }
     ```

3. **Transcript with Timestamps** (`get_transcript_with_timestamps()`)
   - **Format**: List of dictionaries
   - **Contains**:
     ```python
     [
         {
             "text": "Hello, welcome to this lecture.",
             "start_time": 0.0,
             "end_time": 3.5,
             "words": [...],
             "confidence": 0.99
         },
         ...
     ]
     ```

#### Speaker Diarization Outputs:
1. **Diarization Results** (`diarize()`)
   - **Format**: List of dictionaries
   - **Contains**:
     ```python
     [
         {
             "speaker_id": "SPEAKER_00",
             "start_time": 0.0,
             "end_time": 15.3,
             "duration": 15.3
         },
         {
             "speaker_id": "SPEAKER_01",
             "start_time": 15.3,
             "end_time": 28.7,
             "duration": 13.4
         },
         ...
     ]
     ```

2. **Enhanced Transcript** (`assign_speakers_to_segments()`)
   - **Format**: List of dictionaries
   - **Contains**:
     ```python
     [
         {
             "text": "Hello, welcome to this lecture.",
             "start_time": 0.0,
             "end_time": 3.5,
             "speaker_id": "SPEAKER_00",
             "speaker_confidence": 0.95,
             "words": [...],
             "confidence": 0.99
         },
         ...
     ]
     ```

3. **Speaker Statistics** (`get_speaker_statistics()`)
   - **Format**: Python dictionary
   - **Contains**:
     ```python
     {
         "total_speakers": 2,
         "total_segments": 45,
         "total_duration": 360.0,
         "speakers": {
             "SPEAKER_00": {
                 "speaker_id": "SPEAKER_00",
                 "total_segments": 25,
                 "total_duration": 180.5,
                 "average_segment_duration": 7.22,
                 "percentage": 50.14
             },
             "SPEAKER_01": {
                 "speaker_id": "SPEAKER_01",
                 "total_segments": 20,
                 "total_duration": 179.5,
                 "average_segment_duration": 8.98,
                 "percentage": 49.86
             }
         },
         "speakers_sorted": [...]  # Sorted by duration
     }
     ```

### How to Use

```python
from src.audio.speech_to_text import SpeechToText
from src.audio.speaker_diarization import SpeakerDiarization

# Initialize models
stt = SpeechToText(model_name="base")
stt.load_model()

diarization = SpeakerDiarization()
diarization.load_model()

# Transcribe audio
audio_path = "data/temp/audio/video.wav"
transcription = stt.transcribe(audio_path)
print(f"Full text: {transcription['text']}")
print(f"Language: {transcription['language']}")
print(f"Number of segments: {transcription['num_segments']}")

# Get transcript with timestamps
transcript = stt.get_transcript_with_timestamps(audio_path)
for segment in transcript[:5]:  # First 5 segments
    print(f"[{segment['start_time']:.2f}s - {segment['end_time']:.2f}s] {segment['text']}")

# Speaker diarization
speakers = diarization.diarize(audio_path)
print(f"Detected {len(set(s['speaker_id'] for s in speakers))} speakers")

# Assign speakers to transcript
enhanced_transcript = diarization.assign_speakers_to_segments(transcript, speakers)
for segment in enhanced_transcript[:5]:
    print(f"[{segment['speaker_id']}] [{segment['start_time']:.2f}s] {segment['text']}")

# Get statistics
stats = diarization.get_speaker_statistics(speakers)
print(f"Total speakers: {stats['total_speakers']}")
for speaker_id, speaker_stats in stats['speakers'].items():
    print(f"{speaker_id}: {speaker_stats['percentage']:.1f}% speaking time")
```

### How to View Outputs

1. **View Transcription Results**:
   ```python
   # Print full transcript
   print(transcription['text'])
   
   # Print segments with timestamps
   for segment in transcription['segments']:
       print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
   
   # Save to file
   with open("transcript.txt", "w") as f:
       f.write(transcription['text'])
   ```

2. **View Speaker Attribution**:
   ```python
   # Print transcript with speakers
   for segment in enhanced_transcript:
       speaker = segment.get('speaker_id', 'Unknown')
       print(f"[{speaker}] {segment['text']}")
   
   # Save with speaker labels
   with open("transcript_with_speakers.txt", "w") as f:
       for segment in enhanced_transcript:
           speaker = segment.get('speaker_id', 'Unknown')
           f.write(f"[{speaker}] [{segment['start_time']:.2f}s] {segment['text']}\n")
   ```

3. **View Statistics**:
   ```python
   # Print statistics
   print(f"Total duration: {stats['total_duration']:.2f} seconds")
   print(f"Number of speakers: {stats['total_speakers']}")
   
   for speaker_id, speaker_stats in stats['speakers'].items():
       print(f"\n{speaker_id}:")
       print(f"  Segments: {speaker_stats['total_segments']}")
       print(f"  Duration: {speaker_stats['total_duration']:.2f}s")
       print(f"  Percentage: {speaker_stats['percentage']:.1f}%")
   ```

---

## Module 3: Visual Processing

### Location
- `src/visual/frame_extractor.py`
- `src/visual/visual_embeddings.py`

### What It Does
- Extracts frames from video as numpy arrays
- Preprocesses frames for model input
- Detects slide transitions
- Extracts on-screen text using OCR
- Generates visual embeddings using CLIP

### Outputs

#### Frame Extraction Outputs:
1. **Frame Arrays** (`extract_frames()`)
   - **Format**: List of tuples `(numpy_array, timestamp)`
   - **Array Shape**: `(height, width, 3)` - RGB format
   - **Example**: `(1080, 1920, 3)` for Full HD
   - **Data Type**: `numpy.uint8` (0-255)

2. **Preprocessed Frames** (`preprocess_frame()`)
   - **Format**: Numpy array
   - **Shape**: `(height, width, 3)` - e.g., `(224, 224, 3)`
   - **Data Type**: `numpy.float32` (0.0-1.0 normalized)

3. **Slide Detection Results** (`detect_slides()`)
   - **Format**: List of dictionaries
   - **Contains**:
     ```python
     [
         {
             "frame_index": 15,
             "timestamp": 15.0,
             "similarity": 0.25,
             "change_magnitude": 0.75
         },
         ...
     ]
     ```

4. **On-Screen Text** (`extract_on_screen_text()`)
   - **Format**: String
   - **Example**: `"Welcome to the Lecture\nChapter 1: Introduction"`

#### Visual Embeddings Outputs:
1. **Frame Embeddings** (`extract_embeddings()`)
   - **Format**: Numpy array
   - **Shape**: `(num_frames, embedding_dim)`
   - **Example**: `(360, 512)` for 360 frames with CLIP ViT-B/32
   - **Data Type**: `numpy.float32`
   - **Normalized**: Yes (L2 normalized)

2. **Pooled Embeddings** (`temporal_pooling()`)
   - **Format**: Numpy array
   - **Shape**: `(embedding_dim,)`
   - **Example**: `(512,)` for CLIP ViT-B/32
   - **Methods**: mean, max, or attention pooling

3. **Aligned Embeddings** (`align_with_audio()`)
   - **Format**: Numpy array
   - **Shape**: `(num_audio_segments, embedding_dim)`
   - **Example**: `(12, 512)` for 12 audio segments
   - **Purpose**: Visual embeddings aligned to audio timestamps

### How to Use

```python
from src.visual.frame_extractor import FrameExtractor
from src.visual.visual_embeddings import VisualEmbeddings

# Initialize processors
frame_extractor = FrameExtractor(target_fps=1.0)
visual_embeddings = VisualEmbeddings()
visual_embeddings.load_model()

# Extract frames
video_path = "data/input/video.mp4"
frames_with_timestamps = frame_extractor.extract_frames(video_path)
frames = [frame for frame, _ in frames_with_timestamps]
timestamps = [ts for _, ts in frames_with_timestamps]

# Preprocess frames
preprocessed = [frame_extractor.preprocess_frame(frame) for frame in frames[:5]]

# Detect slide transitions
slide_changes = frame_extractor.detect_slides(frames, timestamps)
print(f"Detected {len(slide_changes)} slide transitions")
for change in slide_changes:
    print(f"Slide change at {change['timestamp']:.2f}s")

# Extract on-screen text
text_from_frame = frame_extractor.extract_on_screen_text(frames[0])
print(f"Text in first frame: {text_from_frame}")

# Extract visual embeddings
embeddings = visual_embeddings.extract_embeddings(frames)
print(f"Embeddings shape: {embeddings.shape}")

# Pool embeddings
pooled = visual_embeddings.temporal_pooling(embeddings, method="mean")
print(f"Pooled embedding shape: {pooled.shape}")

# Align with audio timestamps
audio_timestamps = [0.0, 30.0, 60.0, 90.0]  # Example audio segment starts
aligned = visual_embeddings.align_with_audio(embeddings, timestamps, audio_timestamps)
print(f"Aligned embeddings shape: {aligned.shape}")
```

### How to View Outputs

1. **View Frame Arrays**:
   ```python
   import matplotlib.pyplot as plt
   
   # Display a frame
   frame, timestamp = frames_with_timestamps[0]
   plt.imshow(frame)
   plt.title(f"Frame at {timestamp:.2f}s")
   plt.axis('off')
   plt.show()
   
   # Save frame
   from PIL import Image
   img = Image.fromarray(frame)
   img.save("frame_0.png")
   ```

2. **View Slide Detection**:
   ```python
   # Print slide transitions
   for change in slide_changes:
       print(f"Slide transition at {change['timestamp']:.2f}s "
             f"(similarity: {change['similarity']:.2f})")
   
   # Visualize on timeline
   import matplotlib.pyplot as plt
   timestamps = [c['timestamp'] for c in slide_changes]
   plt.scatter(timestamps, [1] * len(timestamps))
   plt.xlabel('Time (seconds)')
   plt.title('Slide Transitions')
   plt.show()
   ```

3. **View Embeddings**:
   ```python
   # Check embedding properties
   print(f"Embeddings shape: {embeddings.shape}")
   print(f"Embedding dimension: {embeddings.shape[1]}")
   print(f"Number of frames: {embeddings.shape[0]}")
   print(f"Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
   
   # Visualize embedding similarity (optional)
   from sklearn.metrics.pairwise import cosine_similarity
   similarity_matrix = cosine_similarity(embeddings[:10])  # First 10 frames
   plt.imshow(similarity_matrix, cmap='viridis')
   plt.colorbar()
   plt.title('Frame Embedding Similarity')
   plt.show()
   ```

---

## Complete Phase 2 Pipeline Example

```python
from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.audio_extractor import AudioExtractor
from src.audio.speech_to_text import SpeechToText
from src.audio.speaker_diarization import SpeakerDiarization
from src.visual.frame_extractor import FrameExtractor
from src.visual.visual_embeddings import VisualEmbeddings

# Initialize all processors
video_processor = VideoProcessor(fps=1.0)
audio_extractor = AudioExtractor()
stt = SpeechToText()
diarization = SpeakerDiarization()
frame_extractor = FrameExtractor(target_fps=1.0)
visual_embeddings = VisualEmbeddings()

# Load models
stt.load_model()
diarization.load_model()
visual_embeddings.load_model()

# Process video
video_path = "data/input/lecture.mp4"

# Step 1: Extract frames and audio
frames_data = video_processor.extract_frames(video_path)
audio_path = audio_extractor.extract_audio(video_path)
chunks = audio_extractor.chunk_audio(audio_path)

# Step 2: Transcribe audio
transcription = stt.transcribe(audio_path)
speakers = diarization.diarize(audio_path)
enhanced_transcript = diarization.assign_speakers_to_segments(
    transcription['segments'], speakers
)

# Step 3: Extract visual features
frames = [frame for frame, _ in frames_data]
timestamps = [ts for _, ts in frames_data]
embeddings = visual_embeddings.extract_embeddings(frames)
slide_changes = frame_extractor.detect_slides(frames, timestamps)

# Step 4: Align visual with audio
audio_segment_starts = [seg['start_time'] for seg in enhanced_transcript]
aligned_visual = visual_embeddings.align_with_audio(
    embeddings, timestamps, audio_segment_starts
)

# Output summary
print(f"Video processed: {video_path}")
print(f"Frames extracted: {len(frames)}")
print(f"Audio chunks: {len(chunks)}")
print(f"Transcript segments: {len(enhanced_transcript)}")
print(f"Speakers detected: {len(set(s['speaker_id'] for s in speakers))}")
print(f"Visual embeddings: {embeddings.shape}")
print(f"Slide transitions: {len(slide_changes)}")
```

---

## Output File Locations

All outputs are stored in the following directory structure:

```
data/
├── temp/
│   ├── frames/
│   │   └── {video_name}/
│   │       └── frame_*.jpg
│   └── audio/
│       ├── {video_name}.wav
│       └── chunks/
│           └── {video_name}/
│               └── chunk_*.wav
└── output/
    └── {job_id}/
        └── (Final outputs from later phases)
```

---

## Dependencies

### Required Packages:
- `opencv-python` - Video and image processing
- `ffmpeg-python` or `ffmpeg` - Audio extraction
- `openai-whisper` - Speech-to-text
- `pyannote.audio` - Speaker diarization
- `clip-by-openai` - Visual embeddings
- `easyocr` or `pytesseract` - OCR (optional)
- `scikit-image` - Frame similarity (optional, has fallback)

### Installation:
```bash
pip install opencv-python openai-whisper pyannote.audio clip-by-openai easyocr
# Or use requirements.txt
pip install -r requirements.txt
```

---

## Troubleshooting

### Common Issues:

1. **FFmpeg not found**
   - Install FFmpeg: `sudo apt-get install ffmpeg` (Linux) or download from ffmpeg.org
   - Ensure it's in your PATH

2. **Whisper model download**
   - Models are downloaded automatically on first use
   - Ensure internet connection for first run

3. **CLIP installation**
   - Install from GitHub: `pip install git+https://github.com/openai/CLIP.git`
   - Requires PyTorch

4. **Speaker diarization authentication**
   - Set `HUGGING_FACE_TOKEN` in `.env` file
   - Accept model license on Hugging Face

5. **OCR not working**
   - Install Tesseract: `sudo apt-get install tesseract-ocr` (Linux)
   - Or use EasyOCR which doesn't require separate installation

---

**Last Updated:** December 6, 2025

