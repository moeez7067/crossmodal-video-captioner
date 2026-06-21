"""
Audio extraction module for extracting audio tracks from video files.
"""

import subprocess
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import config
from src.utils.logger import get_logger
from src.utils.file_utils import ensure_directory, get_video_output_dir, get_video_audio_dir, get_video_metadata_dir, save_json

logger = get_logger(__name__)


class AudioExtractor:
    """Extracts audio tracks from video files."""
    
    def __init__(self, output_format: str = "wav", sample_rate: Optional[int] = None):
        """
        Initialize audio extractor.
        
        Args:
            output_format: Output audio format (default: wav)
            sample_rate: Sample rate in Hz (default: from config, 16kHz for Whisper)
        """
        self.output_format = output_format
        self.sample_rate = sample_rate if sample_rate is not None else config.AUDIO_SAMPLE_RATE

    def has_audio_stream(self, video_path: str) -> bool:
        """
        Check whether the given media file contains at least one audio stream.

        Returns False for videos with no audio track (e.g. silent stock clips),
        so callers can handle that case cleanly instead of failing on FFmpeg
        producing an empty output file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",  # audio streams only
            "-show_entries", "stream=codec_type",
            "-of", "json",
            str(video_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout or "{}")
            return len(data.get("streams", [])) > 0
        except subprocess.CalledProcessError as e:
            # If probing fails for some other reason, assume audio is present and
            # let the normal extraction path surface any real error.
            logger.warning(f"Could not probe audio streams ({e.stderr}); assuming audio present")
            return True
        except json.JSONDecodeError:
            return True
        except FileNotFoundError:
            error_msg = "FFprobe not found. Please install FFmpeg and ensure it's in your PATH."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio track from video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save extracted audio (if None, auto-generates)
            
        Returns:
            Path to extracted audio file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        video_path_obj = Path(video_path)
        
        # Generate output path if not provided - use organized structure
        if output_path is None:
            audio_dir = get_video_audio_dir(video_path)
            output_path = audio_dir / f"{video_path_obj.stem}.{self.output_format}"
        else:
            output_path = Path(output_path)
            ensure_directory(output_path.parent)
        
        output_path_str = str(output_path)
        
        logger.info(f"Extracting audio from {video_path} to {output_path_str}")
        
        # Build ffmpeg command
        # -i: input file
        # -vn: disable video
        # -acodec: audio codec (pcm_s16le for WAV)
        # -ar: sample rate
        # -ac: audio channels (1 for mono)
        # -y: overwrite output file
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le" if self.output_format == "wav" else "copy",
            "-ar", str(self.sample_rate),  # Sample rate
            "-ac", "1",  # Mono channel
            "-y",  # Overwrite output
            output_path_str
        ]
        
        try:
            # Run ffmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if not os.path.exists(output_path_str):
                raise IOError(f"Audio extraction failed: output file not created")
            
            logger.info(f"Successfully extracted audio to {output_path_str}")
            return output_path_str
            
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except FileNotFoundError:
            error_msg = "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def chunk_audio(self, audio_path: str, video_path: Optional[str] = None, chunk_duration: float = None) -> List[Tuple[str, float]]:
        """
        Split audio into chunks for processing.
        
        Args:
            audio_path: Path to audio file
            video_path: Path to original video file (for directory structure, optional)
            chunk_duration: Duration of each chunk in seconds (default: from config)
            
        Returns:
            List of tuples (chunk_path, start_time) for each audio chunk
        """
        if chunk_duration is None:
            chunk_duration = config.AUDIO_CHUNK_DURATION
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        audio_path_obj = Path(audio_path)
        
        # Get audio duration
        metadata = self.get_audio_metadata(audio_path)
        duration = metadata.get("duration", 0)
        
        if duration <= 0:
            raise ValueError(f"Invalid audio duration: {duration}")
        
        # Create chunks directory - use organized structure
        if video_path:
            # Use provided video path to get audio directory
            audio_dir = get_video_audio_dir(video_path)
        else:
            # Derive from audio path structure: data/{video_name}/audio/{audio}.wav
            # Go up one level to get audio dir, which is already correct
            audio_dir = audio_path_obj.parent
        
        chunks_dir = audio_dir / "chunks"
        ensure_directory(chunks_dir)
        
        logger.info(f"Chunking audio: {duration:.2f}s into {chunk_duration}s chunks")
        
        chunks = []
        start_time = 0.0
        chunk_index = 0
        
        while start_time < duration:
            end_time = min(start_time + chunk_duration, duration)
            chunk_duration_actual = end_time - start_time
            
            # Generate chunk filename
            chunk_filename = f"chunk_{chunk_index:04d}_{start_time:.2f}s_{end_time:.2f}s.{self.output_format}"
            chunk_path = chunks_dir / chunk_filename
            chunk_path_str = str(chunk_path)
            
            # Build ffmpeg command to extract chunk
            # Using input seeking (-ss before -i) for faster processing
            # Re-encoding to ensure valid chunks regardless of source format
            cmd = [
                "ffmpeg",
                "-ss", str(start_time),  # Start time (input seeking - faster)
                "-i", str(audio_path),
                "-t", str(chunk_duration_actual),  # Duration
                "-acodec", "pcm_s16le",  # Re-encode to WAV for reliability
                "-ar", str(self.sample_rate),  # Ensure correct sample rate
                "-ac", "1",  # Mono channel
                "-avoid_negative_ts", "make_zero",  # Handle timestamp edge cases
                "-y",  # Overwrite
                chunk_path_str
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=max(chunk_duration_actual * 2 + 10, 30)  # Safety timeout
                )
                
                # Validate chunk was created and has content
                if os.path.exists(chunk_path_str):
                    file_size = os.path.getsize(chunk_path_str)
                    if file_size > 0:
                        chunks.append((chunk_path_str, start_time))
                        logger.debug(f"Created chunk {chunk_index}: {chunk_path_str} (start: {start_time:.2f}s, size: {file_size} bytes)")
                    else:
                        logger.warning(f"Chunk file is empty: {chunk_path_str}")
                else:
                    logger.warning(f"Chunk file not created: {chunk_path_str}")
                
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout creating chunk {chunk_index} at {start_time:.2f}s")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error creating chunk {chunk_index}: {e.stderr}")
                # Continue with next chunk (non-critical for processing)
            
            start_time = end_time
            chunk_index += 1
        
        # Report chunking results
        total_attempted = chunk_index
        successful = len(chunks)
        failed = total_attempted - successful
        
        if failed > 0:
            logger.warning(f"Chunking completed: {successful} successful, {failed} failed out of {total_attempted} total")
            # Raise error if too many chunks failed (>10%)
            if failed > total_attempted * 0.1:
                raise RuntimeError(
                    f"Too many chunk failures: {failed}/{total_attempted} chunks failed. "
                    "This may indicate issues with the audio file or FFmpeg configuration."
                )
        else:
            logger.info(f"Successfully created {successful} audio chunks")
        
        return chunks
    
    def get_audio_metadata(self, audio_path: str) -> Dict:
        """
        Extract metadata from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing audio metadata (duration, sample_rate, channels, etc.)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Use ffprobe to get audio metadata
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration,size,bit_rate",
            "-show_entries", "stream=sample_rate,channels,codec_name",
            "-of", "json",
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            probe_data = json.loads(result.stdout)
            
            # Extract format information
            format_info = probe_data.get("format", {})
            duration = float(format_info.get("duration", 0))
            file_size = int(format_info.get("size", 0))
            bit_rate = int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else 0
            
            # Extract stream information (audio stream)
            streams = probe_data.get("streams", [])
            audio_stream = None
            for stream in streams:
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break
            
            if audio_stream:
                sample_rate = int(audio_stream.get("sample_rate", 0))
                channels = int(audio_stream.get("channels", 0))
                codec = audio_stream.get("codec_name", "unknown")
            else:
                sample_rate = 0
                channels = 0
                codec = "unknown"
            
            metadata = {
                "file_path": audio_path,
                "file_name": Path(audio_path).name,
                "file_size": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "duration": duration,
                "duration_formatted": self._format_duration(duration),
                "sample_rate": sample_rate,
                "channels": channels,
                "codec": codec,
                "bit_rate": bit_rate,
                "bit_rate_kbps": bit_rate / 1000 if bit_rate > 0 else 0,
            }
            
            return metadata
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFprobe error: {e.stderr}")
            raise RuntimeError(f"Failed to get audio metadata: {e.stderr}") from e
        except FileNotFoundError:
            error_msg = "FFprobe not found. Please install FFmpeg and ensure it's in your PATH."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ffprobe output: {e}")
            raise RuntimeError(f"Failed to parse audio metadata: {e}") from e
    
    def save_metadata_to_json(self, video_path: str, audio_path: str, metadata: Optional[Dict] = None) -> Path:
        """
        Save audio metadata to JSON file.
        
        Args:
            video_path: Path to original video file (for output directory)
            audio_path: Path to audio file
            metadata: Metadata dictionary (if None, will extract it)
            
        Returns:
            Path to saved JSON file
        """
        if metadata is None:
            metadata = self.get_audio_metadata(audio_path)
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "audio_metadata.json"
        
        save_json(metadata, str(json_path))
        logger.info(f"Saved audio metadata to {json_path}")
        return json_path
    
    def save_chunks_info_to_json(self, video_path: str, chunks: List[Tuple[str, float]]) -> Path:
        """
        Save audio chunk information to JSON file.
        
        Args:
            video_path: Path to original video file (for output directory)
            chunks: List of (chunk_path, start_time) tuples
            
        Returns:
            Path to saved JSON file
        """
        chunks_info = {
            "video_path": video_path,
            "total_chunks": len(chunks),
            "chunk_duration": config.AUDIO_CHUNK_DURATION,
            "chunks": [
                {
                    "chunk_path": chunk_path,
                    "start_time": start_time,
                    "chunk_index": idx
                }
                for idx, (chunk_path, start_time) in enumerate(chunks)
            ]
        }
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "audio_chunks_info.json"
        
        save_json(chunks_info, str(json_path))
        logger.info(f"Saved audio chunks info to {json_path}")
        return json_path
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string (e.g., "1h 23m 45s")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)
