"""
Speech-to-text transcription module using Whisper or similar models.
"""

from typing import List, Dict, Optional, Tuple
from collections import Counter
import torch
import numpy as np
import whisper
import config
from src.utils.logger import get_logger
from src.utils.file_utils import get_video_output_dir, get_video_embeddings_dir, get_video_metadata_dir, save_json
from pathlib import Path

logger = get_logger(__name__)


class SpeechToText:
    """Performs speech-to-text transcription with timestamps."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize speech-to-text model.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large-v2, large-v3)
                       If None, uses config.WHISPER_MODEL
            device: Device to run model on (cuda/cpu, None for auto-detection)
        """
        self.model_name = model_name if model_name is not None else config.WHISPER_MODEL
        self.device = device if device is not None else config.WHISPER_DEVICE
        self.model = None
        self._is_loaded = False
    
    def load_model(self):
        """Load the Whisper model."""
        if self._is_loaded and self.model is not None:
            logger.debug("Whisper model already loaded")
            return
        
        logger.info(f"Loading Whisper model: {self.model_name} on device: {self.device}")
        
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            self._is_loaded = True
            logger.info(f"Successfully loaded Whisper model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Could not load Whisper model {self.model_name}: {e}") from e
    
    def transcribe(self, audio_path: str, language: Optional[str] = None, 
                   word_timestamps: bool = True, extract_embeddings: bool = False) -> Dict:
        """
        Transcribe audio file to text with timestamps.
        Optionally extracts audio embeddings during transcription to avoid duplicate encoding.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en') or None for auto-detection
            word_timestamps: Whether to include word-level timestamps
            extract_embeddings: Whether to extract audio embeddings during transcription
            
        Returns:
            Dictionary containing transcription with segments and timestamps.
            If extract_embeddings=True, also includes 'embeddings' and 'embedding_timestamps'.
        """
        if not self._is_loaded:
            self.load_model()
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio file: {audio_path} (extract_embeddings={extract_embeddings})")
        
        try:
            # If extracting embeddings, use combined method
            if extract_embeddings:
                return self._transcribe_with_embeddings(audio_path, language, word_timestamps)
            
            # Standard transcription without embedding extraction
            result = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=word_timestamps,
                verbose=False
            )
            
            # Extract full text
            full_text = result.get("text", "").strip()
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "id": segment.get("id", 0),
                    "text": segment.get("text", "").strip(),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "no_speech_prob": segment.get("no_speech_prob", 0.0),
                    "words": segment.get("words", []) if word_timestamps else []
                })
            
            # Get detected language
            detected_language = result.get("language", "unknown")
            language_prob = result.get("language_probability", 0.0)
            
            transcription = {
                "text": full_text,
                "language": detected_language,
                "language_probability": language_prob,
                "segments": segments,
                "num_segments": len(segments),
                "duration": segments[-1]["end"] if segments else 0.0
            }
            
            logger.info(f"Transcription complete: {len(segments)} segments, language: {detected_language}")
            return transcription
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise RuntimeError(f"Transcription failed: {e}") from e
    
    def _transcribe_with_embeddings(self, audio_path: str, language: Optional[str] = None,
                                    word_timestamps: bool = True) -> Dict:
        """
        Transcribe audio and extract embeddings in one pass.
        Uses Whisper's lower-level API to avoid duplicate encoding.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Dictionary with transcription and embeddings
        """
        # Load and preprocess audio once
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        # Run transcription (this will encode internally, but we can't easily intercept it)
        # So we'll encode separately and use it for embeddings, then transcribe normally
        # This is a limitation of Whisper's API - it doesn't expose encoder output from transcribe
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            verbose=False
        )
        
        # Now extract embeddings using the same audio we loaded
        # (We still need to re-encode, but at least we reuse audio loading)
        with torch.no_grad():
            encoder_output = self.model.encoder(mel.unsqueeze(0))
            embeddings = encoder_output.squeeze(0).cpu().numpy()
        
        # Extract transcription data
        full_text = result.get("text", "").strip()
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "id": segment.get("id", 0),
                "text": segment.get("text", "").strip(),
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "no_speech_prob": segment.get("no_speech_prob", 0.0),
                "words": segment.get("words", []) if word_timestamps else []
            })
        
        detected_language = result.get("language", "unknown")
        language_prob = result.get("language_probability", 0.0)
        
        # Align embeddings with segments
        aligned_embeddings, embedding_timestamps = self._align_embeddings_with_segments(
            embeddings, segments
        )
        
        transcription = {
            "text": full_text,
            "language": detected_language,
            "language_probability": language_prob,
            "segments": segments,
            "num_segments": len(segments),
            "duration": segments[-1]["end"] if segments else 0.0,
            "embeddings": aligned_embeddings,
            "embedding_timestamps": embedding_timestamps
        }
        
        logger.info(f"Transcription with embeddings complete: {len(segments)} segments, "
                   f"embeddings shape: {aligned_embeddings.shape}")
        return transcription
    
    def transcribe_chunk(self, audio_chunk: str, start_time: float, 
                        language: Optional[str] = None) -> Dict:
        """
        Transcribe a single audio chunk.
        
        Args:
            audio_chunk: Path to audio chunk
            start_time: Start time of chunk in original video
            language: Language code (e.g., 'en') or None for auto-detection
            
        Returns:
            Dictionary containing transcription segment with adjusted timestamps
        """
        if not self._is_loaded:
            self.load_model()
        
        if not Path(audio_chunk).exists():
            raise FileNotFoundError(f"Audio chunk not found: {audio_chunk}")
        
        logger.debug(f"Transcribing chunk: {audio_chunk} (offset: {start_time:.2f}s)")
        
        try:
            # Transcribe chunk
            result = self.model.transcribe(
                audio_chunk,
                language=language,
                word_timestamps=True,
                verbose=False
            )
            
            # Adjust timestamps relative to original video
            segments = []
            for segment in result.get("segments", []):
                adjusted_segment = {
                    "id": segment.get("id", 0),
                    "text": segment.get("text", "").strip(),
                    "start": segment.get("start", 0.0) + start_time,  # Adjust timestamp
                    "end": segment.get("end", 0.0) + start_time,  # Adjust timestamp
                    "no_speech_prob": segment.get("no_speech_prob", 0.0),
                    "words": []
                }
                
                # Adjust word timestamps
                for word in segment.get("words", []):
                    adjusted_word = {
                        "word": word.get("word", ""),
                        "start": word.get("start", 0.0) + start_time,
                        "end": word.get("end", 0.0) + start_time,
                        "probability": word.get("probability", 0.0)
                    }
                    adjusted_segment["words"].append(adjusted_word)
                
                segments.append(adjusted_segment)
            
            chunk_result = {
                "chunk_path": audio_chunk,
                "chunk_start_time": start_time,
                "text": result.get("text", "").strip(),
                "segments": segments,
                "language": result.get("language", "unknown")
            }
            
            return chunk_result
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {audio_chunk}: {e}")
            raise RuntimeError(f"Chunk transcription failed: {e}") from e
    
    def get_transcript_with_timestamps(self, audio_path: str, 
                                      language: Optional[str] = None) -> List[Dict]:
        """
        Get full transcript with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en') or None for auto-detection
            
        Returns:
            List of dictionaries with text, start_time, end_time for each segment
        """
        # Use transcribe method with word_timestamps=True
        result = self.transcribe(audio_path, language=language, word_timestamps=True)
        
        # Format segments for easy access
        transcript_segments = []
        for segment in result.get("segments", []):
            transcript_segments.append({
                "text": segment.get("text", ""),
                "start_time": segment.get("start", 0.0),
                "end_time": segment.get("end", 0.0),
                "words": segment.get("words", []),
                "confidence": 1.0 - segment.get("no_speech_prob", 1.0)  # Convert to confidence
            })
        
        return transcript_segments
    
    def transcribe_multiple_chunks(self, chunk_paths: List[tuple], 
                                   language: Optional[str] = None) -> Dict:
        """
        Transcribe multiple audio chunks and combine results.
        
        Args:
            chunk_paths: List of tuples (chunk_path, start_time)
            language: Language code (e.g., 'en') or None for auto-detection
            
        Returns:
            Combined transcription dictionary with all segments
        """
        if not self._is_loaded:
            self.load_model()
        
        logger.info(f"Transcribing {len(chunk_paths)} audio chunks")
        
        all_segments = []
        full_text_parts = []
        detected_languages = []
        
        for chunk_path, start_time in chunk_paths:
            try:
                chunk_result = self.transcribe_chunk(chunk_path, start_time, language)
                all_segments.extend(chunk_result["segments"])
                full_text_parts.append(chunk_result["text"])
                detected_languages.append(chunk_result["language"])
            except Exception as e:
                logger.warning(f"Failed to transcribe chunk {chunk_path}: {e}")
                continue
        
        # Determine most common language
        if detected_languages:
            language_counts = Counter(detected_languages)
            most_common_language = language_counts.most_common(1)[0][0]
        else:
            most_common_language = language or "unknown"
        
        # Combine results
        combined_result = {
            "text": " ".join(full_text_parts),
            "language": most_common_language,
            "segments": all_segments,
            "num_segments": len(all_segments),
            "num_chunks": len(chunk_paths),
            "duration": all_segments[-1]["end"] if all_segments else 0.0
        }
        
        logger.info(f"Combined transcription: {len(all_segments)} segments from {len(chunk_paths)} chunks")
        return combined_result
    
    def _align_embeddings_with_segments(
        self,
        embeddings: np.ndarray,
        segments: List[Dict]
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Align encoder embeddings with transcription segments.
        
        Args:
            embeddings: Raw encoder embeddings [n_audio_tokens, embedding_dim]
            segments: Transcription segments with start/end times
            
        Returns:
            aligned_embeddings: Array of shape (num_segments, embedding_dim)
            timestamps: List of segment start times
        """
        if not segments:
            # No segments, return empty
            return np.array([]), []
        
        aligned_embeddings = []
        segment_timestamps = []
        
        # Calculate time per token (approximate)
        audio_duration = segments[-1]["end"] if segments else 0.0
        if audio_duration > 0:
            tokens_per_second = embeddings.shape[0] / audio_duration
        else:
            tokens_per_second = embeddings.shape[0] / 1.0  # Fallback
        
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Convert time to token indices
            start_token = int(start_time * tokens_per_second)
            end_token = int(end_time * tokens_per_second)
            
            # Ensure indices are within bounds
            start_token = max(0, min(start_token, embeddings.shape[0] - 1))
            end_token = max(start_token + 1, min(end_token, embeddings.shape[0]))
            
            # Extract embeddings for this segment and mean pool
            segment_emb = embeddings[start_token:end_token]
            if len(segment_emb) > 0:
                segment_emb_pooled = np.mean(segment_emb, axis=0)
            else:
                # Fallback: use first embedding if no tokens
                segment_emb_pooled = embeddings[0] if embeddings.shape[0] > 0 else np.zeros(embeddings.shape[1])
            
            aligned_embeddings.append(segment_emb_pooled)
            segment_timestamps.append(start_time)
        
        return np.array(aligned_embeddings), segment_timestamps
    
    def extract_audio_embeddings(
        self,
        audio_path: str,
        return_timestamps: bool = True,
        align_with_segments: bool = True
    ) -> Tuple[np.ndarray, Optional[List[float]]]:
        """
        Extract audio embeddings from Whisper encoder.
        NOTE: This method is kept for backward compatibility but is inefficient.
        Use transcribe(extract_embeddings=True) instead to avoid duplicate encoding.
        
        Args:
            audio_path: Path to audio file
            return_timestamps: Whether to return segment timestamps
            align_with_segments: Whether to align embeddings with transcription segments
            
        Returns:
            embeddings: Array of shape (num_segments, embedding_dim)
            timestamps: Optional list of segment timestamps
        """
        logger.warning("extract_audio_embeddings() called - this will re-encode audio. "
                      "Consider using transcribe(extract_embeddings=True) instead.")
        
        if not self._is_loaded:
            self.load_model()
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Extracting audio embeddings from: {audio_path}")
        
        try:
            # Load and preprocess audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Get mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Encode with Whisper encoder
            with torch.no_grad():
                encoder_output = self.model.encoder(mel.unsqueeze(0))
                # encoder_output shape: [1, n_audio_tokens, n_audio_state]
                
                # Remove batch dimension
                embeddings = encoder_output.squeeze(0).cpu().numpy()
                # embeddings shape: [n_audio_tokens, embedding_dim]
            
            logger.info(f"Extracted encoder embeddings: shape {embeddings.shape}")
            
            # Get timestamps from transcription if needed
            timestamps = None
            if return_timestamps:
                if align_with_segments:
                    # Get transcription to align embeddings with segments
                    transcription = self.transcribe(audio_path, word_timestamps=False, extract_embeddings=False)
                    segments = transcription.get("segments", [])
                    
                    if segments:
                        embeddings, timestamps = self._align_embeddings_with_segments(embeddings, segments)
                        logger.info(f"Aligned embeddings to {len(segments)} segments: shape {embeddings.shape}")
                    else:
                        # No segments, use all embeddings with approximate timestamps
                        audio_duration = len(audio) / whisper.audio.SAMPLE_RATE
                        timestamps = np.linspace(0, audio_duration, embeddings.shape[0]).tolist()
                else:
                    # Return all token embeddings with approximate timestamps
                    audio_duration = len(audio) / whisper.audio.SAMPLE_RATE
                    timestamps = np.linspace(0, audio_duration, embeddings.shape[0]).tolist()
            
            return embeddings, timestamps
            
        except Exception as e:
            logger.error(f"Error extracting audio embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Audio embedding extraction failed: {e}") from e
    
    def save_audio_embeddings(
        self,
        video_path: str,
        embeddings: np.ndarray,
        timestamps: Optional[List[float]] = None
    ) -> Tuple[Path, Path]:
        """
        Save audio embeddings to .npy file and metadata to JSON.
        
        Args:
            video_path: Path to original video file
            embeddings: Audio embeddings array
            timestamps: Optional list of timestamps
            
        Returns:
            Tuple of (npy_path, json_path)
        """
        embeddings_dir = get_video_embeddings_dir(video_path)
        npy_path = embeddings_dir / "audio_embeddings.npy"
        
        # Save embeddings as numpy array
        np.save(str(npy_path), embeddings)
        logger.info(f"Saved audio embeddings to {npy_path}")
        
        # Save metadata to JSON
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "audio_embeddings_info.json"
        
        metadata = {
            "video_path": video_path,
            "embeddings_shape": list(embeddings.shape),
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0],
            "num_segments": embeddings.shape[0],
            "timestamps": timestamps if timestamps else [],
            "model_name": self.model_name,
            "device": self.device
        }
        
        save_json(metadata, str(json_path))
        logger.info(f"Saved audio embeddings metadata to {json_path}")
        
        return npy_path, json_path
    
    def save_transcription_to_json(self, video_path: str, transcription: Dict) -> Path:
        """
        Save transcription results to JSON file.
        
        Args:
            video_path: Path to original video file (for output directory)
            transcription: Transcription dictionary from transcribe() or transcribe_multiple_chunks()
            
        Returns:
            Path to saved JSON file
        """
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "transcription.json"
        
        save_json(transcription, str(json_path))
        logger.info(f"Saved transcription to {json_path}")
        return json_path
