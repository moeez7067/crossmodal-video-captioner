"""
Speaker diarization module for identifying and attributing speakers.
"""

from typing import List, Dict, Optional
import warnings
import torch
import config
from src.utils.logger import get_logger
from src.utils.file_utils import get_video_output_dir, save_json
from pathlib import Path

logger = get_logger(__name__)

# Suppress known harmless warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*degrees of freedom.*", category=UserWarning)


class SpeakerDiarization:
    """Performs speaker diarization to identify different speakers."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize speaker diarization model.
        
        Args:
            model_name: Model identifier for pyannote.audio (default: from config)
        """
        self.model_name = model_name if model_name is not None else config.DIARIZATION_MODEL
        self.model = None
        self._is_loaded = False
    
    def load_model(self):
        """Load the speaker diarization model."""
        if self._is_loaded and self.model is not None:
            logger.debug("Speaker diarization model already loaded")
            return
        
        logger.info(f"Loading speaker diarization model: {self.model_name}")
        
        try:
            from pyannote.audio import Pipeline
            
            # Fix for PyTorch 2.6+: Handle weights_only parameter change
            # PyTorch 2.6 changed torch.load default to weights_only=True for security
            # The model checkpoint contains pyannote.audio classes which need to be allowlisted
            original_torch_load = torch.load
            torch_load_patched = False
            
            # Try to add required classes to safe globals for PyTorch 2.6+
            safe_globals_list = []
            try:
                # Add TorchVersion if available
                if hasattr(torch, 'torch_version') and hasattr(torch.torch_version, 'TorchVersion'):
                    safe_globals_list.append(torch.torch_version.TorchVersion)
                
                # Add pyannote.audio.core.task.Specifications
                try:
                    from pyannote.audio.core.task import Specifications
                    safe_globals_list.append(Specifications)
                except ImportError:
                    logger.debug("Could not import pyannote.audio.core.task.Specifications")
                
                # Add safe globals if API is available
                if safe_globals_list and hasattr(torch.serialization, 'add_safe_globals'):
                    torch.serialization.add_safe_globals(safe_globals_list)
                    logger.debug(f"Added {len(safe_globals_list)} classes to safe globals for PyTorch 2.6+ compatibility")
            except (AttributeError, TypeError) as e:
                logger.debug(f"Could not configure safe globals, will use fallback: {e}")
            
            # Patch torch.load to force weights_only=False
            # This is needed because pyannote.audio calls torch.load internally
            # and PyTorch 2.6+ defaults to weights_only=True
            def patched_torch_load(*args, **kwargs):
                # Force weights_only=False to allow loading pyannote.audio models
                # This is safe since we're loading from trusted Hugging Face models
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            # Temporarily patch torch.load during model loading
            torch.load = patched_torch_load
            torch_load_patched = True
            logger.debug("Temporarily patched torch.load for PyTorch 2.6+ compatibility")
            
            try:
                # Check for Hugging Face token
                hf_token = config.HUGGING_FACE_TOKEN
                if not hf_token:
                    logger.warning(
                        "HUGGING_FACE_TOKEN not set. Some models may require authentication. "
                        "Set it in .env file or environment variables."
                    )
                
                # Load pipeline
                if hf_token:
                    self.model = Pipeline.from_pretrained(
                        self.model_name,
                        token=hf_token
                    )
                else:
                    # Try without token (may fail for private models)
                    self.model = Pipeline.from_pretrained(self.model_name)
            finally:
                # Restore original torch.load
                if torch_load_patched:
                    torch.load = original_torch_load
                    logger.debug("Restored original torch.load")
            
            # Move to appropriate device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            self._is_loaded = True
            logger.info(f"Successfully loaded speaker diarization model on {device}")
            
        except ImportError:
            error_msg = (
                "pyannote.audio not installed. Install it with: "
                "pip install pyannote.audio"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            error_str = str(e)
            logger.error(f"Failed to load speaker diarization model: {e}")
            
            # Check for gated repository access errors
            if "gated repo" in error_str.lower() or "not in the authorized list" in error_str.lower() or "restricted" in error_str.lower():
                error_msg = (
                    f"Access denied to gated Hugging Face repository.\n\n"
                    f"To fix this:\n"
                    f"1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the terms\n"
                    f"2. Visit https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the terms\n"
                    f"3. Make sure your HUGGING_FACE_TOKEN in .env has 'read' permission\n"
                    f"4. Get your token from: https://huggingface.co/settings/tokens\n\n"
                    f"Original error: {error_str}"
                )
                raise RuntimeError(error_msg) from e
            
            raise RuntimeError(f"Could not load diarization model {self.model_name}: {e}") from e
    
    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of dictionaries with speaker_id, start_time, end_time for each segment
        """
        if not self._is_loaded:
            self.load_model()
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Running speaker diarization on: {audio_path}")
        
        try:
            # Load audio as waveform to avoid AudioDecoder dependency
            # This works around the torchcodec issue where AudioDecoder is not available
            try:
                import soundfile as sf
                waveform, sample_rate = sf.read(audio_path)
                
                # Convert to torch tensor if needed
                if not isinstance(waveform, torch.Tensor):
                    waveform = torch.from_numpy(waveform).float()
                
                # Ensure waveform is 2D: (channels, time)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # Add channel dimension
                elif waveform.dim() == 2 and waveform.shape[0] > waveform.shape[1]:
                    # If shape is (time, channels), transpose to (channels, time)
                    waveform = waveform.transpose(0, 1)
                
                # Create waveform dictionary as expected by pyannote.audio
                audio_input = {
                    "waveform": waveform,
                    "sample_rate": sample_rate
                }
                
                logger.debug(f"Loaded audio: shape={waveform.shape}, sample_rate={sample_rate}")
                
                # Run diarization with preloaded waveform
                diarization = self.model(audio_input)
                
            except ImportError:
                # Fallback: try with file path (may fail if torchcodec not available)
                logger.warning("soundfile not available, trying file path (may fail if torchcodec not installed)")
                diarization = self.model(str(audio_path))
            except Exception as load_error:
                # If loading with soundfile fails, try fallback to file path
                logger.warning(f"Failed to load audio with soundfile: {load_error}. Trying file path fallback.")
                try:
                    diarization = self.model(str(audio_path))
                except Exception as fallback_error:
                    # Provide helpful error message
                    error_msg = (
                        f"Failed to load audio for diarization. "
                        f"Soundfile error: {load_error}. "
                        f"File path error: {fallback_error}. "
                        f"Make sure the audio file is valid and accessible."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from fallback_error
            
            # Convert to list of dictionaries
            # DiarizeOutput is a wrapper object that contains speaker_diarization attribute
            # which is the actual pyannote.core.Annotation object
            segments = []
            
            try:
                # Extract the actual annotation from DiarizeOutput
                # DiarizeOutput has a 'speaker_diarization' attribute that contains the Annotation
                if hasattr(diarization, 'speaker_diarization'):
                    annotation = diarization.speaker_diarization
                    logger.debug("Using speaker_diarization attribute")
                elif hasattr(diarization, 'exclusive_speaker_diarization'):
                    # Some versions might use exclusive_speaker_diarization
                    annotation = diarization.exclusive_speaker_diarization
                    logger.debug("Using exclusive_speaker_diarization attribute")
                else:
                    # If it's already an Annotation object, use it directly
                    annotation = diarization
                    logger.debug("Using diarization object directly")
                
                # Validate annotation
                if annotation is None:
                    raise ValueError("Annotation is None - diarization may have failed")
                
                logger.debug(f"Annotation type: {type(annotation).__name__}")
                logger.debug(f"Annotation methods: {[m for m in dir(annotation) if not m.startswith('_')][:10]}")
                
                # Check if annotation has any content
                if hasattr(annotation, '__len__'):
                    ann_len = len(annotation)
                    logger.debug(f"Annotation length: {ann_len}")
                    if ann_len == 0:
                        logger.warning("Annotation is empty - no speaker segments found")
                        return []
                
                # Now process the annotation object (pyannote.core.Annotation)
                # Use itertracks() which is the recommended way to iterate over annotations
                # It yields (segment, track, label) tuples
                if hasattr(annotation, 'itertracks'):
                    # Preferred method: use itertracks
                    logger.debug("Using itertracks method")
                    try:
                        for segment, track, speaker in annotation.itertracks(yield_label=True):
                            # Handle case where speaker might be a set or other type
                            if isinstance(speaker, (set, list, tuple)):
                                # If multiple speakers, take the first one
                                speaker = list(speaker)[0] if speaker else None
                            
                            # Convert to string and handle None/unknown cases
                            if speaker is None or str(speaker).strip() == '' or str(speaker) == '_':
                                speaker_id = None  # Unknown speaker
                            else:
                                speaker_id = str(speaker)
                            
                            segments.append({
                                "speaker_id": speaker_id,
                                "start_time": segment.start,
                                "end_time": segment.end,
                                "duration": segment.end - segment.start
                            })
                    except Exception as iter_error:
                        logger.error(f"Error in itertracks: {iter_error}")
                        raise
                elif hasattr(annotation, 'get_timeline'):
                    # Fallback: use get_timeline + indexing
                    try:
                        timeline = annotation.get_timeline()
                        logger.debug(f"Got timeline with {len(timeline)} segments")
                        for segment in timeline:
                            try:
                                # Get speaker label for this segment using annotation indexing
                                # Annotation can return a set of labels, so handle that
                                speaker_labels = annotation[segment]
                                
                                # Handle case where annotation returns a set or list
                                if isinstance(speaker_labels, (set, list, tuple)):
                                    # If multiple speakers, take the first one
                                    speaker = list(speaker_labels)[0] if speaker_labels else None
                                else:
                                    speaker = speaker_labels
                                
                                # Handle None/unknown cases
                                if speaker is None or str(speaker).strip() == '' or str(speaker) == '_':
                                    speaker_id = None
                                else:
                                    speaker_id = str(speaker)
                                
                                segments.append({
                                    "speaker_id": speaker_id,
                                    "start_time": segment.start,
                                    "end_time": segment.end,
                                    "duration": segment.end - segment.start
                                })
                            except (KeyError, IndexError) as seg_error:
                                # Segment might not have a speaker label
                                logger.debug(f"Segment {segment} has no speaker label: {seg_error}")
                                segments.append({
                                    "speaker_id": None,
                                    "start_time": segment.start,
                                    "end_time": segment.end,
                                    "duration": segment.end - segment.start
                                })
                            except Exception as seg_error:
                                logger.warning(f"Error processing segment {segment}: {seg_error}")
                                continue
                    except Exception as timeline_error:
                        logger.error(f"Error getting timeline: {timeline_error}")
                        raise
                else:
                    # Try direct iteration: Annotation objects support iteration
                    # Format: for (segment, track, label) in annotation
                    logger.debug("Trying direct iteration")
                    try:
                        for segment, track, label in annotation:
                            segments.append({
                                "speaker_id": str(label),
                                "start_time": segment.start,
                                "end_time": segment.end,
                                "duration": segment.end - segment.start
                            })
                    except (TypeError, ValueError) as iter_error:
                        logger.error(f"Direct iteration failed: {iter_error}")
                        raise AttributeError(f"Annotation object does not support iteration: {iter_error}") from iter_error
                
                if not segments:
                    logger.warning("No segments extracted from diarization output")
                    # Return empty list instead of failing
                    return []
                        
            except Exception as api_error:
                # Provide helpful error message with object inspection
                diarization_type = type(diarization).__name__
                available_methods = [m for m in dir(diarization) if not m.startswith('_')]
                
                # Get full exception details
                error_str = str(api_error)
                error_repr = repr(api_error)
                error_type = type(api_error).__name__
                
                logger.error(f"Error parsing diarization output")
                logger.error(f"  Error type: {error_type}")
                logger.error(f"  Error string: {error_str}")
                logger.error(f"  Error repr: {error_repr}")
                logger.error(f"  Diarization object type: {diarization_type}")
                logger.error(f"  Available methods: {available_methods}")
                
                # Try to get more info about the annotation if we got that far
                if 'annotation' in locals():
                    try:
                        logger.error(f"  Annotation type: {type(annotation).__name__}")
                        ann_methods = [m for m in dir(annotation) if not m.startswith('_')]
                        logger.error(f"  Annotation methods: {ann_methods[:20]}")
                        if hasattr(annotation, '__len__'):
                            logger.error(f"  Annotation length: {len(annotation)}")
                    except Exception as ann_error:
                        logger.error(f"  Could not inspect annotation: {ann_error}")
                
                raise RuntimeError(
                    f"Failed to parse diarization output. "
                    f"Type: {diarization_type}, "
                    f"Error type: {error_type}, "
                    f"Error: {error_str} ({error_repr}). "
                    f"Available methods: {available_methods}"
                ) from api_error
            
            # Sort by start time
            segments.sort(key=lambda x: x["start_time"])
            
            # Get unique speakers
            unique_speakers = list(set(seg["speaker_id"] for seg in segments))
            
            logger.info(
                f"Diarization complete: {len(segments)} segments, "
                f"{len(unique_speakers)} speakers detected"
            )
            
            return segments
            
        except Exception as e:
            logger.error(f"Error during diarization: {e}")
            raise RuntimeError(f"Diarization failed: {e}") from e
    
    def assign_speakers_to_segments(self, transcript_segments: List[Dict], 
                                    diarization_results: List[Dict]) -> List[Dict]:
        """
        Assign speaker labels to transcript segments.
        
        Args:
            transcript_segments: List of transcript segments with timestamps
                               Each segment should have 'start_time' and 'end_time'
            diarization_results: List of diarization results with speaker IDs
                               Each result should have 'speaker_id', 'start_time', 'end_time'
            
        Returns:
            List of transcript segments with speaker attribution
        """
        if not transcript_segments:
            return []
        
        if not diarization_results:
            logger.warning("No diarization results provided, returning segments without speaker attribution")
            for segment in transcript_segments:
                segment["speaker_id"] = None
            return transcript_segments
        
        logger.info(f"Assigning speakers to {len(transcript_segments)} transcript segments")
        
        # Create enhanced segments with speaker attribution
        enhanced_segments = []
        diarization_idx = 0
        
        for transcript_seg in transcript_segments:
            seg_start = transcript_seg.get("start_time", 0.0)
            seg_end = transcript_seg.get("end_time", 0.0)
            seg_mid = (seg_start + seg_end) / 2.0
            
            # Find the diarization segment that overlaps most with transcript segment
            best_speaker = None
            best_overlap = 0.0
            
            for diar_seg in diarization_results:
                diar_start = diar_seg.get("start_time", 0.0)
                diar_end = diar_seg.get("end_time", 0.0)
                
                # Calculate overlap
                overlap_start = max(seg_start, diar_start)
                overlap_end = min(seg_end, diar_end)
                overlap = max(0.0, overlap_end - overlap_start)
                
                # Use the segment with the most overlap
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg.get("speaker_id")
                
                # Also check if midpoint falls within diarization segment
                if diar_start <= seg_mid <= diar_end:
                    if overlap > best_overlap or best_speaker is None:
                        best_speaker = diar_seg.get("speaker_id")
                        best_overlap = overlap
            
            # Create enhanced segment
            enhanced_segment = transcript_seg.copy()
            enhanced_segment["speaker_id"] = best_speaker
            enhanced_segment["speaker_confidence"] = min(1.0, best_overlap / (seg_end - seg_start)) if (seg_end - seg_start) > 0 else 0.0
            
            enhanced_segments.append(enhanced_segment)
        
        # Count segments per speaker
        speaker_counts = {}
        for seg in enhanced_segments:
            speaker = seg.get("speaker_id", "unknown")
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        logger.info(f"Speaker assignment complete. Distribution: {speaker_counts}")
        return enhanced_segments
    
    def get_speaker_statistics(self, diarization_results: List[Dict]) -> Dict:
        """
        Get statistics about speakers (total speaking time, number of segments, etc.).
        
        Args:
            diarization_results: List of diarization results
            
        Returns:
            Dictionary with speaker statistics
        """
        if not diarization_results:
            return {
                "total_speakers": 0,
                "total_segments": 0,
                "total_duration": 0.0,
                "speakers": {}
            }
        
        # Calculate statistics per speaker
        speaker_stats = {}
        total_duration = 0.0
        
        for segment in diarization_results:
            speaker_id = segment.get("speaker_id", "unknown")
            duration = segment.get("duration", 0.0)
            
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    "speaker_id": speaker_id,
                    "total_segments": 0,
                    "total_duration": 0.0,
                    "average_segment_duration": 0.0,
                    "percentage": 0.0
                }
            
            speaker_stats[speaker_id]["total_segments"] += 1
            speaker_stats[speaker_id]["total_duration"] += duration
            total_duration += duration
        
        # Calculate percentages and averages
        for speaker_id, stats in speaker_stats.items():
            if stats["total_segments"] > 0:
                stats["average_segment_duration"] = (
                    stats["total_duration"] / stats["total_segments"]
                )
            if total_duration > 0:
                stats["percentage"] = (stats["total_duration"] / total_duration) * 100
        
        # Sort speakers by total duration (descending)
        sorted_speakers = sorted(
            speaker_stats.values(),
            key=lambda x: x["total_duration"],
            reverse=True
        )
        
        statistics = {
            "total_speakers": len(speaker_stats),
            "total_segments": len(diarization_results),
            "total_duration": total_duration,
            "speakers": {speaker_id: stats for speaker_id, stats in speaker_stats.items()},
            "speakers_sorted": sorted_speakers
        }
        
        logger.debug(f"Speaker statistics: {statistics['total_speakers']} speakers, "
                    f"{statistics['total_segments']} segments, "
                    f"{statistics['total_duration']:.2f}s total duration")
        
        return statistics
    
    def get_speaker_labels(self, diarization_results: List[Dict]) -> Dict[str, str]:
        """
        Generate human-readable speaker labels from speaker IDs.
        
        Args:
            diarization_results: List of diarization results
            
        Returns:
            Dictionary mapping speaker_id to human-readable label (e.g., "Speaker 1")
        """
        unique_speakers = sorted(list(set(
            seg.get("speaker_id", "unknown") for seg in diarization_results
        )))
        
        speaker_labels = {}
        for idx, speaker_id in enumerate(unique_speakers, 1):
            speaker_labels[speaker_id] = f"Speaker {idx}"
        
        return speaker_labels
    
    def save_diarization_to_json(self, video_path: str, diarization_results: List[Dict]) -> Path:
        """
        Save speaker diarization results to JSON file.
        
        Args:
            video_path: Path to original video file (for output directory)
            diarization_results: List of diarization results from diarize()
            
        Returns:
            Path to saved JSON file
        """
        from src.utils.file_utils import get_video_metadata_dir
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "speaker_diarization.json"
        
        diarization_data = {
            "video_path": video_path,
            "total_segments": len(diarization_results),
            "unique_speakers": len(set(seg.get("speaker_id") for seg in diarization_results)),
            "segments": diarization_results
        }
        
        save_json(diarization_data, str(json_path))
        logger.info(f"Saved speaker diarization to {json_path}")
        return json_path
    
    def save_enhanced_transcript_to_json(self, video_path: str, enhanced_segments: List[Dict]) -> Path:
        """
        Save enhanced transcript with speaker attribution to JSON file.
        
        Args:
            video_path: Path to original video file (for output directory)
            enhanced_segments: List of transcript segments with speaker attribution
            
        Returns:
            Path to saved JSON file
        """
        from src.utils.file_utils import get_video_metadata_dir
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "enhanced_transcript.json"
        
        enhanced_data = {
            "video_path": video_path,
            "total_segments": len(enhanced_segments),
            "segments": enhanced_segments
        }
        
        save_json(enhanced_data, str(json_path))
        logger.info(f"Saved enhanced transcript to {json_path}")
        return json_path
    
    def save_speaker_statistics_to_json(self, video_path: str, statistics: Dict) -> Path:
        """
        Save speaker statistics to JSON file.
        
        Args:
            video_path: Path to original video file (for output directory)
            statistics: Statistics dictionary from get_speaker_statistics()
            
        Returns:
            Path to saved JSON file
        """
        from src.utils.file_utils import get_video_metadata_dir
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "speaker_statistics.json"
        
        save_json(statistics, str(json_path))
        logger.info(f"Saved speaker statistics to {json_path}")
        return json_path
