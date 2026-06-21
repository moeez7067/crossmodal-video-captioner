"""
Processing service that orchestrates video processing pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.audio_extractor import AudioExtractor
from src.audio.speech_to_text import SpeechToText
from src.audio.speaker_diarization import SpeakerDiarization
from src.visual.frame_extractor import FrameExtractor
from src.visual.visual_embeddings import VisualEmbeddings
from src.fusion.fusion_service import FusionService
from src.generation.caption_generator import CaptionGenerator
from src.generation.summarizer import Summarizer
from src.output.caption_formatter import CaptionFormatter
from src.output.transcript_formatter import TranscriptFormatter
from src.utils.logger import get_logger
from src.api.services.job_manager import job_manager, JobStatus
import config

logger = get_logger(__name__)


class ProcessingService:
    """Orchestrates the complete video processing pipeline."""
    
    def __init__(self):
        """Initialize processing service with all modules."""
        self.video_processor = VideoProcessor()
        self.audio_extractor = AudioExtractor()
        self.speech_to_text = SpeechToText()
        self.speaker_diarization = SpeakerDiarization()
        self.frame_extractor = FrameExtractor()
        self.visual_embeddings = VisualEmbeddings()
        self.fusion_service = FusionService()
        self.caption_generator = CaptionGenerator()
        self.summarizer = Summarizer()
        self.caption_formatter = CaptionFormatter()
        self.transcript_formatter = TranscriptFormatter()
        self.job_manager = job_manager
    
    def process_video(self, job_id: str, video_path: str) -> Dict:
        """
        Process video through complete pipeline.
        
        Args:
            job_id: Job identifier
            video_path: Path to video file
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Update status: Processing started
            self.job_manager.update_status(job_id, JobStatus.PROCESSING, progress=5.0, stage="Initializing")
            
            logger.info(f"Starting processing for job {job_id}: {video_path}")

            # A video with no audio track cannot be transcribed. Detect this early
            # and fail with a clear message instead of a cryptic FFmpeg error later.
            if not self.audio_extractor.has_audio_stream(video_path):
                raise RuntimeError(
                    "This video has no audio track, so there is no speech to transcribe. "
                    "Please upload a video that contains spoken audio."
                )

            # Steps 1 & 2: Video Preprocessing and Audio Extraction (Parallel)
            # Both only need video_path, so they can run simultaneously
            self.job_manager.update_status(job_id, JobStatus.PROCESSING, progress=10.0, stage="Extracting frames and audio (parallel)")
            logger.info(f"Starting parallel extraction: frames and audio for job {job_id}")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks in parallel
                video_future = executor.submit(self._process_video_preprocessing, video_path)
                audio_future = executor.submit(self._process_audio_extraction, video_path)
                
                # Wait for both to complete and get results
                try:
                    frames_data, video_metadata = video_future.result()
                    audio_path, audio_metadata, chunks = audio_future.result()
                    logger.info(f"Parallel extraction completed for job {job_id}")
                except Exception as e:
                    logger.error(f"Error in parallel extraction for job {job_id}: {e}")
                    raise RuntimeError(f"Failed during parallel extraction: {e}") from e
            
            # Update progress after both complete
            self.job_manager.update_status(job_id, JobStatus.PROCESSING, progress=20.0, stage="Preprocessing complete")
            
            # Step 3: Speech-to-Text + Audio Embeddings (40-45%)
            # Extract embeddings during transcription to avoid duplicate encoding
            extract_embeddings = config.AUDIO_EMBEDDING_EXTRACT and config.FUSION_ENABLED
            self.job_manager.update_status(
                job_id, 
                JobStatus.PROCESSING, 
                progress=40.0, 
                stage="Transcribing audio" + (" and extracting embeddings" if extract_embeddings else "")
            )
            
            transcription = self.speech_to_text.transcribe(
                audio_path,
                extract_embeddings=extract_embeddings
            )
            self.speech_to_text.save_transcription_to_json(video_path, transcription)
            
            # Save audio embeddings if extracted during transcription
            if extract_embeddings and "embeddings" in transcription:
                self.job_manager.update_status(job_id, JobStatus.PROCESSING, progress=45.0, stage="Saving audio embeddings")
                try:
                    audio_embeddings = transcription["embeddings"]
                    audio_emb_timestamps = transcription.get("embedding_timestamps")
                    self.speech_to_text.save_audio_embeddings(video_path, audio_embeddings, audio_emb_timestamps)
                    logger.info(f"Saved audio embeddings extracted during transcription: shape {audio_embeddings.shape}")
                except Exception as e:
                    logger.warning(f"Failed to save audio embeddings: {e}, continuing without embeddings")
            
            # Step 4: Speaker Diarization (55%)
            self.job_manager.update_status(job_id, JobStatus.PROCESSING, progress=55.0, stage="Identifying speakers")
            try:
                diarization_results = self.speaker_diarization.diarize(audio_path)
            except Exception as e:
                # Diarization is optional: it requires pyannote.audio plus a Hugging Face
                # token with the gated model license accepted. If unavailable, continue
                # without speaker attribution instead of failing the whole job.
                logger.warning(
                    f"Speaker diarization unavailable ({e}); continuing without speaker attribution"
                )
                diarization_results = []
            enhanced_segments = self.speaker_diarization.assign_speakers_to_segments(
                transcription["segments"], diarization_results
            )
            speaker_stats = self.speaker_diarization.get_speaker_statistics(diarization_results)
            
            # Save to JSON
            self.speaker_diarization.save_diarization_to_json(video_path, diarization_results)
            self.speaker_diarization.save_enhanced_transcript_to_json(video_path, enhanced_segments)
            self.speaker_diarization.save_speaker_statistics_to_json(video_path, speaker_stats)
            
            # Step 5: Visual Processing (75%)
            self.job_manager.update_status(job_id, JobStatus.PROCESSING, progress=75.0, stage="Processing visual content")
            frames_with_timestamps = self.frame_extractor.extract_frames(video_path, return_arrays=True)
            frames = [f[0] for f in frames_with_timestamps]
            timestamps = [f[1] for f in frames_with_timestamps]
            
            # Detect slides
            slide_changes = self.frame_extractor.detect_slides(frames, timestamps)
            self.frame_extractor.save_slide_detection_to_json(video_path, slide_changes)
            
            # Extract visual embeddings
            embeddings = self.visual_embeddings.extract_embeddings(frames)
            # Save embeddings to .npy file
            self.visual_embeddings.save_embeddings(video_path, embeddings)
            # Save embeddings metadata to JSON
            self.visual_embeddings.save_embeddings_info_to_json(video_path, embeddings, timestamps)
            
            # Step 5.5: Multimodal Fusion (77%)
            fused_embeddings = None
            if config.FUSION_ENABLED:
                self.job_manager.update_status(job_id, JobStatus.PROCESSING, progress=77.0, stage="Fusing audio and visual embeddings")
                try:
                    # Prepare timestamps for fusion
                    audio_timestamps = [seg.get("start", 0.0) for seg in transcription.get("segments", [])]
                    visual_timestamps = timestamps
                    
                    # Perform fusion
                    fusion_result = self.fusion_service.fuse_from_phase2_outputs(
                        video_path,
                        audio_timestamps=audio_timestamps if audio_timestamps else None,
                        visual_timestamps=visual_timestamps if visual_timestamps else None
                    )
                    
                    if fusion_result:
                        fused_embeddings = fusion_result['fused_embeddings']
                        logger.info(f"Multimodal fusion completed successfully - Output shape: {fused_embeddings.shape}")
                    else:
                        logger.warning("Fusion failed, continuing with text-based generation")
                        fused_embeddings = None
                        
                except Exception as e:
                    logger.warning(f"Fusion error: {e}, continuing with text-based generation", exc_info=True)
                    fused_embeddings = None
            else:
                logger.debug("Fusion is disabled in configuration")
            
            # Step 6: Generate Captions (80%)
            self.job_manager.update_status(job_id, JobStatus.PROCESSING, progress=80.0, stage="Generating captions")
            
            # Generate captions - use fused embeddings if available, otherwise use transcript
            try:
                # Prepare timestamps for caption generation
                caption_timestamps = [seg.get("start", 0.0) for seg in transcription.get("segments", [])]
                
                # Try to use fused embeddings if available
                if fused_embeddings is not None:
                    logger.info("Using fused embeddings for caption generation")
                    # For now, still use transcript-based generation but with embeddings available
                    # Future: implement embedding-based generation
                    captions = self.caption_generator.generate_captions(
                        multimodal_embeddings=fused_embeddings,
                        transcript_segments=transcription.get("segments", []),
                        timestamps=caption_timestamps
                    )
                else:
                    # Fall back to transcript-based generation
                    captions = self.caption_generator.generate_captions_from_transcript(
                        transcription.get("segments", []),
                        refine=True
                    )
                
                # Save generated captions to JSON
                self.caption_generator.save_captions_to_json(video_path, captions)
                logger.info(f"Generated {len(captions)} captions")
            except Exception as e:
                logger.warning(f"Caption generation failed, using fallback: {e}")
                # Fallback to simple mapping with basic cleaning
                captions = []
                for seg in transcription.get("segments", []):
                    text = seg.get("text", "").strip()
                    if text:
                        captions.append({
                            "text": text,
                            "start_time": seg.get("start", 0.0),
                            "end_time": seg.get("end", 0.0)
                        })
                logger.info(f"Used fallback: generated {len(captions)} captions")
            
            # Step 7: Generate Summary (87%)
            self.job_manager.update_status(job_id, JobStatus.PROCESSING, progress=87.0, stage="Generating summary")
            
            # Extract visual context
            visual_context = self._extract_visual_context(slide_changes, {
                "frames_count": len(frames),
                "slide_transitions": len(slide_changes)
            })
            
            # Generate summary using T5
            key_points = []  # Initialize key_points
            try:
                # Get full transcript text
                full_transcript = transcription.get("text", "")
                
                # Generate summary with visual context
                summary_text = self.summarizer.generate_summary(
                    full_transcript,
                    visual_context=visual_context,
                    max_length=config.SUMMARY_MAX_LENGTH,
                    min_length=config.SUMMARY_MIN_LENGTH
                )
                
                # Extract key points separately
                try:
                    key_points = self.summarizer.extract_key_points(
                        full_transcript,
                        num_points=config.SUMMARY_NUM_KEY_POINTS
                    )
                except Exception as e:
                    logger.warning(f"Key points extraction failed: {e}")
                    key_points = []
                
                # Save generated summary to JSON
                self.summarizer.save_summary_to_json(video_path, summary_text, key_points)
                logger.info(f"Generated summary using T5 (length: {len(summary_text)} chars, {len(key_points)} key points)")
            except Exception as e:
                logger.warning(f"Summary generation failed, using fallback: {e}")
                # Fallback to simple summary
                summary_text = self._create_simple_summary(transcription.get("text", ""), enhanced_segments)
                key_points = []
            
            # Step 8: Format Outputs (95%)
            self.job_manager.update_status(job_id, JobStatus.PROCESSING, progress=95.0, stage="Formatting outputs")
            
            # Format captions as SRT and VTT
            captions_srt = self.caption_formatter.format_srt(captions)
            captions_vtt = self.caption_formatter.format_vtt(captions)
            
            # Format transcript with speaker attribution
            transcript_text = self.transcript_formatter.format_with_speakers(enhanced_segments)
            
            # Save formatted outputs to outputs folder
            from src.utils.file_utils import get_video_outputs_dir
            
            outputs_dir = get_video_outputs_dir(video_path)
            
            # Save caption files
            (outputs_dir / "captions.srt").write_text(captions_srt, encoding='utf-8')
            (outputs_dir / "captions.vtt").write_text(captions_vtt, encoding='utf-8')
            
            # Save transcript
            (outputs_dir / "transcript.txt").write_text(transcript_text, encoding='utf-8')
            
            # Save summary
            (outputs_dir / "summary.txt").write_text(summary_text, encoding='utf-8')
            
            logger.info(f"Saved formatted outputs to {outputs_dir}")
            
            # Step 9: Compile Results (100%)
            results = {
                "job_id": job_id,
                "captions": captions,
                "captions_text": captions_srt,
                "transcript_text": transcript_text,
                "summary_text": summary_text,
                "key_points": key_points,
                "video_url": f"/api/video/stream/{job_id}",  # URL to stream video
                "metadata": {
                    "video": video_metadata,
                    "audio": audio_metadata,
                    "speakers": speaker_stats,
                    "frames_count": len(frames),
                    "slide_transitions": len(slide_changes),
                    "generation_models": {
                        "caption_model": "transcript-based",
                        "summary_model": self.summarizer.model_name
                    },
                    "fusion_enabled": config.FUSION_ENABLED,
                    "fusion_successful": fused_embeddings is not None if 'fused_embeddings' in locals() else False,
                    "enhanced_segments": enhanced_segments  # Include for DOCX export
                }
            }
            
            # Save results
            self.job_manager.set_results(job_id, results)
            logger.info(f"Processing completed for job {job_id}")
            
            return results
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"Error processing job {job_id}: {error_msg}", exc_info=True)
            self.job_manager.update_status(job_id, JobStatus.FAILED, error=error_msg)
            raise
    
    def _process_video_preprocessing(self, video_path: str) -> Tuple[List, Dict]:
        """
        Process video preprocessing (extract frames and metadata).
        This method is designed to run in parallel with audio extraction.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames_data, video_metadata)
        """
        logger.debug(f"Starting video preprocessing: {video_path}")
        frames_data = self.video_processor.extract_frames(video_path)
        video_metadata = self.video_processor.get_video_metadata(video_path)
        
        # Save metadata to JSON
        self.video_processor.save_metadata_to_json(video_path, video_metadata)
        self.video_processor.save_frames_info_to_json(video_path, frames_data)
        
        logger.debug(f"Video preprocessing completed: {video_path}")
        return frames_data, video_metadata
    
    def _process_audio_extraction(self, video_path: str) -> Tuple[str, Dict, List]:
        """
        Process audio extraction (extract audio, metadata, and chunks).
        This method is designed to run in parallel with video preprocessing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (audio_path, audio_metadata, chunks)
        """
        logger.debug(f"Starting audio extraction: {video_path}")
        audio_path = self.audio_extractor.extract_audio(video_path)
        audio_metadata = self.audio_extractor.get_audio_metadata(audio_path)
        chunks = self.audio_extractor.chunk_audio(audio_path, video_path=video_path)
        
        # Save to JSON
        self.audio_extractor.save_metadata_to_json(video_path, audio_path, audio_metadata)
        self.audio_extractor.save_chunks_info_to_json(video_path, chunks)
        
        logger.debug(f"Audio extraction completed: {audio_path}")
        return audio_path, audio_metadata, chunks
    
    def _extract_visual_context(self,
                               slide_changes: Optional[List[Dict]] = None,
                               visual_metadata: Optional[Dict] = None) -> str:
        """
        Extract visual context description for summary generation.
        
        Args:
            slide_changes: List of slide transition detections
            visual_metadata: Visual processing metadata
            
        Returns:
            Visual context description string
        """
        context_parts = []
        
        if slide_changes:
            context_parts.append(f"Presentation with {len(slide_changes)} slide transitions")
        
        if visual_metadata:
            frames_count = visual_metadata.get("frames_count", 0)
            if frames_count:
                context_parts.append(f"{frames_count} frames processed")
        
        return ". ".join(context_parts) if context_parts else ""
    
    def _create_simple_summary(self, full_text: str, segments: List[Dict]) -> str:
        """
        Create a simple summary from transcript (fallback method).
        Used when T5 summarization fails.
        
        Args:
            full_text: Full transcript text
            segments: Enhanced transcript segments with speakers
            
        Returns:
            Summary text
        """
        # Simple summary: first few sentences + key points
        sentences = full_text.split('.')
        if len(sentences) > 3:
            summary = '. '.join(sentences[:3]) + '.'
        else:
            summary = full_text
        
        # Add speaker summary if available
        speakers = set(seg.get("speaker_id") for seg in segments if seg.get("speaker_id"))
        if speakers:
            summary += f"\n\nSpeakers: {len(speakers)} participant(s)"
        
        return summary


# Exported helper for RQ workers
def process_video_job(job_id: str, video_path: str) -> Dict:
    """
    Importable wrapper for RQ workers to call.

    Example enqueue path: "src.api.services.processing_service.process_video_job"
    """
    service = ProcessingService()
    return service.process_video(job_id, video_path)

