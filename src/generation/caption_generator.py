"""
Caption generation module for creating time-synchronized captions from transcripts.
Uses transcript segments directly with text cleaning and formatting.
"""

import re
from typing import List, Dict, Optional
import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CaptionGenerator:
    """Generates captions from transcript segments with text cleaning and formatting."""
    
    def __init__(self):
        """
        Initialize caption generator.
        No model loading needed - uses transcript segments directly.
        """
        self.model_name = "transcript-based"  # For metadata/logging
    
    def generate_captions_from_transcript(self,
                                         transcript_segments: List[Dict],
                                         max_length: Optional[int] = None,
                                         refine: bool = True) -> List[Dict]:
        """
        Generate captions from transcription segments.
        Uses transcript text directly with cleaning and formatting.
        
        Args:
            transcript_segments: List of transcript segments with 'text', 'start', 'end'
            max_length: Maximum caption length (default: from config)
            refine: Whether to refine captions after generation
            
        Returns:
            List of dictionaries with caption text and timestamps
        """
        if not transcript_segments:
            return []
        
        if max_length is None:
            max_length = config.CAPTION_MAX_LENGTH
        
        logger.info(f"Generating captions from {len(transcript_segments)} transcript segments")
        
        captions = []
        for segment in transcript_segments:
            try:
                text = segment.get("text", "").strip()
                if not text:
                    continue
                
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)
                
                # Clean and format the transcript text
                caption_text = self._clean_caption_text(text, max_length)
                
                if caption_text:
                    captions.append({
                        "text": caption_text,
                        "start_time": start_time,
                        "end_time": end_time
                    })
            except Exception as e:
                logger.warning(f"Error processing caption segment: {e}")
                # Fallback to original text with basic cleaning
                text = segment.get("text", "").strip()
                if text:
                    captions.append({
                        "text": self._basic_clean(text),
                        "start_time": segment.get("start", 0.0),
                        "end_time": segment.get("end", 0.0)
                    })
        
        # Refine captions if requested
        if refine:
            captions = self.refine_captions(captions)
        
        logger.info(f"Generated {len(captions)} captions from transcript")
        return captions
    
    def _clean_caption_text(self, text: str, max_length: int) -> str:
        """
        Clean and format caption text from transcript.
        
        Args:
            text: Raw transcript text
            max_length: Maximum caption length
            
        Returns:
            Cleaned and formatted caption text
        """
        # Basic cleaning
        text = self._basic_clean(text)
        
        # Truncate if too long (preserve word boundaries)
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0]
            # Ensure proper ending
            if text and text[-1] not in '.!?':
                text += '.'
        
        return text
    
    def _basic_clean(self, text: str) -> str:
        """
        Basic text cleaning: remove extra whitespace, normalize.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove leading/trailing punctuation artifacts
        text = text.strip('.,;:!?')
        
        # Ensure proper capitalization (sentence case)
        if text:
            # Only capitalize if it's all lowercase or all uppercase
            if text.islower() or text.isupper():
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text
    
    def generate_single_caption(self,
                               text: str,
                               start_time: float,
                               end_time: float,
                               max_length: Optional[int] = None) -> Dict:
        """
        Generate a single caption for a given text segment.
        
        Args:
            text: Input text to create caption from
            start_time: Start timestamp
            end_time: End timestamp
            max_length: Maximum caption length
            
        Returns:
            Dictionary with caption text and timestamps
        """
        if max_length is None:
            max_length = config.CAPTION_MAX_LENGTH
        
        caption_text = self._clean_caption_text(text, max_length)
        
        return {
            "text": caption_text,
            "start_time": start_time,
            "end_time": end_time
        }
    
    def refine_captions(self, captions: List[Dict]) -> List[Dict]:
        """
        Refine and post-process captions.
        
        Args:
            captions: List of captions
            
        Returns:
            Refined captions with improved quality
        """
        if not captions:
            return []
        
        logger.debug(f"Refining {len(captions)} captions")
        
        refined = []
        for caption in captions:
            text = caption.get("text", "").strip()
            if not text:
                continue
            
            # Apply refinements
            text = self._fix_punctuation(text)
            text = self._normalize_text(text)
            
            # Validate length
            if len(text) < config.CAPTION_MIN_LENGTH:
                # Skip very short captions (will be merged later)
                continue
            
            if len(text) > config.CAPTION_MAX_LENGTH:
                # Truncate if too long
                text = text[:config.CAPTION_MAX_LENGTH].rsplit(' ', 1)[0]
                if text and text[-1] not in '.!?':
                    text += '.'
            
            caption["text"] = text
            refined.append(caption)
        
        # Merge very short captions
        refined = self._merge_short_captions(refined)
        
        logger.debug(f"Refined to {len(refined)} captions")
        return refined
    
    def _fix_punctuation(self, text: str) -> str:
        """Ensure proper punctuation."""
        text = text.strip()
        if not text:
            return text
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[.!?]{2,}', '.', text)
        
        # Remove punctuation artifacts
        text = re.sub(r'[,;:]{2,}', ',', text)
        
        # Ensure ending punctuation (but don't force if it's mid-sentence)
        # Only add if there's no punctuation at all
        if text and text[-1] not in '.!?,;:':
            # Check if it looks like a complete sentence
            if len(text) > 10:  # Only for longer captions
                text += '.'
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text (remove extra spaces, fix capitalization)."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix capitalization (sentence case) - preserve existing mixed case
        if text:
            # Only fix if it's all lowercase or all uppercase
            if text.islower() and len(text) > 1:
                text = text[0].upper() + text[1:]
            elif text.isupper() and len(text) > 1:
                # Convert all caps to sentence case
                text = text[0] + text[1:].lower()
        
        return text
    
    def _merge_short_captions(self, captions: List[Dict], min_length: int = 10) -> List[Dict]:
        """
        Merge very short captions with adjacent ones.
        
        Args:
            captions: List of captions
            min_length: Minimum character length before merging
            
        Returns:
            Merged captions list
        """
        if not captions:
            return []
        
        merged = []
        i = 0
        
        while i < len(captions):
            current = captions[i].copy()
            current_text = current.get("text", "").strip()
            
            # If current caption is too short, try to merge with next
            if len(current_text) < min_length and i + 1 < len(captions):
                next_caption = captions[i + 1]
                next_text = next_caption.get("text", "").strip()
                merged_text = current_text + " " + next_text
                
                # Only merge if combined length is reasonable
                if len(merged_text) <= config.CAPTION_MAX_LENGTH:
                    current["text"] = merged_text.strip()
                    current["end_time"] = next_caption.get("end_time", current["end_time"])
                    i += 2  # Skip next caption as it's merged
                else:
                    i += 1
            else:
                i += 1
            
            merged.append(current)
        
        return merged
    
    def generate_captions(self,
                         multimodal_embeddings: Optional[any] = None,
                         transcript_segments: Optional[List[Dict]] = None,
                         timestamps: Optional[List[float]] = None,
                         max_length: Optional[int] = None) -> List[Dict]:
        """
        Main entry point for caption generation.
        Supports both multimodal embeddings (Phase 3) and transcript-based (current).
        
        Args:
            multimodal_embeddings: Fused audio-visual embeddings (from Phase 3)
                                   Can be numpy array or None
            transcript_segments: Transcription segments (current approach)
            timestamps: List of timestamps (if using embeddings)
            max_length: Maximum caption length
            
        Returns:
            List of dictionaries with caption text and timestamps
        """
        # If multimodal embeddings available, use them (Phase 3)
        if multimodal_embeddings is not None:
            logger.info("Fused embeddings available, using enhanced caption generation")
            # For MVP: Use embeddings to enhance transcript-based generation
            # Future: Implement full embedding-to-text conversion
            return self.generate_captions_from_embeddings(
                multimodal_embeddings,
                transcript_segments,
                timestamps,
                max_length
            )
        
        # Otherwise, use transcript-based generation (current)
        if transcript_segments:
            return self.generate_captions_from_transcript(transcript_segments, max_length)
        
        raise ValueError("Either multimodal_embeddings or transcript_segments must be provided")
    
    def generate_captions_from_embeddings(
        self,
        fused_embeddings,
        transcript_segments: Optional[List[Dict]] = None,
        timestamps: Optional[List[float]] = None,
        max_length: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate captions from fused multimodal embeddings (MVP implementation).
        
        Strategy: Use embeddings to enhance transcript segments.
        For now, we use the transcript-based approach but with embeddings available
        for future enhancement. This maintains backward compatibility.
        
        Args:
            fused_embeddings: Fused multimodal embeddings (numpy array)
            transcript_segments: Transcription segments (required for MVP)
            timestamps: Optional timestamps
            max_length: Maximum caption length
            
        Returns:
            List of dictionaries with caption text and timestamps
        """
        import numpy as np
        
        # Validate embeddings
        if isinstance(fused_embeddings, np.ndarray):
            logger.debug(f"Processing fused embeddings: shape {fused_embeddings.shape}")
        else:
            logger.warning(f"Unexpected embedding type: {type(fused_embeddings)}, falling back to transcript")
            if transcript_segments:
                return self.generate_captions_from_transcript(transcript_segments, max_length)
            return []
        
        # For MVP: Use transcript-based generation
        # The embeddings are available for future use (e.g., weighting segments, filtering)
        # This maintains backward compatibility while enabling future enhancements
        if transcript_segments:
            logger.info("Using transcript-based generation with embeddings available for enhancement")
            # Generate captions from transcript (embeddings can be used for filtering/weighting in future)
            captions = self.generate_captions_from_transcript(transcript_segments, max_length, refine=True)
            
            # Future enhancement: Use embeddings to:
            # - Filter out low-quality segments
            # - Weight segment importance
            # - Enhance caption text based on visual context
            
            return captions
        else:
            logger.warning("No transcript segments provided, cannot generate captions")
            return []
    
    def save_captions_to_json(self, video_path: str, captions: List[Dict]) -> str:
        """
        Save generated captions to JSON file.
        
        Args:
            video_path: Path to original video file
            captions: List of generated captions
            
        Returns:
            Path to saved JSON file
        """
        from src.utils.file_utils import get_video_metadata_dir, save_json
        from pathlib import Path
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "generated_captions.json"
        
        captions_data = {
            "video_path": video_path,
            "model_name": self.model_name,
            "total_captions": len(captions),
            "captions": captions
        }
        
        save_json(captions_data, str(json_path))
        logger.info(f"Saved generated captions to {json_path}")
        return str(json_path)
