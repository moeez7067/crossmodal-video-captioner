"""
Audio-only baseline model for video captioning.
Uses only audio transcription (Whisper) without visual information.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
from src.comparison.base_model import BaseCaptionModel
from src.audio.speech_to_text import SpeechToText
from src.generation.caption_generator import CaptionGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioOnlyBaseline(BaseCaptionModel):
    """
    Audio-only baseline that uses Whisper transcription directly.
    No visual information is used.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize audio-only baseline.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        super().__init__(config_dict)
        self.speech_to_text = SpeechToText()
        self.caption_generator = CaptionGenerator()
    
    def generate_captions(self, video_path: str) -> List[Dict]:
        """
        Generate captions using only audio transcription.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of caption dictionaries with 'text', 'start_time', 'end_time'
        """
        logger.info(f"Generating captions (audio-only) for {video_path}")
        
        # Extract audio (if needed)
        from src.preprocessing.audio_extractor import AudioExtractor
        audio_extractor = AudioExtractor()
        audio_path = audio_extractor.extract_audio(video_path)
        
        # Transcribe audio
        transcription = self.speech_to_text.transcribe(audio_path)
        segments = transcription.get("segments", [])
        
        # Generate captions from transcript segments
        captions = self.caption_generator.generate_captions_from_transcript(
            transcript_segments=segments,
            refine=True
        )
        
        logger.info(f"Generated {len(captions)} captions (audio-only)")
        
        return captions

