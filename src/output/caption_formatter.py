"""
Caption formatting module for generating .srt and .vtt files.
"""

from typing import List, Dict
from pathlib import Path
from src.utils.time_utils import seconds_to_srt_timestamp, seconds_to_vtt_timestamp


class CaptionFormatter:
    """Formats captions into .srt and .vtt subtitle formats."""
    
    def __init__(self):
        """Initialize caption formatter."""
        pass
    
    def format_srt(self, captions: List[Dict]) -> str:
        """
        Format captions to SRT subtitle format string.
        
        Args:
            captions: List of caption dictionaries with 'text', 'start_time', 'end_time'
            
        Returns:
            SRT formatted string
        """
        srt_lines = []
        for idx, caption in enumerate(captions, 1):
            start_time = caption.get("start_time", 0.0)
            end_time = caption.get("end_time", 0.0)
            text = caption.get("text", "").strip()
            
            if not text:
                continue
            
            start_str = seconds_to_srt_timestamp(start_time)
            end_str = seconds_to_srt_timestamp(end_time)
            
            srt_lines.append(f"{idx}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between captions
        
        return "\n".join(srt_lines)
    
    def format_vtt(self, captions: List[Dict]) -> str:
        """
        Format captions to VTT subtitle format string.
        
        Args:
            captions: List of caption dictionaries with 'text', 'start_time', 'end_time'
            
        Returns:
            VTT formatted string
        """
        vtt_lines = ["WEBVTT", ""]  # VTT header
        
        for caption in captions:
            start_time = caption.get("start_time", 0.0)
            end_time = caption.get("end_time", 0.0)
            text = caption.get("text", "").strip()
            
            if not text:
                continue
            
            start_str = seconds_to_vtt_timestamp(start_time)
            end_str = seconds_to_vtt_timestamp(end_time)
            
            vtt_lines.append(f"{start_str} --> {end_str}")
            vtt_lines.append(text)
            vtt_lines.append("")  # Empty line between captions
        
        return "\n".join(vtt_lines)
    
    def format_to_srt(self, captions: List[Dict], output_path: str) -> str:
        """
        Format captions to SRT file.
        
        Args:
            captions: List of caption dictionaries
            output_path: Path to save SRT file
            
        Returns:
            Path to saved SRT file
        """
        srt_content = self.format_srt(captions)
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_path_obj.write_text(srt_content, encoding='utf-8')
        return str(output_path_obj)
    
    def format_to_vtt(self, captions: List[Dict], output_path: str) -> str:
        """
        Format captions to VTT file.
        
        Args:
            captions: List of caption dictionaries
            output_path: Path to save VTT file
            
        Returns:
            Path to saved VTT file
        """
        vtt_content = self.format_vtt(captions)
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_path_obj.write_text(vtt_content, encoding='utf-8')
        return str(output_path_obj)
    
    def merge_captions(self, captions: List[Dict], max_duration: float = 5.0) -> List[Dict]:
        """
        Merge short captions to avoid flickering.
        
        Args:
            captions: List of caption dictionaries
            max_duration: Maximum duration for a single caption
            
        Returns:
            Merged captions list
        """
        if not captions:
            return []
        
        merged = []
        current = captions[0].copy()
        
        for next_caption in captions[1:]:
            duration = current.get("end_time", 0.0) - current.get("start_time", 0.0)
            
            if duration < max_duration and next_caption.get("start_time", 0.0) - current.get("end_time", 0.0) < 0.5:
                # Merge with next caption
                current["end_time"] = next_caption.get("end_time", 0.0)
                current["text"] = current.get("text", "") + " " + next_caption.get("text", "")
            else:
                # Save current and start new
                merged.append(current)
                current = next_caption.copy()
        
        merged.append(current)
        return merged

