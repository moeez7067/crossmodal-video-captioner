"""
Abstractive summarization module for generating concise summaries using T5.
"""

import torch
import re
from typing import List, Dict, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer
import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Summarizer:
    """Generates abstractive summaries from transcripts and visual context using T5."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize summarizer.
        
        Args:
            model_name: T5 model name for summarization
                       If None, uses config.T5_MODEL
            device: Device to run model on (cuda/cpu, None for auto-detection)
        """
        self.model_name = model_name if model_name is not None else config.T5_MODEL
        self.device = device if device is not None else config.T5_DEVICE
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    def load_model(self):
        """Load the T5 model and tokenizer for summarization."""
        if self._is_loaded and self.model is not None:
            logger.debug("T5 model already loaded")
            return
        
        logger.info(f"Loading T5 model for summarization: {self.model_name} on device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self._is_loaded = True
            logger.info(f"Successfully loaded T5 model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load T5 model: {e}")
            raise RuntimeError(f"Could not load T5 model {self.model_name}: {e}") from e
    
    def generate_summary(self,
                        transcript: str,
                        visual_context: Optional[str] = None,
                        max_length: Optional[int] = None,
                        min_length: Optional[int] = None) -> str:
        """
        Generate abstractive summary from transcript.
        
        Args:
            transcript: Full transcript text
            visual_context: Optional visual context description
            max_length: Maximum summary length (default: from config)
            min_length: Minimum summary length (default: from config)
            
        Returns:
            Generated summary text
        """
        if not self._is_loaded:
            self.load_model()
        
        if not transcript or not transcript.strip():
            return ""
        
        # Calculate adaptive summary length based on transcript length
        transcript_length = len(transcript)
        estimated_words = transcript_length / 5  # Rough estimate: 5 chars per word
        
        if max_length is None:
            # Adaptive max_length: ~1 token per 20-30 words of transcript
            # Minimum 100, maximum 1024 tokens
            # For a 10-minute video (~1500 words), we want ~50-75 tokens summary
            # For a 30-minute video (~4500 words), we want ~150-225 tokens summary
            adaptive_max = max(100, min(1024, int(estimated_words / 20)))
            max_length = adaptive_max
            logger.info(f"Adaptive summary max_length: {transcript_length} chars (~{int(estimated_words)} words) -> {max_length} tokens")
        else:
            max_length = max_length
        
        if min_length is None:
            # Adaptive min_length: ~30% of max_length, minimum 30
            adaptive_min = max(30, int(max_length * 0.3))
            min_length = adaptive_min
            logger.info(f"Adaptive summary min_length: {min_length} tokens")
        else:
            min_length = min_length
        
        logger.info(f"Generating summary from transcript (length: {len(transcript)} chars)")
        
        # Combine transcript and visual context
        input_text = transcript
        if visual_context:
            input_text = f"{transcript}\n\nVisual context: {visual_context}"
        
        # Handle long transcripts (T5 has ~512 token limit)
        if len(input_text) > 2000:  # Rough estimate for token count
            logger.info("Transcript too long, using chunking strategy")
            return self._generate_summary_chunked(input_text, max_length, min_length)
        
        try:
            # Create summarization prompt
            prompt = f"summarize: {input_text}"
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    temperature=1.0
                )
            
            # Decode to text
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean and post-process
            summary = self._clean_summary(summary)
            
            logger.info(f"Generated summary (length: {len(summary)} chars)")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to extractive summary
            return self._extractive_fallback(transcript, max_length)
    
    def _generate_summary_chunked(self, text: str, max_length: int, min_length: int) -> str:
        """
        Generate summary for long text by chunking.
        
        Args:
            text: Long input text
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Combined summary
        """
        # Split text into chunks (preserve sentences)
        chunks = self._chunk_text(text, chunk_size=1500)
        
        logger.info(f"Processing {len(chunks)} chunks")
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                prompt = f"summarize: {chunk}"
                inputs = self.tokenizer(
                    prompt,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_length=max_length // len(chunks),
                        min_length=min_length // len(chunks),
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False
                    )
                
                chunk_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                chunk_summaries.append(chunk_summary)
                
            except Exception as e:
                logger.warning(f"Error summarizing chunk {i}: {e}")
                continue
        
        # Combine chunk summaries
        combined_text = " ".join(chunk_summaries)
        
        # If combined summary is still long, summarize it again
        if len(combined_text) > max_length:
            return self.generate_summary(combined_text, max_length=max_length, min_length=min_length)
        
        return self._clean_summary(combined_text)
    
    def _chunk_text(self, text: str, chunk_size: int = 1500) -> List[str]:
        """
        Split text into chunks preserving sentence boundaries.
        
        Args:
            text: Input text
            chunk_size: Approximate chunk size in characters
            
        Returns:
            List of text chunks
        """
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _clean_summary(self, text: str) -> str:
        """Clean and normalize summary text with improved grammar and flow."""
        # Remove backslashes and escape sequences
        text = text.replace('\\', ' ').replace('\n', ' ')
        
        # Remove extra whitespace but preserve sentence boundaries
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences and format properly
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Format sentences with proper grammar
        formatted_sentences = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
                
            # Remove multiple punctuation
            sentence = re.sub(r'[.!?]+', '.', sentence)
            
            # Remove leading lowercase letters that shouldn't be there
            sentence = sentence.strip()
            
            # Ensure proper capitalization
            if sentence:
                # Capitalize first letter (handle special cases)
                if sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                elif not sentence[0].isupper() and sentence[0].isalpha():
                    sentence = sentence[0].upper() + sentence[1:]
            
            # Fix common grammar issues
            sentence = self._fix_grammar_issues(sentence)
            
            # Ensure ending punctuation
            if sentence and sentence[-1] not in '.!?':
                sentence += '.'
            
            # Add transition words for better flow (except first sentence)
            if i > 0 and formatted_sentences:
                # Check if sentence starts with lowercase (might need connection)
                if sentence and sentence[0].islower():
                    # Don't add transition if it already starts with a conjunction
                    if not re.match(r'^(and|but|or|so|however|therefore|furthermore|additionally|also|moreover|in addition|for example|for instance)', sentence.lower()):
                        # Add appropriate transition based on context
                        pass  # Let the model handle transitions naturally
            
            if sentence:
                formatted_sentences.append(sentence)
        
        # Improve sentence flow and coherence
        formatted_sentences = self._improve_sentence_flow(formatted_sentences)
        
        # Join sentences with proper spacing for better flow
        text = ' '.join(formatted_sentences)
        
        # Final cleanup: ensure proper spacing around punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
        
        # Final grammar check
        text = self._final_grammar_pass(text)
        
        return text
    
    def _improve_sentence_flow(self, sentences: List[str]) -> List[str]:
        """Improve flow between sentences for better coherence."""
        if not sentences or len(sentences) <= 1:
            return sentences
        
        improved = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
            
            # Remove redundant starting words if previous sentence already has them
            if i > 0 and improved:
                prev_sentence = improved[-1].lower()
                current_lower = sentence.lower()
                
                # Avoid repetition of "this video", "the video", etc.
                if prev_sentence.endswith('video') and current_lower.startswith('this video'):
                    sentence = re.sub(r'^[Tt]his video\s+', '', sentence, count=1)
                    if sentence and sentence[0].islower():
                        sentence = sentence[0].upper() + sentence[1:]
            
            # Ensure sentence starts properly
            if sentence and sentence[0].islower() and i > 0:
                # Check if it should start with capital (not a continuation)
                if not re.match(r'^(and|but|or|so|however|therefore|furthermore|additionally|also|moreover|in addition|for example|for instance|specifically|particularly)', sentence.lower()):
                    sentence = sentence[0].upper() + sentence[1:]
            
            improved.append(sentence)
        
        return improved
    
    def _final_grammar_pass(self, text: str) -> str:
        """Final grammar and formatting pass."""
        if not text:
            return text
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])([A-Za-z])', r'\1 \2', text)
        
        # Ensure proper capitalization at start
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Fix "i" to "I" throughout
        text = re.sub(r'\bi\b', 'I', text)
        text = re.sub(r'\bi\'', 'I\'', text)
        
        # Remove redundant punctuation
        text = re.sub(r'([.!?])+', r'\1', text)
        
        # Ensure it ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text.strip()
    
    def _fix_grammar_issues(self, text: str) -> str:
        """Fix common grammar issues in text."""
        if not text:
            return text
        
        # Fix double spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])([A-Za-z])', r'\1 \2', text)
        
        # Fix common lowercase issues at start
        if text and text[0].islower() and len(text) > 1:
            # Only capitalize if it's not a proper noun or acronym
            if not re.match(r'^(i|i\'m|i\'ve|i\'ll|i\'d)', text.lower()):
                text = text[0].upper() + text[1:]
        
        # Fix "i" to "I"
        text = re.sub(r'\bi\b', 'I', text)
        text = re.sub(r'\bi\'', 'I\'', text)
        
        # Remove redundant punctuation
        text = re.sub(r'([.!?])+', r'\1', text)
        
        return text.strip()
    
    def _extractive_fallback(self, text: str, max_length: int) -> str:
        """
        Fallback to extractive summarization if T5 fails.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Extractive summary (first sentences)
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        summary_sentences = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            summary_sentences.append(sentence)
            current_length += len(sentence) + 1
        
        return " ".join(summary_sentences)
    
    def generate_hierarchical_summary(self,
                                     segments: List[Dict],
                                     max_length: Optional[int] = None) -> Dict:
        """
        Generate hierarchical summary (key points + full summary).
        
        Args:
            segments: List of transcript segments with timestamps
            max_length: Maximum summary length
            
        Returns:
            Dictionary with key_points and full_summary
        """
        if not segments:
            return {"key_points": [], "full_summary": ""}
        
        # Combine segments into full transcript
        transcript = " ".join(seg.get("text", "") for seg in segments)
        
        # Generate full summary
        full_summary = self.generate_summary(transcript, max_length=max_length)
        
        # Extract key points
        key_points = self.extract_key_points(transcript, num_points=config.SUMMARY_NUM_KEY_POINTS)
        
        return {
            "key_points": key_points,
            "full_summary": full_summary
        }
    
    def extract_key_points(self, transcript: str, num_points: Optional[int] = None) -> List[str]:
        """
        Extract key discussion points from transcript.
        Automatically determines number of key points based on transcript length.
        
        Args:
            transcript: Full transcript text
            num_points: Number of key points to extract (default: adaptive based on length)
            
        Returns:
            List of key points
        """
        if not self._is_loaded:
            self.load_model()
        
        if not transcript or not transcript.strip():
            return []
        
        # Calculate adaptive number of key points based on transcript length
        if num_points is None:
            # Base: 1 key point per ~500 words or ~2500 characters
            # Minimum 3, maximum 15
            transcript_length = len(transcript)
            estimated_words = transcript_length / 5  # Rough estimate: 5 chars per word
            num_points = max(3, min(15, int(estimated_words / 500) + 3))
            logger.info(f"Adaptive key points calculation: {transcript_length} chars -> {num_points} key points")
        else:
            num_points = num_points
        
        logger.info(f"Extracting {num_points} key points from transcript")
        
        try:
            # Create prompt for key point extraction
            prompt = f"list key points: {transcript}"
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Calculate adaptive max_length for key points generation
            # Each key point is roughly 50-100 tokens, so we need more tokens for more points
            # Base: 100 tokens per key point, minimum 300, maximum 800
            key_points_max_length = max(300, min(800, num_points * 100))
            
            # Generate key points
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=key_points_max_length,  # Adaptive length based on number of points
                    num_beams=4,
                    early_stopping=False,  # Don't stop early - we want all points
                    do_sample=False
                )
            
            # Decode to text
            key_points_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse key points (split by numbers, bullets, or newlines)
            key_points = self._parse_key_points(key_points_text, num_points)
            
            logger.info(f"Extracted {len(key_points)} key points")
            return key_points
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            # Fallback to extractive approach
            return self._extractive_key_points(transcript, num_points)
    
    def _parse_key_points(self, text: str, num_points: int) -> List[str]:
        """
        Parse key points from generated text.
        
        Args:
            text: Generated key points text
            num_points: Expected number of points
            
        Returns:
            List of key point strings
        """
        # Try to split by numbered list (1., 2., etc.)
        points = re.split(r'\d+[.)]\s+', text)
        points = [p.strip() for p in points if p.strip()]
        
        # If no numbered list, try bullet points
        if len(points) <= 1:
            points = re.split(r'[-•*]\s+', text)
            points = [p.strip() for p in points if p.strip()]
        
        # If still no split, try newlines
        if len(points) <= 1:
            points = [p.strip() for p in text.split('\n') if p.strip()]
        
        # If still one long string, try splitting by sentence endings followed by speaker names or common patterns
        if len(points) <= 1 and len(text) > 100:
            # Split by sentence endings that might indicate new points
            # Look for patterns like "speaker: " or sentence endings followed by capital letters
            points = re.split(r'(?<=[.!?])\s+(?=[A-Z][a-z]+:|[A-Z][a-z]+\s+[a-z]+:)', text)
            points = [p.strip() for p in points if p.strip()]
        
        # If still one point, try splitting by multiple sentence endings
        if len(points) <= 1:
            # Split by sentence endings, but only if we have multiple sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 2:
                # Group sentences into logical points (2-3 sentences per point)
                points = []
                current_point = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    current_point.append(sentence)
                    # If we have 2-3 sentences or the sentence is long, start a new point
                    if len(current_point) >= 2 or len(' '.join(current_point)) > 150:
                        points.append(' '.join(current_point))
                        current_point = []
                if current_point:
                    points.append(' '.join(current_point))
        
        # Clean and format points with proper grammar
        cleaned_points = []
        for point in points:
            point = point.strip()
            if not point:
                continue
                
            # Remove leading punctuation and markers
            point = re.sub(r'^[-\d.)•*]\s*', '', point)
            
            # Remove speaker prefixes if they appear at the start
            point = re.sub(r'^[A-Z][a-z]+\s*[a-z]*:\s*', '', point, count=1)
            
            # Fix grammar and formatting
            point = self._format_key_point(point)
            
            if point and len(point) > 10:  # Minimum length
                cleaned_points.append(point)
        
        # If we found more points than requested, that's fine - keep them all
        # Only limit if we have way too many (more than 2x requested) to avoid overwhelming output
        if len(cleaned_points) > num_points * 2:
            logger.info(f"Found {len(cleaned_points)} key points, limiting to {num_points * 2} to avoid overwhelming output")
            return cleaned_points[:num_points * 2]
        elif len(cleaned_points) < num_points:
            logger.info(f"Found {len(cleaned_points)} key points (requested {num_points})")
        
        return cleaned_points
    
    def _format_key_point(self, text: str) -> str:
        """Format a key point with proper grammar, capitalization, and punctuation."""
        if not text:
            return text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove multiple punctuation
        text = re.sub(r'[.!?]+', '.', text)
        
        # Remove trailing commas and other punctuation that shouldn't be at the end
        text = re.sub(r'[,;:]+$', '', text)
        
        # Ensure proper capitalization at start
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        elif text and not text[0].isupper() and text[0].isalpha():
            text = text[0].upper() + text[1:]
        
        # Fix "i" to "I"
        text = re.sub(r'\bi\b', 'I', text)
        text = re.sub(r'\bi\'', 'I\'', text)
        
        # Fix common grammar issues
        text = self._fix_grammar_issues(text)
        
        # Make it a complete, standalone statement
        # Remove incomplete sentence fragments
        if text.lower().startswith(('to ', 'for ', 'with ', 'in ', 'on ', 'at ', 'by ')):
            # If it starts with a preposition, it might be a fragment
            # Check if it's a complete thought
            if not re.search(r'\b(is|are|was|were|has|have|had|do|does|did|can|could|will|would|should|may|might|shows|demonstrates|explains|describes|discusses)\b', text.lower()):
                # Try to make it a complete sentence
                if not text.endswith('.'):
                    # Add a verb if missing
                    pass  # Keep as is for now, let context determine
        
        # Ensure it ends with proper punctuation
        if text and text[-1] not in '.!?':
            # Always add period for key points for consistency
            text += '.'
        
        # Remove trailing spaces and clean up
        text = text.strip()
        
        # Capitalize first letter if needed (in case fix_grammar_issues didn't)
        if text and text[0].islower() and len(text) > 1:
            text = text[0].upper() + text[1:]
        
        # Final pass: ensure it reads well as a standalone point
        # Remove redundant words at the start
        text = re.sub(r'^(The video|This video|The presentation|This presentation)\s+', '', text, flags=re.IGNORECASE)
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        return text
    
    def _extractive_key_points(self, transcript: str, num_points: int) -> List[str]:
        """
        Fallback extractive key point extraction.
        
        Args:
            transcript: Full transcript
            num_points: Number of points to extract
            
        Returns:
            List of key sentences
        """
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        
        # Simple heuristic: take longer sentences (likely more informative)
        sentences_with_length = [(s, len(s)) for s in sentences if len(s) > 50]
        sentences_with_length.sort(key=lambda x: x[1], reverse=True)
        
        key_sentences = [s[0] for s in sentences_with_length[:num_points]]
        return key_sentences
    
    def _extract_visual_context(self,
                               slide_changes: Optional[List[Dict]] = None,
                               visual_metadata: Optional[Dict] = None) -> str:
        """
        Extract visual context description from visual processing results.
        
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
    
    def save_summary_to_json(self,
                           video_path: str,
                           summary: str,
                           key_points: Optional[List[str]] = None) -> str:
        """
        Save generated summary to JSON file.
        
        Args:
            video_path: Path to original video file
            summary: Generated summary text
            key_points: Optional list of key points
            
        Returns:
            Path to saved JSON file
        """
        from src.utils.file_utils import get_video_metadata_dir, save_json
        from pathlib import Path
        
        metadata_dir = get_video_metadata_dir(video_path)
        json_path = metadata_dir / "generated_summary.json"
        
        summary_data = {
            "video_path": video_path,
            "model_name": self.model_name,
            "summary": summary,
            "key_points": key_points or [],
            "summary_length": len(summary)
        }
        
        save_json(summary_data, str(json_path))
        logger.info(f"Saved generated summary to {json_path}")
        return str(json_path)
