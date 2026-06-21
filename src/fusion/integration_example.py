"""
Integration Example: Connecting Phase 2 and Phase 3
Shows how to use fusion pipeline with existing audio/visual modules
"""

import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Phase 2 imports (adjust paths as needed)
# from src.audio.speech_to_text import transcribe
# from src.visual.visual_embeddings import extract_embeddings
# from src.preprocessing.video_processor import extract_frames

# Phase 3 imports
from src.fusion.fusion_pipeline import create_fusion_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedVideoProcessor:
    """
    Integrated processor that combines Phase 2 extraction with Phase 3 fusion.
    """
    
    def __init__(
        self,
        fusion_config: Dict = None,
        device: str = "cuda"
    ):
        """
        Initialize integrated processor.
        
        Args:
            fusion_config: Configuration for fusion pipeline
            device: Device to use
        """
        self.device = device
        
        # Initialize fusion pipeline
        logger.info("Initializing fusion pipeline...")
        self.fusion_pipeline = create_fusion_pipeline(
            config=fusion_config,
            device=device
        )
        
        # Note: In real implementation, also initialize:
        # - Speech-to-text model
        # - CLIP model for visual embeddings
        # - Video processor
        
    def process_video(
        self,
        video_path: str,
        output_dir: str = "output"
    ) -> Dict:
        """
        Complete video processing pipeline.
        
        Args:
            video_path: Path to video file
            output_dir: Directory for outputs
            
        Returns:
            Dictionary with all processing results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Processing video: {video_path}")
        
        # Step 1: Extract audio embeddings (from Phase 2)
        logger.info("Step 1: Extracting audio embeddings...")
        audio_embeddings, audio_timestamps = self.extract_audio_embeddings(video_path)
        logger.info(f"Audio embeddings shape: {audio_embeddings.shape}")
        
        # Step 2: Extract visual embeddings (from Phase 2)
        logger.info("Step 2: Extracting visual embeddings...")
        visual_embeddings, visual_timestamps = self.extract_visual_embeddings(video_path)
        logger.info(f"Visual embeddings shape: {visual_embeddings.shape}")
        
        # Step 3: Fuse audio and visual (Phase 3)
        logger.info("Step 3: Fusing modalities...")
        fusion_result = self.fusion_pipeline.fuse(
            audio_embeddings=audio_embeddings,
            visual_embeddings=visual_embeddings,
            timestamps=audio_timestamps,
            align_method="interpolate",
            return_attention=True
        )
        
        # Step 4: Create segment representations
        logger.info("Step 4: Creating segment representations...")
        segments = self.create_segments(
            fusion_result['fused_embeddings'],
            audio_timestamps
        )
        
        # Compile results
        results = {
            'audio_embeddings': audio_embeddings,
            'visual_embeddings': visual_embeddings,
            'fused_embeddings': fusion_result['fused_embeddings'],
            'segments': segments,
            'timestamps': audio_timestamps,
            'attention_weights': fusion_result.get('attention_weights'),
            'fusion_dim': fusion_result['fusion_dim']
        }
        
        # Save results
        self.save_results(results, output_path)
        
        logger.info("Processing complete!")
        return results
    
    def extract_audio_embeddings(
        self,
        video_path: str
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Extract audio embeddings using Phase 2 modules.
        
        In real implementation, this would:
        1. Extract audio from video
        2. Transcribe with Whisper (gets embeddings)
        3. Return embeddings and timestamps
        
        For now, returns dummy data for demonstration.
        """
        # Dummy implementation - replace with actual Phase 2 code
        logger.info("Extracting audio (using dummy data for demo)...")
        
        # Simulate 100 audio segments of 768-dim embeddings (Whisper size)
        num_segments = 100
        audio_dim = 768
        audio_embeddings = np.random.randn(num_segments, audio_dim).astype(np.float32)
        
        # Simulate timestamps (every 0.5 seconds)
        timestamps = [i * 0.5 for i in range(num_segments)]
        
        return audio_embeddings, timestamps
    
    def extract_visual_embeddings(
        self,
        video_path: str
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Extract visual embeddings using Phase 2 modules.
        
        In real implementation, this would:
        1. Extract frames
        2. Get CLIP embeddings
        3. Return embeddings and timestamps
        
        For now, returns dummy data for demonstration.
        """
        # Dummy implementation - replace with actual Phase 2 code
        logger.info("Extracting visual (using dummy data for demo)...")
        
        # Simulate 100 visual segments of 512-dim embeddings (CLIP size)
        num_segments = 100
        visual_dim = 512
        visual_embeddings = np.random.randn(num_segments, visual_dim).astype(np.float32)
        
        # Simulate timestamps (every 0.5 seconds)
        timestamps = [i * 0.5 for i in range(num_segments)]
        
        return visual_embeddings, timestamps
    
    def create_segments(
        self,
        fused_embeddings: np.ndarray,
        timestamps: List[float],
        segment_length: float = 5.0
    ) -> List[Dict]:
        """
        Create time-based segments from fused embeddings.
        
        Args:
            fused_embeddings: Fused embeddings [seq_len, dim]
            timestamps: Timestamp for each embedding
            segment_length: Length of each segment in seconds
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        current_start = 0
        current_embeddings = []
        
        for i, ts in enumerate(timestamps):
            current_embeddings.append(fused_embeddings[i])
            
            # Check if we've reached segment boundary
            if ts - current_start >= segment_length or i == len(timestamps) - 1:
                # Pool embeddings in this segment
                segment_repr = np.mean(current_embeddings, axis=0)
                
                segments.append({
                    'start_time': current_start,
                    'end_time': ts,
                    'embedding': segment_repr,
                    'duration': ts - current_start
                })
                
                # Reset for next segment
                current_start = ts
                current_embeddings = []
        
        logger.info(f"Created {len(segments)} segments")
        return segments
    
    def save_results(self, results: Dict, output_path: Path):
        """Save processing results to disk."""
        # Save fused embeddings
        np.save(output_path / "fused_embeddings.npy", results['fused_embeddings'])
        
        # Save segments
        segment_data = {
            'segments': [
                {
                    'start_time': seg['start_time'],
                    'end_time': seg['end_time'],
                    'duration': seg['duration']
                }
                for seg in results['segments']
            ]
        }
        
        import json
        with open(output_path / "segments.json", 'w') as f:
            json.dump(segment_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def example_usage():
    """Example of how to use the integrated processor."""
    
    # Configuration
    fusion_config = {
        'audio_dim': 768,      # Whisper embedding size
        'visual_dim': 512,     # CLIP embedding size
        'hidden_dim': 512,     # Transformer hidden size
        'num_layers': 4,       # Number of transformer layers
        'num_heads': 8,        # Attention heads
        'dropout': 0.1
    }
    
    # Initialize processor
    processor = IntegratedVideoProcessor(
        fusion_config=fusion_config,
        device="cuda"  # or "cpu"
    )
    
    # Process video
    results = processor.process_video(
        video_path="path/to/video.mp4",
        output_dir="output/fused_results"
    )
    
    # Access results
    print(f"Fused embeddings shape: {results['fused_embeddings'].shape}")
    print(f"Number of segments: {len(results['segments'])}")
    print(f"Fusion dimension: {results['fusion_dim']}")
    
    # Example: Use segments for caption generation (Phase 4)
    for i, segment in enumerate(results['segments'][:5]):  # First 5 segments
        print(f"\nSegment {i}:")
        print(f"  Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s")
        print(f"  Embedding shape: {segment['embedding'].shape}")
        # In Phase 4, you would pass segment['embedding'] to caption generator


def example_with_real_data():
    """
    Example showing how to integrate with real Phase 2 outputs.
    Assumes you have already processed a video through Phase 2.
    """
    
    # Load pre-extracted embeddings from Phase 2
    audio_embeddings = np.load("output/audio_embeddings.npy")
    visual_embeddings = np.load("output/visual_embeddings.npy")
    
    # Load timestamps
    import json
    with open("output/timestamps.json", 'r') as f:
        data = json.load(f)
        timestamps = data['timestamps']
    
    # Initialize fusion pipeline
    fusion_pipeline = create_fusion_pipeline(device="cuda")
    
    # Fuse embeddings
    result = fusion_pipeline.fuse(
        audio_embeddings=audio_embeddings,
        visual_embeddings=visual_embeddings,
        timestamps=timestamps,
        align_method="interpolate",
        return_attention=True
    )
    
    print(f"Fused embeddings shape: {result['fused_embeddings'].shape}")
    
    # Visualize attention (optional)
    if result.get('attention_weights'):
        from src.fusion.fusion_pipeline import visualize_attention
        visualize_attention(
            result['attention_weights'],
            layer_idx=0,
            save_path="output/attention_map.png"
        )
    
    # Save fused embeddings
    np.save("output/fused_embeddings.npy", result['fused_embeddings'])
    print("Fused embeddings saved!")


if __name__ == "__main__":
    # Run example with dummy data
    print("=" * 50)
    print("Example 1: Using dummy data")
    print("=" * 50)
    example_usage()
    
    print("\n" + "=" * 50)
    print("Example 2: Using real Phase 2 outputs")
    print("=" * 50)
    print("Uncomment and modify paths in example_with_real_data()")
    # example_with_real_data()