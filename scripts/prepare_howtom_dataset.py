"""
Script to prepare HowToM dataset for evaluation.
Loads a subset of HowToM videos (50 videos) for testing.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import VideoCaptionDataset
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


def load_howtom_metadata(howtom_path: Path, max_videos: int = 50) -> List[Dict]:
    """
    Load HowToM dataset metadata.
    
    Args:
        howtom_path: Path to HowToM dataset directory
        max_videos: Maximum number of videos to load (default: 50)
        
    Returns:
        List of video metadata dictionaries
    """
    videos = []
    
    # HowToM typically has structure like:
    # howtom/
    #   video_001/
    #     video.mp4
    #     captions.json
    #   video_002/
    #     ...
    
    howtom_path = Path(howtom_path)
    if not howtom_path.exists():
        logger.warning(f"HowToM dataset path not found: {howtom_path}")
        return videos
    
    # Find all video directories
    video_dirs = [d for d in howtom_path.iterdir() if d.is_dir()][:max_videos]
    
    for video_dir in video_dirs:
        video_file = None
        # Look for video file
        for ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm']:
            video_files = list(video_dir.glob(f"*{ext}"))
            if video_files:
                video_file = video_files[0]
                break
        
        if not video_file:
            continue
        
        # Try to get duration using ffprobe or default
        duration = None
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(video_file)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                duration = float(result.stdout.strip())
        except Exception as e:
            logger.debug(f"Could not get duration for {video_file}: {e}")
        
        video_metadata = {
            "video_id": f"howtom_{video_dir.name}",
            "file_path": str(video_file.relative_to(config.DATA_DIR)) if video_file.is_relative_to(config.DATA_DIR) else str(video_file),
            "category": "howtom",
            "duration": duration,
            "split": "test",
            "source": "howtom"
        }
        
        videos.append(video_metadata)
    
    logger.info(f"Loaded {len(videos)} HowToM videos")
    return videos


def add_howtom_to_dataset(howtom_path: str, max_videos: int = 50):
    """
    Add HowToM videos to the test dataset.
    
    Args:
        howtom_path: Path to HowToM dataset directory
        max_videos: Maximum number of videos to add (default: 50)
    """
    howtom_path = Path(howtom_path)
    
    if not howtom_path.exists():
        logger.error(f"HowToM dataset path not found: {howtom_path}")
        print(f"ERROR: HowToM dataset not found at {howtom_path}")
        print("Please provide the correct path to your HowToM dataset.")
        return
    
    print(f"Loading HowToM videos from: {howtom_path}")
    print(f"Maximum videos to load: {max_videos}")
    
    # Load HowToM videos
    howtom_videos = load_howtom_metadata(howtom_path, max_videos)
    
    if not howtom_videos:
        print("No HowToM videos found!")
        return
    
    # Load existing dataset
    metadata_path = config.DATA_DIR / "test_set" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {
            "dataset_name": "video_captioning_test_set",
            "description": "Test dataset including custom YouTube videos and HowToM subset",
            "total_videos": 0,
            "categories": {},
            "split": {"train": 0, "test": 0},
            "videos": []
        }
    
    # Add HowToM videos
    existing_ids = {v.get("video_id") for v in metadata.get("videos", [])}
    new_videos = [v for v in howtom_videos if v.get("video_id") not in existing_ids]
    
    metadata["videos"].extend(new_videos)
    metadata["total_videos"] = len(metadata["videos"])
    
    # Update category counts
    category_counts = {}
    for video in metadata["videos"]:
        cat = video.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    metadata["categories"] = category_counts
    
    # Update split counts
    split_counts = {"train": 0, "test": 0}
    for video in metadata["videos"]:
        split = video.get("split", "test")
        split_counts[split] = split_counts.get(split, 0) + 1
    metadata["split"] = split_counts
    
    # Save updated metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Added {len(new_videos)} HowToM videos to dataset")
    print(f"Total videos in dataset: {metadata['total_videos']}")


def main():
    """Main function to prepare HowToM dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare HowToM dataset for evaluation")
    parser.add_argument(
        "--howtom-path",
        type=str,
        required=True,
        help="Path to HowToM dataset directory"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=50,
        help="Maximum number of videos to load (default: 50)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HOWTOM DATASET PREPARATION")
    print("=" * 80)
    print()
    
    add_howtom_to_dataset(args.howtom_path, args.max_videos)
    
    print()
    print("=" * 80)
    print("HOWTOM DATASET PREPARATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPreparation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error preparing HowToM dataset: {e}", exc_info=True)
        print(f"\n\nERROR: {e}")
        sys.exit(1)
