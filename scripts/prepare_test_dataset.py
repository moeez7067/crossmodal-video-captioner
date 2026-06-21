"""
Script to help prepare test dataset for comparative analysis.
This script helps organize videos and create ground truth captions.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import shutil
from typing import List, Dict, Optional
import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_dataset_structure():
    """Create the test dataset directory structure."""
    base_dir = config.TEST_DATA_DIR
    dirs = [
        base_dir / "videos",
        base_dir / "ground_truth"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    # Create metadata file if it doesn't exist
    metadata_path = base_dir / "metadata.json"
    if not metadata_path.exists():
        metadata = {
            "dataset_name": "video_captioning_test_set",
            "description": "Test dataset for comparative analysis",
            "total_videos": 0,
            "categories": {
                "lecture": 0,
                "tutorial": 0,
                "conversation": 0,
                "presentation": 0
            },
            "split": {
                "train": 0,
                "test": 0
            },
            "videos": []
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Created metadata file: {metadata_path}")


def add_video_to_dataset(
    video_path: str,
    video_id: Optional[str] = None,
    category: str = "lecture",
    split: str = "test",
    duration: Optional[float] = None
):
    """
    Add a video to the test dataset.
    
    Args:
        video_path: Path to source video file
        video_id: Unique video identifier (default: filename without extension)
        category: Video category (lecture, tutorial, conversation, presentation)
        split: Dataset split (train or test)
        duration: Video duration in seconds (optional)
    """
    source_path = Path(video_path)
    if not source_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return False
    
    # Generate video_id if not provided
    if video_id is None:
        video_id = source_path.stem
    
    # Load existing metadata
    metadata_path = config.TEST_DATA_DIR / "metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Check if video already exists
    existing_ids = [v.get("video_id") for v in metadata["videos"]]
    if video_id in existing_ids:
        logger.warning(f"Video {video_id} already exists in dataset")
        return False
    
    # Copy video to dataset directory
    dest_dir = config.TEST_DATA_DIR / "videos"
    dest_path = dest_dir / source_path.name
    
    try:
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied video: {source_path.name} -> {dest_path}")
    except Exception as e:
        logger.error(f"Error copying video: {e}")
        return False
    
    # Get video duration if not provided
    if duration is None:
        try:
            import cv2
            cap = cv2.VideoCapture(str(dest_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
            duration = 0
    
    # Add to metadata
    video_entry = {
        "video_id": video_id,
        "file_path": f"videos/{source_path.name}",
        "category": category,
        "duration": duration,
        "ground_truth": f"ground_truth/{video_id}.json",
        "split": split
    }
    
    metadata["videos"].append(video_entry)
    metadata["total_videos"] = len(metadata["videos"])
    
    # Update category count
    if category in metadata["categories"]:
        metadata["categories"][category] += 1
    
    # Update split count
    metadata["split"][split] += 1
    
    # Save metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Added video {video_id} to dataset")
    return True


def create_ground_truth_template(video_id: str, num_segments: int = 10):
    """
    Create a template ground truth file for manual annotation.
    
    Args:
        video_id: Video identifier
        num_segments: Number of caption segments to create template for
    """
    gt_dir = config.TEST_DATA_DIR / "ground_truth"
    gt_path = gt_dir / f"{video_id}.json"
    
    # Calculate segment duration (assuming 30-second video segments)
    segment_duration = 30.0 / num_segments
    
    template = {
        "video_id": video_id,
        "captions": [
            {
                "text": f"[Caption {i+1} - Edit this text]",
                "start_time": i * segment_duration,
                "end_time": (i + 1) * segment_duration
            }
            for i in range(num_segments)
        ],
        "num_captions": num_segments
    }
    
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created ground truth template: {gt_path}")
    return gt_path


def import_srt_as_ground_truth(video_id: str, srt_path: str):
    """
    Import SRT subtitle file as ground truth captions.
    
    Args:
        video_id: Video identifier
        srt_path: Path to SRT file
    """
    from src.utils.time_utils import parse_srt
    
    try:
        captions = parse_srt(srt_path)
        
        gt_dir = config.TEST_DATA_DIR / "ground_truth"
        gt_path = gt_dir / f"{video_id}.json"
        
        data = {
            "video_id": video_id,
            "captions": captions,
            "num_captions": len(captions),
            "source": "srt_import"
        }
        
        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Imported {len(captions)} captions from SRT: {gt_path}")
        return True
    except Exception as e:
        logger.error(f"Error importing SRT: {e}")
        return False


def scan_and_add_all_videos(
    category: str = "lecture",
    split: str = "test",
    auto_detect: bool = True
):
    """
    Scan the videos directory and add all videos found to the metadata.
    
    Args:
        category: Default category for videos (lecture, tutorial, conversation, presentation)
        split: Dataset split (train or test)
        auto_detect: If True, try to infer video_id from filename (test_001, test_002, etc.)
    
    Returns:
        Number of videos added
    """
    videos_dir = config.TEST_DATA_DIR / "videos"
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return 0
    
    # Find all video files
    video_extensions = ['.mp4', '.mkv', '.mov', '.avi', '.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(videos_dir.glob(f"*{ext}")))
    
    if not video_files:
        logger.warning(f"No video files found in {videos_dir}")
        return 0
    
    logger.info(f"Found {len(video_files)} video files in {videos_dir}")
    
    # Load existing metadata
    metadata_path = config.TEST_DATA_DIR / "metadata.json"
    if not metadata_path.exists():
        create_dataset_structure()
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    existing_ids = {v.get("video_id") for v in metadata["videos"]}
    existing_files = {Path(v.get("file_path", "")).name for v in metadata["videos"]}
    
    added_count = 0
    
    for video_file in video_files:
        filename = video_file.name
        
        # Skip if already in metadata
        if filename in existing_files:
            logger.debug(f"Video {filename} already in metadata, skipping")
            continue
        
        # Generate video_id
        if auto_detect:
            # Try to extract test_XXX pattern
            stem = video_file.stem
            if stem.startswith("test_") and len(stem) > 5:
                video_id = stem
            else:
                # Use filename as video_id
                video_id = stem
        else:
            video_id = video_file.stem
        
        # Skip if video_id already exists
        if video_id in existing_ids:
            logger.warning(f"Video ID {video_id} already exists, skipping {filename}")
            continue
        
        # Get video duration
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
        except Exception as e:
            logger.warning(f"Could not get video duration for {filename}: {e}")
            duration = 0
        
        # Detect source based on video_id or category
        source = "other"
        if video_id.startswith("yt_") or category.lower() == "youtube":
            source = "youtube"
        elif video_id.startswith("howtom_") or category.lower() == "howtom":
            source = "howtom"
        
        # Add to metadata
        video_entry = {
            "video_id": video_id,
            "file_path": f"videos/{filename}",
            "category": category,
            "duration": duration,
            "ground_truth": f"ground_truth/{video_id}.json",
            "split": split,
            "source": source
        }
        
        metadata["videos"].append(video_entry)
        existing_ids.add(video_id)
        existing_files.add(filename)
        added_count += 1
        
        logger.info(f"Added video {video_id} ({filename}) to dataset")
    
    if added_count > 0:
        # Update metadata statistics
        metadata["total_videos"] = len(metadata["videos"])
        
        # Recalculate category counts
        metadata["categories"] = {
            "lecture": 0,
            "tutorial": 0,
            "conversation": 0,
            "presentation": 0
        }
        for video in metadata["videos"]:
            cat = video.get("category", "lecture")
            if cat in metadata["categories"]:
                metadata["categories"][cat] += 1
        
        # Recalculate split counts
        metadata["split"] = {"train": 0, "test": 0}
        for video in metadata["videos"]:
            split_name = video.get("split", "test")
            if split_name in metadata["split"]:
                metadata["split"][split_name] += 1
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Added {added_count} new video(s) to dataset. Total: {metadata['total_videos']}")
    else:
        logger.info("No new videos to add")
    
    return added_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare test dataset")
    parser.add_argument("--create-structure", action="store_true", help="Create directory structure")
    parser.add_argument("--add-video", type=str, help="Path to video file to add")
    parser.add_argument("--video-id", type=str, help="Video ID (default: filename)")
    parser.add_argument("--category", type=str, default="lecture", 
                       choices=["lecture", "tutorial", "conversation", "presentation"],
                       help="Video category")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                       help="Dataset split")
    parser.add_argument("--create-template", type=str, help="Create ground truth template for video_id")
    parser.add_argument("--import-srt", type=str, nargs=2, metavar=("VIDEO_ID", "SRT_PATH"),
                       help="Import SRT file as ground truth")
    parser.add_argument("--scan-videos", action="store_true",
                       help="Scan videos directory and add all videos to metadata")
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_dataset_structure()
    elif args.add_video:
        add_video_to_dataset(
            args.add_video,
            video_id=args.video_id,
            category=args.category,
            split=args.split
        )
    elif args.create_template:
        create_ground_truth_template(args.create_template)
    elif args.import_srt:
        video_id, srt_path = args.import_srt
        import_srt_as_ground_truth(video_id, srt_path)
    elif args.scan_videos:
        scan_and_add_all_videos(
            category=args.category,
            split=args.split
        )
    else:
        parser.print_help()
