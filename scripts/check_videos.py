"""Quick script to check what videos are in the directory"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config

videos_dir = config.TEST_DATA_DIR / "videos"
print(f"Checking directory: {videos_dir}")
print(f"Directory exists: {videos_dir.exists()}")

if videos_dir.exists():
    # Check all files
    all_files = list(videos_dir.iterdir())
    print(f"\nAll items in directory ({len(all_files)}):")
    for item in all_files:
        print(f"  - {item.name} (is_file: {item.is_file()}, is_dir: {item.is_dir()})")
    
    # Check video files specifically
    video_extensions = ['.mp4', '.mkv', '.mov', '.avi', '.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(videos_dir.glob(f"*{ext}")))
    
    print(f"\nVideo files found ({len(video_files)}):")
    for vf in video_files:
        print(f"  - {vf.name}")
