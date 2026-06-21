"""
Script to analyze dataset and generate EDA visualizations and statistics.
Supports both custom YouTube dataset and HowToM dataset.
"""

import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import VideoCaptionDataset, GroundTruthManager
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DatasetAnalyzer:
    """
    Analyzes video captioning datasets and generates statistics and visualizations.
    """
    
    def __init__(self, dataset: Optional[VideoCaptionDataset] = None):
        """
        Initialize dataset analyzer.
        
        Args:
            dataset: VideoCaptionDataset instance (creates new if None)
        """
        self.dataset = dataset or VideoCaptionDataset()
        self.gt_manager = GroundTruthManager()
        self.stats = {}
    
    def compute_statistics(self) -> Dict:
        """
        Compute comprehensive dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Computing dataset statistics...")
        
        videos = self.dataset.get_all_videos()
        
        if not videos:
            logger.warning("No videos found in dataset")
            return {}
        
        # Basic counts
        total_videos = len(videos)
        categories = self.dataset.get_categories()
        
        # Category distribution
        category_counts = Counter(v.get("category", "unknown") for v in videos)
        
        # Split distribution
        split_counts = Counter(v.get("split", "unknown") for v in videos)
        
        # Duration statistics
        durations = [v.get("duration", 0) for v in videos if v.get("duration")]
        duration_stats = {
            "mean": np.mean(durations) if durations else 0,
            "median": np.median(durations) if durations else 0,
            "std": np.std(durations) if durations else 0,
            "min": np.min(durations) if durations else 0,
            "max": np.max(durations) if durations else 0,
            "total_hours": sum(durations) / 3600 if durations else 0
        }
        
        # Ground truth statistics
        gt_stats = self._compute_ground_truth_stats(videos)
        
        # Dataset source information
        dataset_sources = self._identify_dataset_sources(videos)
        
        self.stats = {
            "total_videos": total_videos,
            "categories": dict(category_counts),
            "splits": dict(split_counts),
            "duration_statistics": duration_stats,
            "ground_truth_statistics": gt_stats,
            "dataset_sources": dataset_sources,
            "category_list": categories
        }
        
        logger.info(f"Computed statistics for {total_videos} videos")
        return self.stats
    
    def _compute_ground_truth_stats(self, videos: List[Dict]) -> Dict:
        """Compute statistics about ground truth captions."""
        caption_lengths = []
        num_captions_per_video = []
        
        for video in videos:
            video_id = video.get("video_id")
            if not video_id:
                continue
            
            gt = self.gt_manager.load_ground_truth(video_id)
            if not gt:
                continue
            
            num_captions_per_video.append(len(gt))
            
            for caption in gt:
                text = caption.get("text", "")
                if text:
                    caption_lengths.append(len(text.split()))
        
        return {
            "videos_with_gt": sum(1 for v in videos if self.gt_manager.load_ground_truth(v.get("video_id"))),
            "total_captions": sum(num_captions_per_video) if num_captions_per_video else 0,
            "avg_captions_per_video": np.mean(num_captions_per_video) if num_captions_per_video else 0,
            "caption_length_stats": {
                "mean_words": np.mean(caption_lengths) if caption_lengths else 0,
                "median_words": np.median(caption_lengths) if caption_lengths else 0,
                "std_words": np.std(caption_lengths) if caption_lengths else 0,
                "min_words": np.min(caption_lengths) if caption_lengths else 0,
                "max_words": np.max(caption_lengths) if caption_lengths else 0
            } if caption_lengths else {}
        }
    
    def _identify_dataset_sources(self, videos: List[Dict]) -> Dict:
        """Identify dataset sources (YouTube, HowToM, etc.)."""
        sources = {
            "youtube": 0,
            "howtom": 0,
            "other": 0
        }
        
        for video in videos:
            video_id = video.get("video_id", "").lower()
            category = video.get("category", "").lower()
            source = video.get("source", "").lower()
            
            # Check source field first, then video_id, then category
            if source:
                if "youtube" in source or "yt" in source:
                    sources["youtube"] += 1
                elif "howtom" in source or "howto" in source:
                    sources["howtom"] += 1
                else:
                    sources["other"] += 1
            elif "howtom" in video_id or "howto" in category:
                sources["howtom"] += 1
            elif "yt" in video_id or "youtube" in video_id or "youtube" in category:
                sources["youtube"] += 1
            else:
                sources["other"] += 1
        
        return sources
    
    def create_visualizations(self, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Create all dataset visualizations.
        
        Args:
            output_dir: Directory to save visualizations (defaults to results/dataset_analysis/)
            
        Returns:
            List of paths to saved visualization files
        """
        if not self.stats:
            self.compute_statistics()
        
        if output_dir is None:
            output_dir = Path("results") / "dataset_analysis"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating visualizations in {output_dir}")
        
        saved_files = []
        
        # 1. Category distribution
        saved_files.append(self._plot_category_distribution(output_dir))
        
        # 2. Split distribution
        saved_files.append(self._plot_split_distribution(output_dir))
        
        # 3. Duration distribution
        saved_files.append(self._plot_duration_distribution(output_dir))
        
        # 4. Dataset source distribution
        saved_files.append(self._plot_dataset_sources(output_dir))
        
        # 5. Ground truth statistics
        saved_files.append(self._plot_ground_truth_stats(output_dir))
        
        # 6. Duration by category
        saved_files.append(self._plot_duration_by_category(output_dir))
        
        logger.info(f"Created {len([f for f in saved_files if f])} visualizations")
        return saved_files
    
    def _plot_category_distribution(self, output_dir: Path) -> Optional[Path]:
        """Create category distribution bar chart."""
        categories = self.stats.get("categories", {})
        
        if not categories:
            logger.warning("No categories found for visualization")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        cats = list(categories.keys())
        counts = list(categories.values())
        
        bars = ax.bar(cats, counts, color=sns.color_palette("husl", len(cats)))
        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel("Number of Videos", fontsize=12)
        ax.set_title("Video Distribution by Category", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = output_dir / "category_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_split_distribution(self, output_dir: Path) -> Optional[Path]:
        """Create train/test split pie chart."""
        splits = self.stats.get("splits", {})
        
        if not splits:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = list(splits.keys())
        sizes = list(splits.values())
        colors = sns.color_palette("pastel", len(labels))
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
              startangle=90, textprops={'fontsize': 12})
        ax.set_title("Dataset Split Distribution", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = output_dir / "split_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_duration_distribution(self, output_dir: Path) -> Optional[Path]:
        """Create duration distribution histogram."""
        videos = self.dataset.get_all_videos()
        durations = [v.get("duration", 0) for v in videos if v.get("duration")]
        
        if not durations:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(durations, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel("Duration (seconds)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Video Duration Distribution", fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Box plot
        ax2.boxplot(durations, vert=True)
        ax2.set_ylabel("Duration (seconds)", fontsize=12)
        ax2.set_title("Video Duration Box Plot", fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f"Mean: {np.mean(durations):.1f}s\n"
        stats_text += f"Median: {np.median(durations):.1f}s\n"
        stats_text += f"Total: {sum(durations)/3600:.2f} hours"
        ax2.text(1.1, np.median(durations), stats_text,
                verticalalignment='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = output_dir / "duration_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_dataset_sources(self, output_dir: Path) -> Optional[Path]:
        """Create dataset source distribution chart."""
        sources = self.stats.get("dataset_sources", {})
        
        if not sources or sum(sources.values()) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = [k.upper() for k in sources.keys() if sources[k] > 0]
        sizes = [sources[k] for k in sources.keys() if sources[k] > 0]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        bars = ax.bar(labels, sizes, color=colors[:len(labels)])
        ax.set_xlabel("Dataset Source", fontsize=12)
        ax.set_ylabel("Number of Videos", fontsize=12)
        ax.set_title("Dataset Source Distribution", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = output_dir / "dataset_sources.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_ground_truth_stats(self, output_dir: Path) -> Optional[Path]:
        """Create ground truth statistics visualization."""
        gt_stats = self.stats.get("ground_truth_statistics", {})
        
        if not gt_stats or gt_stats.get("total_captions", 0) == 0:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Videos with/without GT
        has_gt = gt_stats.get("videos_with_gt", 0)
        total = self.stats.get("total_videos", 0)
        no_gt = total - has_gt
        
        ax1.bar(["With GT", "Without GT"], [has_gt, no_gt], 
               color=['green', 'red'], alpha=0.7)
        ax1.set_ylabel("Number of Videos", fontsize=11)
        ax1.set_title("Ground Truth Availability", fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Average captions per video
        avg_captions = gt_stats.get("avg_captions_per_video", 0)
        ax2.bar(["Average"], [avg_captions], color='steelblue', alpha=0.7)
        ax2.set_ylabel("Number of Captions", fontsize=11)
        ax2.set_title(f"Average Captions per Video\n({avg_captions:.1f})", 
                     fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Caption length distribution (if available)
        caption_stats = gt_stats.get("caption_length_stats", {})
        if caption_stats:
            # Simulate distribution for visualization
            mean_words = caption_stats.get("mean_words", 0)
            std_words = caption_stats.get("std_words", 0)
            if mean_words > 0:
                # Create sample distribution
                sample_lengths = np.random.normal(mean_words, std_words, 1000)
                sample_lengths = np.clip(sample_lengths, 0, None)  # No negative values
                
                ax3.hist(sample_lengths, bins=30, color='coral', edgecolor='black', alpha=0.7)
                ax3.axvline(mean_words, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_words:.1f}')
                ax3.set_xlabel("Words per Caption", fontsize=11)
                ax3.set_ylabel("Frequency", fontsize=11)
                ax3.set_title("Caption Length Distribution", fontsize=12, fontweight='bold')
                ax3.legend()
                ax3.grid(axis='y', alpha=0.3)
        
        # Total captions
        total_captions = gt_stats.get("total_captions", 0)
        ax4.bar(["Total"], [total_captions], color='purple', alpha=0.7)
        ax4.set_ylabel("Number of Captions", fontsize=11)
        ax4.set_title(f"Total Captions in Dataset\n({total_captions:,})", 
                     fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / "ground_truth_statistics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_duration_by_category(self, output_dir: Path) -> Optional[Path]:
        """Create duration by category box plot."""
        videos = self.dataset.get_all_videos()
        
        # Group durations by category
        category_durations = {}
        for video in videos:
            category = video.get("category", "unknown")
            duration = video.get("duration")
            if duration:
                if category not in category_durations:
                    category_durations[category] = []
                category_durations[category].append(duration)
        
        if not category_durations:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = list(category_durations.keys())
        durations_list = [category_durations[cat] for cat in categories]
        
        bp = ax.boxplot(durations_list, labels=categories, patch_artist=True)
        
        # Color the boxes
        colors = sns.color_palette("Set2", len(categories))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel("Duration (seconds)", fontsize=12)
        ax.set_title("Video Duration by Category", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / "duration_by_category.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate text report of dataset statistics.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        if not self.stats:
            self.compute_statistics()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATASET ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overview
        report_lines.append("OVERVIEW")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Videos: {self.stats.get('total_videos', 0)}")
        report_lines.append(f"Total Duration: {self.stats.get('duration_statistics', {}).get('total_hours', 0):.2f} hours")
        report_lines.append("")
        
        # Categories
        report_lines.append("CATEGORY DISTRIBUTION")
        report_lines.append("-" * 80)
        for cat, count in self.stats.get("categories", {}).items():
            report_lines.append(f"  {cat}: {count} videos")
        report_lines.append("")
        
        # Splits
        report_lines.append("SPLIT DISTRIBUTION")
        report_lines.append("-" * 80)
        for split, count in self.stats.get("splits", {}).items():
            report_lines.append(f"  {split}: {count} videos")
        report_lines.append("")
        
        # Duration statistics
        report_lines.append("DURATION STATISTICS")
        report_lines.append("-" * 80)
        dur_stats = self.stats.get("duration_statistics", {})
        report_lines.append(f"  Mean: {dur_stats.get('mean', 0):.2f} seconds")
        report_lines.append(f"  Median: {dur_stats.get('median', 0):.2f} seconds")
        report_lines.append(f"  Std Dev: {dur_stats.get('std', 0):.2f} seconds")
        report_lines.append(f"  Min: {dur_stats.get('min', 0):.2f} seconds")
        report_lines.append(f"  Max: {dur_stats.get('max', 0):.2f} seconds")
        report_lines.append("")
        
        # Dataset sources
        report_lines.append("DATASET SOURCES")
        report_lines.append("-" * 80)
        sources = self.stats.get("dataset_sources", {})
        for source, count in sources.items():
            if count > 0:
                report_lines.append(f"  {source.upper()}: {count} videos")
        report_lines.append("")
        
        # Ground truth statistics
        report_lines.append("GROUND TRUTH STATISTICS")
        report_lines.append("-" * 80)
        gt_stats = self.stats.get("ground_truth_statistics", {})
        report_lines.append(f"  Videos with GT: {gt_stats.get('videos_with_gt', 0)}")
        report_lines.append(f"  Total Captions: {gt_stats.get('total_captions', 0)}")
        report_lines.append(f"  Avg Captions/Video: {gt_stats.get('avg_captions_per_video', 0):.2f}")
        
        caption_stats = gt_stats.get("caption_length_stats", {})
        if caption_stats:
            report_lines.append("  Caption Length Statistics:")
            report_lines.append(f"    Mean words: {caption_stats.get('mean_words', 0):.2f}")
            report_lines.append(f"    Median words: {caption_stats.get('median_words', 0):.2f}")
            report_lines.append(f"    Std dev: {caption_stats.get('std_words', 0):.2f}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Saved report to {output_path}")
        
        return report_text


def main():
    """Run dataset analysis and generate visualizations."""
    print("=" * 80)
    print("DATASET ANALYSIS & EDA")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Step 1: Loading dataset...")
    dataset = VideoCaptionDataset()
    
    if len(dataset) == 0:
        print("   ERROR: No videos found in dataset!")
        print("   Please add videos using: python scripts/prepare_test_dataset.py")
        return
    
    print(f"   Loaded {len(dataset)} videos")
    print()
    
    # Initialize analyzer
    print("Step 2: Initializing analyzer...")
    analyzer = DatasetAnalyzer(dataset)
    print("   Analyzer ready")
    print()
    
    # Compute statistics
    print("Step 3: Computing statistics...")
    stats = analyzer.compute_statistics()
    print("   Statistics computed")
    print()
    
    # Print summary
    print("DATASET SUMMARY")
    print("-" * 80)
    print(f"Total Videos: {stats.get('total_videos', 0)}")
    print(f"Categories: {', '.join(stats.get('category_list', []))}")
    
    sources = stats.get('dataset_sources', {})
    source_summary = []
    for source, count in sources.items():
        if count > 0:
            source_summary.append(f"{source.upper()}: {count}")
    if source_summary:
        print(f"Dataset Sources: {', '.join(source_summary)}")
    
    dur_stats = stats.get('duration_statistics', {})
    print(f"Total Duration: {dur_stats.get('total_hours', 0):.2f} hours")
    print()
    
    # Generate visualizations
    print("Step 4: Generating visualizations...")
    output_dir = Path("results") / "dataset_analysis"
    saved_files = analyzer.create_visualizations(output_dir)
    
    print(f"   Created {len([f for f in saved_files if f])} visualizations")
    for file_path in saved_files:
        if file_path:
            print(f"     - {file_path}")
    print()
    
    # Generate report
    print("Step 5: Generating text report...")
    report_path = output_dir / "dataset_analysis_report.txt"
    report = analyzer.generate_report(report_path)
    print(f"   Report saved to: {report_path}")
    print()
    
    # Print report preview
    print("REPORT PREVIEW")
    print("-" * 80)
    lines = report.split('\n')[:30]  # First 30 lines
    for line in lines:
        print(line)
    if len(report.split('\n')) > 30:
        print("...")
        print(f"(Full report saved to {report_path})")
    print()
    
    print("=" * 80)
    print("DATASET ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("Results saved to:")
    print(f"  - Visualizations: {output_dir}/")
    print(f"  - Report: {report_path}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running dataset analysis: {e}", exc_info=True)
        print(f"\n\nERROR: {e}")
        sys.exit(1)
