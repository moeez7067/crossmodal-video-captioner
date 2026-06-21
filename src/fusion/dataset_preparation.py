"""
Dataset Preparation Tools for Fusion Model Training.

This module provides utilities for preparing datasets for training the fusion model,
including data loading, preprocessing, splitting, and augmentation.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import shutil

logger = logging.getLogger(__name__)


class DatasetPreparator:
    """
    Prepares datasets for fusion model training.
    
    Handles:
    - Loading embeddings from processed videos
    - Creating train/val/test splits
    - Data augmentation
    - Metadata management
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize dataset preparator.
        
        Args:
            data_dir: Directory containing processed video data
            output_dir: Directory to save prepared datasets
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_samples(self) -> List[Dict]:
        """
        Collect all available samples from data directory.
        
        Returns:
            List of sample dictionaries with paths to embeddings
        """
        samples = []
        
        # Look for embeddings in data/{video_name}/embeddings/
        for video_dir in self.data_dir.iterdir():
            if not video_dir.is_dir():
                continue
                
            embeddings_dir = video_dir / "embeddings"
            if not embeddings_dir.exists():
                continue
                
            audio_path = embeddings_dir / "audio_embeddings.npy"
            visual_path = embeddings_dir / "visual_embeddings.npy"
            metadata_path = video_dir / "metadata.json"
            
            if audio_path.exists() and visual_path.exists():
                sample = {
                    'video_id': video_dir.name,
                    'audio_path': str(audio_path),
                    'visual_path': str(visual_path),
                    'metadata_path': str(metadata_path) if metadata_path.exists() else None
                }
                samples.append(sample)
        
        logger.info(f"Collected {len(samples)} samples from {self.data_dir}")
        return samples
    
    def create_splits(
        self,
        samples: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split samples into train/val/test sets.
        
        Args:
            samples: List of sample dictionaries
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        np.random.seed(random_seed)
        indices = np.random.permutation(len(samples))
        
        n_train = int(len(samples) * train_ratio)
        n_val = int(len(samples) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        test_samples = [samples[i] for i in test_indices]
        
        logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        
        return train_samples, val_samples, test_samples
    
    def prepare_dataset(
        self,
        samples: List[Dict],
        split_name: str,
        copy_files: bool = False
    ) -> Path:
        """
        Prepare dataset for a specific split.
        
        Args:
            samples: List of samples for this split
            split_name: Name of split (train/val/test)
            copy_files: Whether to copy files to output directory
            
        Returns:
            Path to prepared dataset directory
        """
        split_dir = self.output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Create metadata file
        metadata = {
            'split': split_name,
            'num_samples': len(samples),
            'samples': []
        }
        
        for idx, sample in enumerate(samples):
            if copy_files:
                # Copy files to split directory
                sample_dir = split_dir / f"sample_{idx:05d}"
                sample_dir.mkdir(exist_ok=True)
                
                shutil.copy(sample['audio_path'], sample_dir / "audio_embeddings.npy")
                shutil.copy(sample['visual_path'], sample_dir / "visual_embeddings.npy")
                
                if sample['metadata_path']:
                    shutil.copy(sample['metadata_path'], sample_dir / "metadata.json")
                
                metadata['samples'].append({
                    'sample_id': f"sample_{idx:05d}",
                    'video_id': sample['video_id'],
                    'audio_path': str(sample_dir / "audio_embeddings.npy"),
                    'visual_path': str(sample_dir / "visual_embeddings.npy")
                })
            else:
                # Just reference original paths
                metadata['samples'].append({
                    'sample_id': f"sample_{idx:05d}",
                    'video_id': sample['video_id'],
                    'audio_path': sample['audio_path'],
                    'visual_path': sample['visual_path']
                })
        
        # Save metadata
        metadata_path = split_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Prepared {split_name} dataset with {len(samples)} samples at {split_dir}")
        return split_dir
    
    def prepare_all_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        copy_files: bool = False
    ) -> Dict[str, Path]:
        """
        Prepare train/val/test splits.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            copy_files: Whether to copy files to output directory
            
        Returns:
            Dictionary mapping split names to their directories
        """
        # Collect samples
        samples = self.collect_samples()
        
        if len(samples) == 0:
            raise ValueError(f"No samples found in {self.data_dir}")
        
        # Create splits
        train_samples, val_samples, test_samples = self.create_splits(
            samples, train_ratio, val_ratio, test_ratio
        )
        
        # Prepare each split
        splits = {}
        splits['train'] = self.prepare_dataset(train_samples, 'train', copy_files)
        splits['val'] = self.prepare_dataset(val_samples, 'val', copy_files)
        splits['test'] = self.prepare_dataset(test_samples, 'test', copy_files)
        
        # Save overall metadata
        overall_metadata = {
            'total_samples': len(samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'splits': {name: str(path) for name, path in splits.items()}
        }
        
        metadata_path = self.output_dir / "dataset_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(overall_metadata, f, indent=2)
        
        logger.info(f"Prepared all splits. Total: {len(samples)} samples")
        return splits


def prepare_training_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    """
    Convenience function to prepare training dataset.
    
    Args:
        data_dir: Directory containing processed video data
        output_dir: Directory to save prepared datasets
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    preparator = DatasetPreparator(data_dir, output_dir)
    splits = preparator.prepare_all_splits(train_ratio, val_ratio, test_ratio)
    return splits


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python dataset_preparation.py <data_dir> <output_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    splits = prepare_training_dataset(data_dir, output_dir)
    print(f"Prepared datasets:")
    for name, path in splits.items():
        print(f"  {name}: {path}")
