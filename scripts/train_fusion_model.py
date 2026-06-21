"""
Enhanced Training Script for Fusion Model.

Usage:
    python scripts/train_fusion_model.py --data_dir <dir> --output_dir <dir> [options]
"""

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.train import DatasetPreparator, MultimodalDataset, EnhancedFusionTrainer
from src.fusion.multimodal_transformer import MultimodalTransformer
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train fusion model")
    parser.add_argument("--data_dir", default="data", help="Directory with processed video data")
    parser.add_argument("--output_dir", default="data/training", help="Directory for prepared datasets")
    parser.add_argument("--checkpoint_dir", default="checkpoints/fusion", help="Directory for checkpoints")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--loss_type", choices=['reconstruction', 'contrastive', 'combined'], 
                       default='combined', help="Loss function type")
    parser.add_argument("--use_tensorboard", action='store_true', help="Use TensorBoard logging")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    preparator = DatasetPreparator(args.data_dir, args.output_dir)
    splits = preparator.prepare_all_splits(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        copy_files=False  # Use original paths to save space
    )
    
    # Create datasets
    train_dataset = MultimodalDataset(
        data_dir=str(splits['train']),
        audio_dim=config.AUDIO_EMBEDDING_DIM,
        visual_dim=config.VISUAL_EMBEDDING_DIM,
        max_length=args.max_length
    )
    
    val_dataset = MultimodalDataset(
        data_dir=str(splits['val']),
        audio_dim=config.AUDIO_EMBEDDING_DIM,
        visual_dim=config.VISUAL_EMBEDDING_DIM,
        max_length=args.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Create model
    model = MultimodalTransformer(
        audio_input_dim=config.AUDIO_EMBEDDING_DIM,
        visual_input_dim=config.VISUAL_EMBEDDING_DIM,
        hidden_dim=config.FUSION_HIDDEN_DIM,
        num_layers=config.FUSION_NUM_LAYERS,
        num_heads=config.FUSION_NUM_HEADS,
        dropout=config.FUSION_DROPOUT
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = EnhancedFusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=device,
        loss_type=args.loss_type,
        use_tensorboard=args.use_tensorboard
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
        logger.info(f"Resumed training from epoch {trainer.current_epoch}")
    
    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.checkpoint_dir,
        save_every=5,
        early_stopping=True
    )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
