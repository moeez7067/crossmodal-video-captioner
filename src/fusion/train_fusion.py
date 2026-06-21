"""
Training Script for Multimodal Fusion Model
Optional: For those who want to fine-tune the fusion transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import json

from src.fusion.multimodal_transformer import MultimodalTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal audio-visual data.
    """
    
    def __init__(
        self,
        data_dir: str,
        audio_dim: int = 768,
        visual_dim: int = 512,
        max_length: int = 100
    ):
        """
        Args:
            data_dir: Directory containing processed data
            audio_dim: Dimension of audio embeddings
            visual_dim: Dimension of visual embeddings
            max_length: Maximum sequence length
        """
        self.data_dir = Path(data_dir)
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.max_length = max_length
        
        # Load data file list
        self.samples = self.load_samples()
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def load_samples(self) -> List[Dict]:
        """Load list of samples from data directory."""
        samples = []
        
        # Assuming data is organized as:
        # data_dir/
        #   video_001/
        #     audio_embeddings.npy
        #     visual_embeddings.npy
        #     metadata.json
        #   video_002/
        #     ...
        
        for video_dir in self.data_dir.iterdir():
            if video_dir.is_dir():
                audio_path = video_dir / "audio_embeddings.npy"
                visual_path = video_dir / "visual_embeddings.npy"
                metadata_path = video_dir / "metadata.json"
                
                if audio_path.exists() and visual_path.exists():
                    sample = {
                        'video_id': video_dir.name,
                        'audio_path': str(audio_path),
                        'visual_path': str(visual_path),
                        'metadata_path': str(metadata_path) if metadata_path.exists() else None
                    }
                    samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - audio: Audio embeddings [seq_len, audio_dim]
                - visual: Visual embeddings [seq_len, visual_dim]
                - audio_mask: Mask for audio [seq_len]
                - visual_mask: Mask for visual [seq_len]
        """
        sample = self.samples[idx]
        
        # Load embeddings
        audio = np.load(sample['audio_path'])
        visual = np.load(sample['visual_path'])
        
        # Truncate or pad to max_length
        audio, audio_mask = self.pad_or_truncate(audio, self.max_length)
        visual, visual_mask = self.pad_or_truncate(visual, self.max_length)
        
        return {
            'audio': torch.from_numpy(audio).float(),
            'visual': torch.from_numpy(visual).float(),
            'audio_mask': torch.from_numpy(audio_mask).bool(),
            'visual_mask': torch.from_numpy(visual_mask).bool(),
            'video_id': sample['video_id']
        }
    
    def pad_or_truncate(
        self,
        embeddings: np.ndarray,
        max_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad or truncate embeddings to max_length.
        
        Returns:
            embeddings: Padded/truncated embeddings [max_length, dim]
            mask: Attention mask [max_length]
        """
        seq_len, dim = embeddings.shape
        
        if seq_len > max_length:
            # Truncate
            embeddings = embeddings[:max_length]
            mask = np.ones(max_length)
        else:
            # Pad
            padding = np.zeros((max_length - seq_len, dim))
            embeddings = np.concatenate([embeddings, padding], axis=0)
            mask = np.concatenate([np.ones(seq_len), np.zeros(max_length - seq_len)])
        
        return embeddings, mask


class FusionTrainer:
    """Trainer for multimodal fusion model."""
    
    def __init__(
        self,
        model: MultimodalTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        device: str = "cuda"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Multimodal transformer model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            learning_rate: Learning rate
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Loss function (for contrastive learning or reconstruction)
        self.criterion = nn.MSELoss()
        
        self.best_val_loss = float('inf')
    
    def contrastive_loss(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Contrastive loss for audio-visual alignment.
        Encourages matched audio-visual pairs to be close in embedding space.
        
        Args:
            audio_features: Audio embeddings [batch, dim]
            visual_features: Visual embeddings [batch, dim]
            temperature: Temperature parameter
            
        Returns:
            loss: Contrastive loss
        """
        # Normalize features
        audio_features = nn.functional.normalize(audio_features, dim=-1)
        visual_features = nn.functional.normalize(visual_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(audio_features, visual_features.T) / temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = audio_features.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy loss for both directions
        loss_audio = nn.functional.cross_entropy(similarity, labels)
        loss_visual = nn.functional.cross_entropy(similarity.T, labels)
        
        return (loss_audio + loss_visual) / 2
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Move to device
            audio = batch['audio'].to(self.device)
            visual = batch['visual'].to(self.device)
            
            # Forward pass
            fused_embeddings, _ = self.model(audio, visual)
            
            # Compute loss (here using simple reconstruction)
            # In practice, you might use contrastive loss or other objectives
            audio_recon = fused_embeddings  # Simplified
            visual_recon = fused_embeddings
            
            # Mean pool for contrastive loss
            audio_pooled = torch.mean(fused_embeddings, dim=1)
            visual_pooled = torch.mean(fused_embeddings, dim=1)
            
            loss = self.contrastive_loss(audio_pooled, visual_pooled)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                audio = batch['audio'].to(self.device)
                visual = batch['visual'].to(self.device)
                
                fused_embeddings, _ = self.model(audio, visual)
                
                # Mean pool for contrastive loss
                audio_pooled = torch.mean(fused_embeddings, dim=1)
                visual_pooled = torch.mean(fused_embeddings, dim=1)
                
                loss = self.contrastive_loss(audio_pooled, visual_pooled)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "checkpoints",
        save_every: int = 5
    ):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            if self.val_loader:
                val_loss = self.validate()
                logger.info(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_loss:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        save_path / "best_model.pt",
                        epoch,
                        val_loss
                    )
                    logger.info(f"Saved best model with val loss: {val_loss:.4f}")
            
            # Regular checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(
                    save_path / f"checkpoint_epoch_{epoch}.pt",
                    epoch,
                    train_loss
                )
        
        logger.info("Training complete!")
    
    def save_checkpoint(self, path: Path, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': {
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers
            }
        }
        torch.save(checkpoint, path)


def main():
    """Main training script."""
    
    # Configuration
    config = {
        'audio_dim': 768,
        'visual_dim': 512,
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'max_length': 100,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create datasets
    train_dataset = MultimodalDataset(
        data_dir="data/train",
        audio_dim=config['audio_dim'],
        visual_dim=config['visual_dim'],
        max_length=config['max_length']
    )
    
    val_dataset = MultimodalDataset(
        data_dir="data/val",
        audio_dim=config['audio_dim'],
        visual_dim=config['visual_dim'],
        max_length=config['max_length']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = MultimodalTransformer(
        audio_input_dim=config['audio_dim'],
        visual_input_dim=config['visual_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = FusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        device=config['device']
    )
    
    # Train
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir="checkpoints/fusion",
        save_every=5
    )


if __name__ == "__main__":
    main()