"""
Enhanced Training Script for Fusion Model.

This module provides an enhanced training script with:
- Better loss functions
- Training metrics tracking
- Checkpointing and resuming
- Learning rate scheduling
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm
import json
import time

from src.fusion.multimodal_transformer import MultimodalTransformer
from src.fusion.training_metrics import TrainingMetrics

logger = logging.getLogger(__name__)


class EnhancedFusionTrainer:
    """
    Enhanced trainer for multimodal fusion model.
    
    Features:
    - Multiple loss functions (reconstruction, contrastive, combined)
    - Training metrics tracking
    - Checkpointing and resuming
    - Learning rate scheduling
    - Early stopping
    - TensorBoard logging (optional)
    """
    
    def __init__(
        self,
        model: MultimodalTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cuda",
        loss_type: str = "combined",
        use_tensorboard: bool = False
    ):
        """
        Initialize enhanced trainer.
        
        Args:
            model: Multimodal transformer model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            device: Device to train on
            loss_type: Type of loss ('reconstruction', 'contrastive', 'combined')
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.loss_type = loss_type
        self.use_tensorboard = use_tensorboard
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Metrics
        self.train_metrics = TrainingMetrics()
        self.val_metrics = TrainingMetrics()
        
        # Training state
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.patience_counter = 0
        self.max_patience = 10
        
        # TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter()
            except ImportError:
                logger.warning("TensorBoard not available. Continuing without it.")
                self.use_tensorboard = False
                self.writer = None
        else:
            self.writer = None
    
    def compute_loss(
        self,
        fused: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss based on loss_type.
        
        Args:
            fused: Fused embeddings [batch, seq_len, dim]
            audio: Audio embeddings [batch, seq_len, dim]
            visual: Visual embeddings [batch, seq_len, dim]
            
        Returns:
            Total loss and loss components dictionary
        """
        loss_components = {}
        
        if self.loss_type == 'reconstruction':
            # Mean pool for global representation
            fused_pooled = torch.mean(fused, dim=1)
            audio_pooled = torch.mean(audio, dim=1)
            visual_pooled = torch.mean(visual, dim=1)
            
            # Target: average of audio and visual
            target = (audio_pooled + visual_pooled) / 2
            
            loss = self.mse_loss(fused_pooled, target)
            loss_components['reconstruction'] = loss.item()
            
        elif self.loss_type == 'contrastive':
            loss = self._contrastive_loss(audio, visual)
            loss_components['contrastive'] = loss.item()
            
        elif self.loss_type == 'combined':
            # Reconstruction component
            fused_pooled = torch.mean(fused, dim=1)
            audio_pooled = torch.mean(audio, dim=1)
            visual_pooled = torch.mean(visual, dim=1)
            target = (audio_pooled + visual_pooled) / 2
            recon_loss = self.mse_loss(fused_pooled, target)
            
            # Contrastive component
            contrastive_loss = self._contrastive_loss(audio, visual)
            
            # Combined (weighted)
            loss = 0.7 * recon_loss + 0.3 * contrastive_loss
            
            loss_components['reconstruction'] = recon_loss.item()
            loss_components['contrastive'] = contrastive_loss.item()
            loss_components['total'] = loss.item()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss, loss_components
    
    def _contrastive_loss(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """Compute contrastive loss."""
        # Mean pool if needed
        if audio.dim() == 3:
            audio = torch.mean(audio, dim=1)
        if visual.dim() == 3:
            visual = torch.mean(visual, dim=1)
        
        # Normalize
        audio_norm = nn.functional.normalize(audio, dim=-1)
        visual_norm = nn.functional.normalize(visual, dim=-1)
        
        # Similarity matrix
        similarity = torch.matmul(audio_norm, visual_norm.T) / temperature
        
        # Labels
        batch_size = audio_norm.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy
        loss_audio = nn.functional.cross_entropy(similarity, labels)
        loss_visual = nn.functional.cross_entropy(similarity.T, labels)
        
        return (loss_audio + loss_visual) / 2
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio = batch['audio'].to(self.device)
            visual = batch['visual'].to(self.device)
            
            # Forward pass
            fused, _ = self.model(audio, visual, return_attention=False)
            
            # Compute loss
            loss, loss_components = self.compute_loss(fused, audio, visual)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(fused, audio, visual)
            
            # Accumulate
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                **{k: f'{v:.4f}' for k, v in loss_components.items()}
            })
            
            # TensorBoard logging
            if self.writer and batch_idx % 100 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                for k, v in loss_components.items():
                    self.writer.add_scalar(f'Train/Loss_{k}', v, global_step)
        
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute_averages()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            
            for batch_idx, batch in enumerate(pbar):
                audio = batch['audio'].to(self.device)
                visual = batch['visual'].to(self.device)
                
                fused, _ = self.model(audio, visual, return_attention=False)
                
                loss, loss_components = self.compute_loss(fused, audio, visual)
                
                self.val_metrics.update(fused, audio, visual)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute_averages()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "checkpoints/fusion",
        save_every: int = 5,
        early_stopping: bool = True
    ):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            early_stopping: Whether to use early stopping
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Loss type: {self.loss_type}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_metrics'].append(train_metrics)
            
            logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            if self.val_loader:
                val_metrics = self.validate(epoch)
                training_history['val_loss'].append(val_metrics['loss'])
                training_history['val_metrics'].append(val_metrics)
                
                logger.info(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_metrics['loss']:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self.save_checkpoint(
                        save_path / "best_model.pt",
                        epoch,
                        val_metrics['loss'],
                        is_best=True
                    )
                    logger.info(f"Saved best model with val loss: {val_metrics['loss']:.4f}")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if early_stopping and self.patience_counter >= self.max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                    for k, v in val_metrics.items():
                        if k != 'loss':
                            self.writer.add_scalar(f'Val/{k}', v, epoch)
            
            # Regular checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(
                    save_path / f"checkpoint_epoch_{epoch}.pt",
                    epoch,
                    train_metrics['loss'],
                    is_best=False
                )
        
        # Save training history
        history_path = save_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training complete! Time: {elapsed_time/60:.2f} minutes")
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_val_loss': self.best_val_loss,
            'config': {
                'hidden_dim': self.model.hidden_dim,
                'num_layers': len(self.model.layers),
                'num_heads': self.model.layers[0].self_attention.num_heads if len(self.model.layers) > 0 else 8
            }
        }
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
