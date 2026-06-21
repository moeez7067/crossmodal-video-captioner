"""
Training Evaluation Metrics for Fusion Model.

This module provides metrics specifically for evaluating fusion model training,
including reconstruction loss, contrastive loss, and alignment metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """
    Metrics calculator for fusion model training.
    
    Computes:
    - Reconstruction loss (MSE, MAE)
    - Contrastive loss
    - Alignment metrics (cosine similarity, correlation)
    - Feature statistics
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.metrics = {
            'mse_loss': [],
            'mae_loss': [],
            'contrastive_loss': [],
            'cosine_similarity': [],
            'correlation': [],
            'audio_norm': [],
            'visual_norm': [],
            'fused_norm': []
        }
    
    def compute_reconstruction_loss(
        self,
        fused: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction losses.
        
        Args:
            fused: Fused embeddings [batch, seq_len, dim]
            audio: Original audio embeddings [batch, seq_len, dim]
            visual: Original visual embeddings [batch, seq_len, dim]
            reduction: 'mean' or 'none'
            
        Returns:
            Dictionary with MSE and MAE losses
        """
        # Mean pool to get global representations
        fused_pooled = torch.mean(fused, dim=1)  # [batch, dim]
        audio_pooled = torch.mean(audio, dim=1)  # [batch, dim]
        visual_pooled = torch.mean(visual, dim=1)  # [batch, dim]
        
        # Combine audio and visual (simple addition for reconstruction target)
        target = (audio_pooled + visual_pooled) / 2
        
        # MSE loss
        mse_loss = F.mse_loss(fused_pooled, target, reduction=reduction)
        if reduction == 'mean':
            mse_loss = mse_loss.item()
        
        # MAE loss
        mae_loss = F.l1_loss(fused_pooled, target, reduction=reduction)
        if reduction == 'mean':
            mae_loss = mae_loss.item()
        
        return {
            'mse_loss': mse_loss,
            'mae_loss': mae_loss
        }
    
    def compute_contrastive_loss(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        temperature: float = 0.07,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute contrastive loss for audio-visual alignment.
        
        Args:
            audio: Audio embeddings [batch, dim] or [batch, seq_len, dim]
            visual: Visual embeddings [batch, dim] or [batch, seq_len, dim]
            temperature: Temperature parameter
            reduction: 'mean' or 'none'
            
        Returns:
            Contrastive loss
        """
        # Handle sequence dimension
        if audio.dim() == 3:
            audio = torch.mean(audio, dim=1)  # [batch, dim]
        if visual.dim() == 3:
            visual = torch.mean(visual, dim=1)  # [batch, dim]
        
        # Normalize
        audio_norm = F.normalize(audio, dim=-1)
        visual_norm = F.normalize(visual, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(audio_norm, visual_norm.T) / temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = audio_norm.shape[0]
        labels = torch.arange(batch_size, device=audio.device)
        
        # Cross-entropy loss for both directions
        loss_audio = F.cross_entropy(similarity, labels, reduction=reduction)
        loss_visual = F.cross_entropy(similarity.T, labels, reduction=reduction)
        
        loss = (loss_audio + loss_visual) / 2
        
        if reduction == 'mean':
            loss = loss.item()
        
        return loss
    
    def compute_alignment_metrics(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        fused: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute alignment metrics between modalities.
        
        Args:
            audio: Audio embeddings [batch, seq_len, dim]
            visual: Visual embeddings [batch, seq_len, dim]
            fused: Optional fused embeddings [batch, seq_len, dim]
            
        Returns:
            Dictionary with alignment metrics
        """
        # Mean pool to get global representations
        audio_pooled = torch.mean(audio, dim=1)  # [batch, dim]
        visual_pooled = torch.mean(visual, dim=1)  # [batch, dim]
        
        # Cosine similarity
        audio_norm = F.normalize(audio_pooled, dim=-1)
        visual_norm = F.normalize(visual_pooled, dim=-1)
        cosine_sim = torch.mean(torch.sum(audio_norm * visual_norm, dim=-1)).item()
        
        # Correlation (Pearson correlation)
        # Flatten and compute correlation
        audio_flat = audio_pooled.flatten()
        visual_flat = visual_pooled.flatten()
        
        audio_centered = audio_flat - torch.mean(audio_flat)
        visual_centered = visual_flat - torch.mean(visual_flat)
        
        numerator = torch.sum(audio_centered * visual_centered)
        denominator = torch.sqrt(torch.sum(audio_centered ** 2) * torch.sum(visual_centered ** 2))
        correlation = (numerator / (denominator + 1e-8)).item()
        
        metrics = {
            'cosine_similarity': cosine_sim,
            'correlation': correlation
        }
        
        # Add fused metrics if provided
        if fused is not None:
            fused_pooled = torch.mean(fused, dim=1)
            fused_norm = F.normalize(fused_pooled, dim=-1)
            
            # Cosine similarity with fused
            audio_fused_sim = torch.mean(torch.sum(audio_norm * fused_norm, dim=-1)).item()
            visual_fused_sim = torch.mean(torch.sum(visual_norm * fused_norm, dim=-1)).item()
            
            metrics['audio_fused_similarity'] = audio_fused_sim
            metrics['visual_fused_similarity'] = visual_fused_sim
        
        return metrics
    
    def compute_feature_statistics(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        fused: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute feature statistics.
        
        Args:
            audio: Audio embeddings [batch, seq_len, dim]
            visual: Visual embeddings [batch, seq_len, dim]
            fused: Optional fused embeddings [batch, seq_len, dim]
            
        Returns:
            Dictionary with feature statistics
        """
        stats = {}
        
        # L2 norms
        audio_norm = torch.norm(audio, dim=-1).mean().item()
        visual_norm = torch.norm(visual, dim=-1).mean().item()
        
        stats['audio_norm'] = audio_norm
        stats['visual_norm'] = visual_norm
        
        if fused is not None:
            fused_norm = torch.norm(fused, dim=-1).mean().item()
            stats['fused_norm'] = fused_norm
        
        # Mean and std
        stats['audio_mean'] = audio.mean().item()
        stats['audio_std'] = audio.std().item()
        stats['visual_mean'] = visual.mean().item()
        stats['visual_std'] = visual.std().item()
        
        if fused is not None:
            stats['fused_mean'] = fused.mean().item()
            stats['fused_std'] = fused.std().item()
        
        return stats
    
    def update(
        self,
        fused: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor
    ):
        """
        Update metrics with a batch.
        
        Args:
            fused: Fused embeddings [batch, seq_len, dim]
            audio: Audio embeddings [batch, seq_len, dim]
            visual: Visual embeddings [batch, seq_len, dim]
        """
        # Reconstruction losses
        recon_losses = self.compute_reconstruction_loss(fused, audio, visual)
        self.metrics['mse_loss'].append(recon_losses['mse_loss'])
        self.metrics['mae_loss'].append(recon_losses['mae_loss'])
        
        # Contrastive loss
        contrastive_loss = self.compute_contrastive_loss(audio, visual)
        self.metrics['contrastive_loss'].append(contrastive_loss)
        
        # Alignment metrics
        alignment = self.compute_alignment_metrics(audio, visual, fused)
        self.metrics['cosine_similarity'].append(alignment['cosine_similarity'])
        self.metrics['correlation'].append(alignment['correlation'])
        
        # Feature statistics
        stats = self.compute_feature_statistics(audio, visual, fused)
        self.metrics['audio_norm'].append(stats['audio_norm'])
        self.metrics['visual_norm'].append(stats['visual_norm'])
        if 'fused_norm' in stats:
            self.metrics['fused_norm'].append(stats['fused_norm'])
    
    def compute_averages(self) -> Dict[str, float]:
        """
        Compute average metrics across all batches.
        
        Returns:
            Dictionary with average metric values
        """
        averages = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                averages[key] = np.mean(values)
            else:
                averages[key] = 0.0
        
        return averages
    
    def get_summary(self) -> str:
        """
        Get formatted summary of metrics.
        
        Returns:
            Formatted string with metric summary
        """
        averages = self.compute_averages()
        
        summary = "Training Metrics Summary:\n"
        summary += f"  MSE Loss: {averages['mse_loss']:.4f}\n"
        summary += f"  MAE Loss: {averages['mae_loss']:.4f}\n"
        summary += f"  Contrastive Loss: {averages['contrastive_loss']:.4f}\n"
        summary += f"  Cosine Similarity: {averages['cosine_similarity']:.4f}\n"
        summary += f"  Correlation: {averages['correlation']:.4f}\n"
        summary += f"  Audio Norm: {averages['audio_norm']:.4f}\n"
        summary += f"  Visual Norm: {averages['visual_norm']:.4f}\n"
        if 'fused_norm' in averages:
            summary += f"  Fused Norm: {averages['fused_norm']:.4f}\n"
        
        return summary
