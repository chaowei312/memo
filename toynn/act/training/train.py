"""
ACT-UNet training with RL-based adaptive computation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional, Dict, Any


class ACTTrainer:
    """Trainer for ACT-UNet with actor-critic updates"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize losses
        from .losses import SegmentationLoss, ACTLoss
        self.seg_loss = SegmentationLoss()
        self.act_loss = ACTLoss()
        
        # Separate optimizers for actor and critic
        self.optimizer_main = optim.AdamW(
            [p for n, p in model.named_parameters() if 'bottleneck' not in n],
            lr=1e-3, weight_decay=1e-4
        )
        
        # Actor (objective_head) optimizer
        self.optimizer_actor = optim.Adam(
            model.bottleneck.objective_head.parameters(),
            lr=1e-4
        )
        
        # Critic (subjective_head) optimizer  
        self.optimizer_critic = optim.Adam(
            model.bottleneck.subjective_head.parameters(),
            lr=1e-3
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.update_critic = True  # Start with critic updates
        self.alternating_freq = 100  # Steps between actor/critic switches
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(self, epoch: int):
        """Train for one epoch with RL updates"""
        self.model.train()
        epoch_metrics = {
            'seg_loss': [], 'act_loss': [], 'optimal_k': [],
            'policy_k': [], 'agreement': [], 'td_error': [], 'pg_loss': []
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass with ACT trajectory
            with autocast():
                predictions, act_info = self.model(images, return_act_info=True)
                
                # Segmentation loss  
                seg_loss = self.seg_loss(predictions, masks)
                
                # ACT loss (RL component)
                act_loss_value, act_metrics = self.act_loss(
                    act_info, masks, self.seg_loss, 
                    update_critic=self.update_critic
                )
                
                # Combined loss
                total_loss = seg_loss + 0.1 * act_loss_value
            
            # Backward pass
            self.optimizer_main.zero_grad()
            if self.update_critic:
                self.optimizer_critic.zero_grad()
            else:
                self.optimizer_actor.zero_grad()
            
            self.scaler.scale(total_loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer_main)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer steps
            self.scaler.step(self.optimizer_main)
            if self.update_critic:
                self.scaler.step(self.optimizer_critic)
            else:
                self.scaler.step(self.optimizer_actor)
            
            self.scaler.update()
            
            # Update metrics
            epoch_metrics['seg_loss'].append(seg_loss.item())
            epoch_metrics['act_loss'].append(act_loss_value.item())
            for k, v in act_metrics.items():
                if k in epoch_metrics:
                    epoch_metrics[k].append(v)
            
            # Alternate between actor and critic updates
            self.global_step += 1
            if self.global_step % self.alternating_freq == 0:
                self.update_critic = not self.update_critic
                mode = "Critic" if self.update_critic else "Actor"
                print(f"\nSwitching to {mode} updates")
            
            # Update progress bar
            pbar.set_postfix({
                'seg_loss': f"{seg_loss.item():.4f}",
                'k*': f"{act_metrics['optimal_k']:.1f}",
                'k_policy': f"{act_metrics['policy_k']:.1f}",
                'agree': f"{act_metrics['agreement']:.2%}"
            })
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items() if v}
        return avg_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model performance"""
        self.model.eval()
        val_metrics = {
            'seg_loss': [], 'iou': [], 'dice': [],
            'avg_iterations': [], 'computation_saved': []
        }
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            predictions, act_info = self.model(images, return_act_info=True)
            
            # Compute losses
            seg_loss = self.seg_loss(predictions, masks)
            val_metrics['seg_loss'].append(seg_loss.item())
            
            # Compute IoU and Dice
            pred_binary = (torch.sigmoid(predictions) > 0.5).float()
            intersection = (pred_binary * masks).sum(dim=(2, 3))
            union = (pred_binary + masks).clamp(0, 1).sum(dim=(2, 3))
            iou = (intersection / (union + 1e-6)).mean()
            dice = (2 * intersection / (pred_binary.sum(dim=(2, 3)) + 
                                       masks.sum(dim=(2, 3)) + 1e-6)).mean()
            
            val_metrics['iou'].append(iou.item())
            val_metrics['dice'].append(dice.item())
            
            # ACT metrics
            avg_iter = act_info['halt_iterations'].float().mean()
            val_metrics['avg_iterations'].append(avg_iter.item())
            
            # Computation saved vs always using max iterations
            saved = 1 - (avg_iter / self.model.max_iterations)
            val_metrics['computation_saved'].append(saved.item())
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        # Save best model
        if avg_metrics['seg_loss'] < self.best_val_loss:
            self.best_val_loss = avg_metrics['seg_loss']
            self.save_checkpoint(epoch, is_best=True)
        
        return avg_metrics
    
    def train(
        self,
        num_epochs: int,
        save_freq: int = 5,
        val_freq: int = 1
    ):
        """Full training loop"""
        
        for epoch in range(self.epoch, self.epoch + num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.epoch + num_epochs}")
            print(f"{'='*50}")
            
            # Training
            train_metrics = self.train_epoch(epoch)
            self.train_metrics.append(train_metrics)
            
            print(f"\nTrain Metrics:")
            for k, v in train_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Validation
            if (epoch + 1) % val_freq == 0:
                val_metrics = self.validate(epoch)
                self.val_metrics.append(val_metrics)
                
                print(f"\nValidation Metrics:")
                for k, v in val_metrics.items():
                    print(f"  {k}: {v:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch)
            
            self.epoch = epoch + 1
        
        print("\nTraining complete!")
        return self.train_metrics, self.val_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_main': self.optimizer_main.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'update_critic': self.update_critic,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer_main.load_state_dict(checkpoint['optimizer_main'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.update_critic = checkpoint['update_critic']
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
