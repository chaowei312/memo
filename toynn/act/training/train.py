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
        
        # Separate optimizers for actor and critic with safer learning rates
        self.optimizer_main = optim.AdamW(
            [p for n, p in model.named_parameters() if 'bottleneck' not in n],
            lr=2e-4, weight_decay=1e-5  # Safer LR to prevent explosions
        )
        
        # Actor (objective_head) optimizer
        self.optimizer_actor = optim.Adam(
            model.bottleneck.objective_head.parameters(),
            lr=5e-5  # Conservative LR for RL component
        )
        
        # Critic (subjective_head) optimizer  
        self.optimizer_critic = optim.Adam(
            model.bottleneck.subjective_head.parameters(),
            lr=1e-4  # Moderate LR
        )
        
        # Store initial learning rates for warmup
        for optimizer in [self.optimizer_main, self.optimizer_actor, self.optimizer_critic]:
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = param_group['lr']
        
        # Mixed precision training with conservative initial scale
        self.scaler = GradScaler(init_scale=2**8)  # Start with smaller scale to prevent overflow
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.warmup_steps = 50  # Shorter warmup to start learning sooner
        self.update_critic = True  # Start with critic updates
        self.alternating_freq = 100  # Steps between actor/critic switches
        self.steps_since_switch = 0  # Track steps since last switch for stability
        self.gradient_skip_count = 0  # Track consecutive gradient skips
        self.max_consecutive_skips = 20  # Reduce LR after this many consecutive skips
        
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
            
            # Check for NaN in inputs
            if torch.isnan(images).any() or torch.isnan(masks).any():
                print(f"Warning: NaN in input data at batch {batch_idx}, skipping...")
                continue
            
            # Forward pass with ACT trajectory (disable autocast briefly for stability)
            use_amp = self.global_step > 50  # Start AMP sooner to match warmup
            with autocast(enabled=use_amp):
                predictions, act_info = self.model(images, return_act_info=True)
                
                # Check for NaN in predictions
                if torch.isnan(predictions).any():
                    print(f"Warning: NaN in predictions at batch {batch_idx}")
                    # Try to recover by reinitializing last layer
                    with torch.no_grad():
                        self.model.outc.weight.data.normal_(0, 0.02)
                        if self.model.outc.bias is not None:
                            self.model.outc.bias.data.zero_()
                    continue
                
                # Segmentation loss  
                seg_loss = self.seg_loss(predictions, masks)
                
                # ACT loss (RL component) with safety check
                try:
                    act_loss_value, act_metrics = self.act_loss(
                        act_info, masks, self.seg_loss, 
                        update_critic=self.update_critic
                    )
                    # Check for NaN in ACT loss
                    if torch.isnan(act_loss_value) or torch.isinf(act_loss_value):
                        print(f"Warning: NaN/Inf in ACT loss at batch {batch_idx}, using seg_loss only")
                        act_loss_value = torch.tensor(0.0, device=self.device, requires_grad=True)
                        act_metrics = {'optimal_k': 0, 'policy_k': 1, 'agreement': 0, 'td_error': 0, 'pg_loss': 0}
                except Exception as e:
                    print(f"Warning: ACT loss computation failed at batch {batch_idx}: {e}")
                    act_loss_value = torch.tensor(0.0, device=self.device, requires_grad=True)
                    act_metrics = {'optimal_k': 0, 'policy_k': 1, 'agreement': 0, 'td_error': 0, 'pg_loss': 0}
                
                # Combined loss with very small ACT weight to prevent instability
                # Gradually increase ACT weight over time
                if self.global_step < self.warmup_steps:
                    act_weight = 0.001  # Very small during warmup
                elif self.global_step < 200:
                    act_weight = 0.01   # Small after warmup
                elif self.global_step < 500:
                    act_weight = 0.05   # Medium
                else:
                    act_weight = 0.1    # Full weight only after substantial training
                    
                total_loss = seg_loss + act_weight * act_loss_value
            
            # Backward pass
            self.optimizer_main.zero_grad()
            aux_optimizer = self.optimizer_critic if self.update_critic else self.optimizer_actor
            aux_optimizer.zero_grad()
            
            # Scale loss only if using mixed precision
            if use_amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # Check if auxiliary optimizer has gradients
            aux_has_grad = any(
                param.grad is not None
                for group in aux_optimizer.param_groups
                for param in group["params"]
            )
            
            # Unscale gradients for both optimizers before clipping (only if using AMP)
            if use_amp:
                self.scaler.unscale_(self.optimizer_main)
                if aux_has_grad:
                    self.scaler.unscale_(aux_optimizer)
            
            # Debug which part has exploding gradients and clip separately
            main_params = [p for n, p in self.model.named_parameters() if 'bottleneck' not in n and p.grad is not None]
            bottleneck_params = [p for n, p in self.model.named_parameters() if 'bottleneck' in n and p.grad is not None]
            
            if main_params:
                main_grad_norm = torch.nn.utils.clip_grad_norm_(main_params, 0.5, error_if_nonfinite=False)
            else:
                main_grad_norm = 0.0
                
            if bottleneck_params:
                # Bottleneck (ACT) gets even stricter clipping since it's causing issues
                bottleneck_grad_norm = torch.nn.utils.clip_grad_norm_(bottleneck_params, 0.1, error_if_nonfinite=False)
            else:
                bottleneck_grad_norm = 0.0
            
            # Overall gradient clipping - more aggressive
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5, error_if_nonfinite=False)
            
            # Skip update if gradients are still too large after clipping
            threshold = 5.0  # Stricter threshold
            if grad_norm > threshold or torch.isnan(grad_norm) or torch.isinf(grad_norm):
                self.gradient_skip_count += 1
                print(f"Warning: Gradient explosion at batch {batch_idx} - Total: {grad_norm:.2f}, Main: {main_grad_norm:.2f}, Bottleneck: {bottleneck_grad_norm:.2f} (skip #{self.gradient_skip_count})")
                
                # Reduce learning rates if we're getting too many consecutive skips
                if self.gradient_skip_count >= self.max_consecutive_skips:
                    print(f"  Reducing learning rates after {self.gradient_skip_count} consecutive skips")
                    for param_group in self.optimizer_main.param_groups:
                        param_group['lr'] *= 0.5
                    for param_group in aux_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    self.gradient_skip_count = 0  # Reset counter
                
                # Reset optimizer states if gradients are extremely bad
                if grad_norm > 100.0 or torch.isnan(grad_norm):
                    print(f"  Resetting optimizer states due to extreme gradients")
                    # Clear all optimizer states to recover
                    self.optimizer_main.state = {}
                    aux_optimizer.state = {}
                    # Also reset learning rates to very conservative values
                    for param_group in self.optimizer_main.param_groups:
                        param_group['lr'] = 5e-5  # Very conservative
                    for param_group in aux_optimizer.param_groups:
                        param_group['lr'] = 1e-5  # Very conservative
                
                # Only update scaler if we were using AMP
                if use_amp:
                    self.scaler.update()  # Still need to update scaler state
                continue
            else:
                # Reset skip counter on successful update
                if self.gradient_skip_count > 0:
                    print(f"  Gradient back to normal after {self.gradient_skip_count} skips")
                self.gradient_skip_count = 0
            
            # Learning rate warmup
            if self.global_step < self.warmup_steps:
                warmup_factor = (self.global_step + 1) / self.warmup_steps
                for param_group in self.optimizer_main.param_groups:
                    param_group['lr'] = param_group.get('initial_lr', param_group['lr']) * warmup_factor
                for param_group in aux_optimizer.param_groups:
                    param_group['lr'] = param_group.get('initial_lr', param_group['lr']) * warmup_factor
            
            # Optimizer steps with NaN check
            old_params = {name: param.clone() for name, param in self.model.named_parameters()}
            
            if use_amp:
                self.scaler.step(self.optimizer_main)
                if aux_has_grad:
                    self.scaler.step(aux_optimizer)
                self.scaler.update()
            else:
                self.optimizer_main.step()
                if aux_has_grad:
                    aux_optimizer.step()
            
            # Check if any parameters became NaN and restore if needed
            has_nan = False
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"Warning: NaN/Inf in parameter {name} after update, restoring...")
                    param.data = old_params[name].data
                    has_nan = True
            
            if has_nan:
                print(f"Restored model parameters at batch {batch_idx} due to NaN/Inf")
                continue
            
            # Update metrics
            epoch_metrics['seg_loss'].append(seg_loss.item())
            epoch_metrics['act_loss'].append(act_loss_value.item())
            for k, v in act_metrics.items():
                if k in epoch_metrics:
                    epoch_metrics[k].append(v)
            
            # Alternate between actor and critic updates
            self.global_step += 1
            self.steps_since_switch += 1
            if self.global_step % self.alternating_freq == 0:
                self.update_critic = not self.update_critic
                mode = "Critic" if self.update_critic else "Actor"
                print(f"\nSwitching to {mode} updates")
                self.steps_since_switch = 0  # Reset counter
                # Reset optimizer state when switching to prevent momentum carryover
                if self.update_critic:
                    self.optimizer_critic.zero_grad()
                    for group in self.optimizer_critic.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                p.grad.data.zero_()
                else:
                    self.optimizer_actor.zero_grad()
                    for group in self.optimizer_actor.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                p.grad.data.zero_()
            
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
            if predictions.shape[1] == 1:
                probs = torch.sigmoid(predictions)
                pred_mask = (probs > 0.5).float()
                target_mask = masks.float()

                intersection = (pred_mask * target_mask).sum(dim=(2, 3))
                union = pred_mask.sum(dim=(2, 3)) + target_mask.sum(dim=(2, 3)) - intersection
                iou = (intersection / (union + 1e-6)).mean()

                dice = (2 * intersection / (
                    pred_mask.sum(dim=(2, 3)) + target_mask.sum(dim=(2, 3)) + 1e-6
                )).mean()
            else:
                num_classes = predictions.shape[1]
                pred_classes = predictions.argmax(dim=1)
                target_classes = masks.squeeze(1).long() if masks.shape[1] == 1 else masks.argmax(dim=1)

                pred_one_hot = torch.nn.functional.one_hot(pred_classes, num_classes).permute(0, 3, 1, 2).float()
                target_one_hot = torch.nn.functional.one_hot(target_classes, num_classes).permute(0, 3, 1, 2).float()

                intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
                union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection
                iou = (intersection / (union + 1e-6)).mean()

                dice = (2 * intersection / (
                    pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) + 1e-6
                )).mean()

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
