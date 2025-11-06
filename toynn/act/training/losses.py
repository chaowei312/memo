"""
Loss functions for ACT-UNet training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities [B, C, H, W]
            target: Ground truth [B, C, H, W] or [B, 1, H, W]
        """
        if pred.shape[1] > 1:  # Multi-class
            pred = F.softmax(pred, dim=1)
        else:  # Binary
            pred = torch.sigmoid(pred)
        
        # Flatten spatial dimensions
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target = target.reshape(target.shape[0], target.shape[1], -1)
        
        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    """Combined loss for segmentation tasks"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions [B, C, H, W]
            target: Ground truth masks [B, 1, H, W] for binary
        """
        # Dice loss
        dice = self.dice_loss(pred, target)
        
        # BCE loss
        if pred.shape[1] == 1:  # Binary segmentation
            bce = F.binary_cross_entropy_with_logits(pred, target)
        else:  # Multi-class
            target_long = target.squeeze(1).long()
            bce = F.cross_entropy(pred, target_long)
        
        return self.dice_weight * dice + self.bce_weight * bce


class ACTLoss(nn.Module):
    """
    Reinforcement Learning loss for ACT mechanism
    Implements TD error and policy gradient loss
    """
    
    def __init__(self, efficiency_penalty=0.01):
        super().__init__()
        self.efficiency_penalty = efficiency_penalty
        
    def compute_optimal_depth(self, trajectory, target, loss_fn):
        """
        Find optimal stopping point k* by evaluating task loss at each step
        
        Args:
            trajectory: States at each iteration [K, B, C, H, W]
            target: Ground truth segmentation [B, 1, H, W]
            loss_fn: Segmentation loss function
            
        Returns:
            optimal_k: Optimal stopping iteration for each sample [B]
            best_losses: Loss at optimal iteration [B]
        """
        K, B = trajectory.shape[:2]
        device = trajectory.device
        
        # Evaluate loss at each iteration
        losses = []
        for k in range(K):
            with torch.no_grad():
                # Get predictions at iteration k
                pred_k = trajectory[k]
                # Compute task loss
                loss_k = []
                for b in range(B):
                    loss_b = loss_fn(pred_k[b:b+1], target[b:b+1])
                    loss_k.append(loss_b.item())
                losses.append(torch.tensor(loss_k, device=device))
        
        losses = torch.stack(losses, dim=0)  # [K, B]
        
        # Add efficiency penalty for later iterations
        iteration_penalty = torch.arange(K, device=device).unsqueeze(1) * self.efficiency_penalty
        losses = losses + iteration_penalty
        
        # Find minimum loss iteration for each sample
        best_losses, optimal_k = losses.min(dim=0)
        
        return optimal_k, best_losses
    
    def td_error(self, subjective_values, objective_sums, optimal_k):
        """
        Compute temporal difference error for critic
        
        Args:
            subjective_values: o(x^n) at each step [K-1, B]
            objective_sums: Cumulative S_n at each step [K-1, B]  
            optimal_k: Optimal stopping point [B]
            
        Returns:
            td_loss: TD error for value function
        """
        K_minus_1, B = subjective_values.shape
        device = subjective_values.device
        
        # Create mask for iterations >= k*
        iteration_indices = torch.arange(K_minus_1, device=device).unsqueeze(1)
        optimal_k_expanded = optimal_k.unsqueeze(0)
        valid_mask = iteration_indices >= optimal_k_expanded  # [K-1, B]
        
        # Compute MSE only for valid iterations
        if valid_mask.any():
            td_error = F.mse_loss(
                subjective_values[valid_mask],
                objective_sums[valid_mask].detach(),  # Detach target
                reduction='mean'
            )
        else:
            td_error = torch.tensor(0.0, device=device)
        
        return td_error
    
    def policy_gradient_loss(self, contributions, subjective_values, policy_actions, optimal_k):
        """
        Compute policy gradient loss for actor
        
        Args:
            contributions: τ_i at each step [K-1, B]
            subjective_values: o(x^n) at each step [K-1, B]
            policy_actions: Where policy decided to stop [B]
            optimal_k: Optimal stopping point [B]
            
        Returns:
            pg_loss: Policy gradient loss
        """
        B = policy_actions.shape[0]
        device = contributions.device
        
        # Only update when policy disagrees with optimal
        disagreement_mask = policy_actions != optimal_k
        
        if disagreement_mask.any():
            # Advantage: positive if should have continued, negative if should have stopped
            advantages = (optimal_k - policy_actions).float()
            
            # Policy gradient loss (simplified)
            # Encourage higher contributions when should continue longer
            # Encourage lower contributions when should stop earlier
            pg_loss = -(contributions.mean(dim=0)[disagreement_mask] * 
                       advantages[disagreement_mask].detach()).mean()
        else:
            pg_loss = torch.tensor(0.0, device=device)
        
        return pg_loss
    
    def forward(self, act_info, target, seg_loss_fn, update_critic=True):
        """
        Compute ACT losses for RL training
        
        Args:
            act_info: Dictionary with trajectory information
            target: Ground truth segmentation
            seg_loss_fn: Segmentation loss function  
            update_critic: If True, update critic; else update actor
            
        Returns:
            loss: TD error (if update_critic) or policy gradient loss
            metrics: Dictionary with training metrics
        """
        trajectory = act_info['trajectory']
        contributions = act_info['contributions']
        thresholds = act_info['thresholds']
        halt_iterations = act_info['halt_iterations']
        cumulative_contributions = act_info['cumulative_contributions']
        
        # Find optimal stopping point k*
        optimal_k, best_losses = self.compute_optimal_depth(
            trajectory, target, seg_loss_fn
        )
        
        # Compute cumulative sums S_n
        if contributions is not None:
            objective_sums = contributions.cumsum(dim=0)
        else:
            objective_sums = torch.zeros_like(thresholds)
        
        metrics = {
            'optimal_k': optimal_k.float().mean().item(),
            'policy_k': halt_iterations.float().mean().item(),
            'agreement': (halt_iterations == optimal_k).float().mean().item(),
            'best_loss': best_losses.mean().item()
        }
        
        if update_critic:
            # Critic update: minimize TD error
            loss = self.td_error(thresholds, objective_sums, optimal_k)
            metrics['td_error'] = loss.item()
        else:
            # Actor update: policy gradient
            loss = self.policy_gradient_loss(
                contributions, thresholds, halt_iterations, optimal_k
            )
            metrics['pg_loss'] = loss.item()
        
        return loss, metrics
