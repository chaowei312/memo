"""
Adaptive Computation Time (ACT) Bottleneck for U-Net
Implements RL-based adaptive depth computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RecurrentBlock(nn.Module):
    """Shared-weight recurrent transformation for iterative refinement"""
    
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        
        # Recurrent transformation f: x^{n+1} = f(x^n)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(8, channels),
        )
        
        # Residual connection
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_prev, x_init):
        """
        Args:
            x_prev: Previous state x^{n-1}
            x_init: Initial input x^0 for skip connection
        Returns:
            x_next: Next state x^n
        """
        transformed = self.transform(x_prev)
        
        # Gated residual with initial state
        gate = self.gate(torch.cat([transformed, x_init], dim=1))
        x_next = gate * transformed + (1 - gate) * x_prev
        
        return x_next


class ObjectiveHead(nn.Module):
    """Estimates incremental contribution τ_i at each step"""
    
    def __init__(self, channels):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Network g for computing contribution from state difference
        self.contribution_net = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Softplus()  # Ensure positive output
        )
        
        self.eps = 1e-6
        
    def forward(self, x_curr, x_prev):
        """
        Compute objective contribution τ_i based on encoded CNN features
        Measures how much the encoded features changed in this iteration
        """
        # Compute feature difference
        diff = x_curr - x_prev
        
        # 1. Spatial change: L2 norm of feature changes across spatial dimensions
        # This captures how much the feature maps changed spatially
        spatial_change = torch.norm(diff, p=2, dim=1).mean(dim=(1, 2))  # [B]
        
        # 2. Feature-level change: Compare global feature representations
        # Use global average pooling to get feature vectors (no CLS tokens in CNN!)
        feat_prev = self.pool(x_prev).flatten(1)  # [B, C] - global feature vector
        feat_curr = self.pool(x_curr).flatten(1)  # [B, C] - global feature vector
        
        # Cosine similarity between feature vectors (normalized to [0, 1])
        feat_prev_norm = F.normalize(feat_prev, p=2, dim=1)
        feat_curr_norm = F.normalize(feat_curr, p=2, dim=1)
        cosine_sim = torch.sum(feat_prev_norm * feat_curr_norm, dim=1)  # [-1, 1]
        feature_change = 1.0 - (cosine_sim + 1.0) / 2.0  # Convert to change metric [0, 1]
        
        # 3. Learned contribution from the difference map
        g_diff = self.contribution_net(diff)
        g_diff = self.pool(g_diff).squeeze(-1)  # [B]
        
        # Combine: spatial change × feature dissimilarity × learned contribution
        tau = spatial_change * (1.0 + feature_change) * g_diff
        
        return tau  # Shape: [batch_size]


class SubjectiveHead(nn.Module):
    """Estimates required computation quality threshold o(x^n)"""
    
    def __init__(self, channels):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Network o for estimating state quality threshold
        self.threshold_net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Softplus(beta=1.0),  # Positive output
        )
        
        # Learnable bias for minimum threshold
        self.min_threshold = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        """
        Compute subjective threshold o(x^n) based on encoded CNN features
        Higher threshold = current encoded features need more refinement
        Lower threshold = features are sufficiently refined
        """
        # Evaluate quality of encoded features
        threshold = self.threshold_net(x)  # [B, 1, H, W]
        
        # Global pooling to get overall quality estimate
        threshold = self.pool(threshold).squeeze(-1)  # [B]
        
        # Ensure minimum threshold (always require some computation)
        threshold = threshold + F.softplus(self.min_threshold)
        
        return threshold  # Shape: [batch_size]


class ACTBottleneck(nn.Module):
    """
    ACT-enhanced bottleneck with adaptive computational depth
    
    Operates on encoded CNN features from U-Net encoder (not raw images!)
    Input: Encoded features of shape [B, C, H/16, W/16] where C=512 or 1024
    Output: Refined features after adaptive iterations
    """
    
    def __init__(self, channels, max_iterations=5, base_channels=None):
        super().__init__()
        self.channels = channels
        self.max_iterations = max_iterations
        
        if base_channels is None:
            base_channels = channels
            
        # Initial projection if needed
        self.input_proj = nn.Conv2d(base_channels, channels, 1) if base_channels != channels else nn.Identity()
        
        # Recurrent processing block (shared weights)
        self.recurrent_block = RecurrentBlock(channels)
        
        # Value function heads for halting policy
        self.objective_head = ObjectiveHead(channels)
        self.subjective_head = SubjectiveHead(channels)
        
        # Output projection
        self.output_proj = nn.Conv2d(channels, base_channels, 1) if base_channels != channels else nn.Identity()
        
    def forward(self, x, return_trajectory=False):
        """
        Forward pass with adaptive computation
        
        Args:
            x: Input features [B, C, H, W]
            return_trajectory: If True, return full trajectory for training
            
        Returns:
            output: Processed features after adaptive iterations
            info: Dictionary with trajectory info (if return_trajectory=True)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Project input if needed
        x_init = self.input_proj(x)
        
        # Initialize trajectory storage
        trajectory = []
        contributions = []
        thresholds = []
        
        # Initialize state
        x_curr = x_init
        x_prev = x_init
        cumulative_contribution = torch.zeros(batch_size, device=device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        halt_iterations = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Iterate up to max_iterations
        for i in range(self.max_iterations):
            # Apply recurrent transformation
            x_next = self.recurrent_block(x_curr, x_init)
            
            if i > 0:  # Skip first iteration (no previous state difference)
                # Compute contribution and threshold
                tau_i = self.objective_head(x_next, x_curr)
                o_i = self.subjective_head(x_next)
                
                # Update cumulative contribution
                cumulative_contribution = cumulative_contribution + tau_i * (~halted).float()
                
                # Check halting condition: S_n > o(x^n)
                should_halt = cumulative_contribution > o_i
                newly_halted = should_halt & ~halted
                
                # Update halting status
                halted = halted | newly_halted
                halt_iterations[newly_halted] = i
                
                # Store for training
                if return_trajectory:
                    contributions.append(tau_i)
                    thresholds.append(o_i)
            
            # Store state in trajectory
            if return_trajectory:
                trajectory.append(x_next)
            
            # Update states
            x_prev = x_curr
            x_curr = x_next
            
            # Early exit if all samples have halted (during inference)
            if not return_trajectory and halted.all():
                break
        
        # Set remaining halt iterations for samples that never halted
        halt_iterations[~halted] = self.max_iterations - 1
        
        # Select output based on halt iterations
        if return_trajectory:
            # Stack trajectory: [max_iterations, B, C, H, W]
            trajectory = torch.stack(trajectory, dim=0)
            
            # Gather outputs at halt iterations
            batch_indices = torch.arange(batch_size, device=device)
            output = trajectory[halt_iterations, batch_indices]
            
            info = {
                'trajectory': trajectory,
                'contributions': torch.stack(contributions) if contributions else None,
                'thresholds': torch.stack(thresholds) if thresholds else None,
                'halt_iterations': halt_iterations,
                'cumulative_contributions': cumulative_contribution
            }
        else:
            output = x_curr
            info = {
                'halt_iterations': halt_iterations,
                'cumulative_contributions': cumulative_contribution
            }
        
        # Project output back if needed
        output = self.output_proj(output)
        
        return output, info
