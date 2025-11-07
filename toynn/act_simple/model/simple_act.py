"""
Simple ACT model for MNIST - Testing ground for ACT theory
Based on the RL formulation in act.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class RecurrentRefinementBlock(nn.Module):
    """Simple recurrent block: h_{n+1} = h_n + FFN(h_n)"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1),
        )
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Residual update: h_next = h + FFN(h)"""
        return h + self.ffn(h)


class SubjectiveHead(nn.Module):
    """Computes contribution τ_i based on state change"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Bounded output [0, 1]
        )
        
    def forward(self, h_curr: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Compute contribution from state difference"""
        diff = h_curr - h_prev
        combined = torch.cat([h_curr, diff], dim=-1)
        contribution = self.net(combined) * 0.3  # Scale to [0, 0.3]
        return contribution.squeeze(-1)


class ObjectiveHead(nn.Module):
    """Computes threshold o(h^n) - how much computation needed"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Bounded output [0, 1]
        )
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute threshold from current state"""
        threshold = self.net(h)
        threshold = 0.1 + threshold * 0.5  # Scale to [0.1, 0.6]
        return threshold.squeeze(-1)


class SimpleACTNet(nn.Module):
    """
    Simple ACT implementation for MNIST digit classification
    Tests the core ACT mechanism without complex architecture
    """
    
    def __init__(
        self,
        input_dim: int = 784,  # 28*28 flattened
        hidden_dim: int = 256,
        num_classes: int = 10,
        max_iterations: int = 5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_iterations = max_iterations
        
        # Simple encoder: image → initial hidden state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Recurrent refinement block
        self.recurrent = RecurrentRefinementBlock(hidden_dim)
        
        # ACT heads
        self.subjective_head = SubjectiveHead(hidden_dim)
        self.objective_head = ObjectiveHead(hidden_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Conservative initialization to prevent gradient explosion"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Extra conservative for ACT heads
        for m in self.subjective_head.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                m.weight.data *= 0.1
                
        for m in self.objective_head.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                m.weight.data *= 0.1
    
    def forward(
        self, 
        x: torch.Tensor,
        return_act_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with adaptive computation
        
        Args:
            x: Input images [B, 784] or [B, 1, 28, 28]
            return_act_info: Whether to return ACT trajectory info
            
        Returns:
            logits: Class predictions [B, 10]
            act_info: Dictionary with ACT details (if requested)
        """
        # Flatten if needed
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        
        batch_size = x.size(0)
        device = x.device
        
        # Encode to initial hidden state
        h = self.encoder(x)
        
        # Initialize ACT tracking
        trajectory = []
        contributions = []
        thresholds = []
        cumulative_contributions = torch.zeros(batch_size, device=device)
        halt_iterations = torch.zeros(batch_size, dtype=torch.long, device=device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Store initial state
        h_prev = h
        trajectory.append(h)
        
        # Adaptive computation loop
        for i in range(self.max_iterations):
            # Recurrent update
            h_curr = self.recurrent(h_prev)
            trajectory.append(h_curr)
            
            if i > 0:  # No contribution for first step
                # Compute contribution τ_i
                tau = self.subjective_head(h_curr, h_prev)
                contributions.append(tau)
                
                # Accumulate contributions
                cumulative_contributions = cumulative_contributions + tau * (~halted).float()
                
                # Compute threshold o(h^n)
                threshold = self.objective_head(h_curr)
                
                # Add exploration noise during training
                if self.training:
                    noise_scale = 0.1 * (1.0 - i / self.max_iterations)  # Decay noise over iterations
                    threshold = threshold + torch.randn_like(threshold) * noise_scale
                    threshold = threshold.clamp(min=0.1, max=0.9)  # Keep in valid range
                
                thresholds.append(threshold)
                
                # Check halting condition: S_n > o(h^n)
                should_halt = (cumulative_contributions > threshold) & (~halted)
                
                # Update halt tracking
                halt_iterations[should_halt] = i
                halted = halted | should_halt
                
                # Stop if all samples have halted
                if halted.all():
                    break
            
            h_prev = h_curr
        
        # Set remaining samples to max iterations
        halt_iterations[~halted] = self.max_iterations - 1
        
        # Get final hidden states at halt points
        h_final = trajectory[-1]  # For now, use last state (can be refined)
        
        # Classify
        logits = self.classifier(h_final)
        
        if return_act_info:
            act_info = {
                'trajectory': torch.stack(trajectory, dim=0),  # [K+1, B, D]
                'contributions': torch.stack(contributions, dim=0) if contributions else None,  # [K, B]
                'thresholds': torch.stack(thresholds, dim=0) if thresholds else None,  # [K, B]
                'cumulative_contributions': cumulative_contributions,  # [B]
                'halt_iterations': halt_iterations,  # [B]
                'halted': halted,  # [B]
                'k_star': halt_iterations + 1  # Convert to 1-indexed for display [B]
            }
            return logits, act_info
        
        return logits, None


class ACTLoss(nn.Module):
    """
    Simplified ACT loss for MNIST test
    Based on the RL formulation in act.md
    """
    
    def __init__(self, efficiency_penalty: float = 0.01):
        super().__init__()
        self.efficiency_penalty = efficiency_penalty
        
    def forward(
        self,
        act_info: Dict,
        labels: torch.Tensor,
        logits: torch.Tensor,
        update_critic: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute ACT loss (TD error for critic or policy gradient for actor)
        
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of metrics for logging
        """
        device = labels.device
        batch_size = labels.size(0)
        
        # Get trajectory info
        trajectory = act_info['trajectory']  # [K+1, B, D]
        contributions = act_info['contributions']  # [K, B] or None
        thresholds = act_info['thresholds']  # [K, B] or None
        halt_iterations = act_info['halt_iterations']  # [B]
        
        if contributions is None or thresholds is None:
            # No ACT updates if we didn't iterate
            return torch.tensor(0.0, device=device), {
                'optimal_k': 0, 'policy_k': 0, 'agreement': 0,
                'td_error': 0, 'pg_loss': 0
            }
        
        K = contributions.size(0)
        
        # Find optimal k* by evaluating classification loss at each step
        with torch.no_grad():
            losses_per_step = []
            # For simplicity, use the final logits for all steps
            # In a more sophisticated version, we'd re-compute logits at each step
            for k in range(K + 1):
                loss_k = F.cross_entropy(logits, labels, reduction='none')
                # Add efficiency penalty based on computation steps
                loss_k = loss_k + k * self.efficiency_penalty
                losses_per_step.append(loss_k)
            
            losses_per_step = torch.stack(losses_per_step, dim=0)  # [K+1, B]
            optimal_k = losses_per_step.argmin(dim=0)  # [B]
        
        # Compute cumulative contributions at each step
        cumulative_contribs = torch.cumsum(contributions, dim=0)  # [K, B]
        
        if update_critic:
            # TD error for critic (subjective head)
            # Loss = MSE(threshold, cumulative_contribution) for steps >= k*
            td_losses = []
            for b in range(batch_size):
                k_star = optimal_k[b].item()
                if k_star > 0 and k_star <= K:
                    # Compare threshold vs cumulative contribution at k*
                    pred_threshold = thresholds[k_star - 1, b]
                    target_cumul = cumulative_contribs[k_star - 1, b]
                    td_loss = F.mse_loss(pred_threshold, target_cumul.detach())
                    td_losses.append(td_loss)
            
            if td_losses:
                loss = torch.stack(td_losses).mean()
            else:
                loss = torch.tensor(0.0, device=device)
            
            loss_type = 'td_error'
        else:
            # Policy gradient for actor (objective head)
            pg_losses = []
            for b in range(batch_size):
                k_star = optimal_k[b].item()
                k_policy = halt_iterations[b].item()
                
                if k_star != k_policy and k_policy > 0:
                    # Advantage = (k* - k_policy) - negative is bad
                    advantage = float(k_star - k_policy) * -0.1
                    # Policy gradient loss
                    log_prob = torch.log(contributions[:k_policy, b].mean() + 1e-8)
                    pg_loss = -advantage * log_prob
                    pg_losses.append(pg_loss)
            
            if pg_losses:
                loss = torch.stack(pg_losses).mean()
            else:
                loss = torch.tensor(0.0, device=device)
                
            loss_type = 'pg_loss'
        
        # Compute metrics
        agreement = (optimal_k == halt_iterations).float().mean()
        
        metrics = {
            'optimal_k': optimal_k.float().mean().item(),
            'policy_k': halt_iterations.float().mean().item(),
            'agreement': agreement.item(),
            'td_error': loss.item() if update_critic else 0,
            'pg_loss': loss.item() if not update_critic else 0
        }
        
        return loss, metrics


if __name__ == "__main__":
    # Quick test
    model = SimpleACTNet()
    x = torch.randn(4, 784)
    logits, act_info = model(x, return_act_info=True)
    
    print(f"Output shape: {logits.shape}")
    print(f"Halt iterations: {act_info['halt_iterations']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
