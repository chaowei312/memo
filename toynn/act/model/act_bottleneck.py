"""
Adaptive Computation Time (ACT) Bottleneck for U-Net
Implements RL-based adaptive depth computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RecurrentTransformerBlock(nn.Module):
    """Shared-weight transformer refinement over patch features"""

    def __init__(self, channels, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = max(1, num_heads)
        if channels % self.num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.head_dim = channels // self.num_heads
        if self.head_dim % 4 != 0:
            raise ValueError("2D RoPE requires head_dim divisible by 4")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.pos_proj = nn.Linear(2, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm_out = nn.LayerNorm(channels)

        self.qkv = nn.Linear(channels, channels * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(channels * mlp_ratio), channels),
            nn.Dropout(dropout),
        )

        self.scale = self.head_dim ** -0.5
        self._rope_cache = {}

        # Residual gating within patch space
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _positional_encoding(self, B, H, W, device, dtype):
        y = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([grid_y, grid_x], dim=-1).view(1, H * W, 2)
        coords = coords.expand(B, -1, -1)
        return self.pos_proj(coords.to(dtype))

    def reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.constant_(self.qkv.bias, 0.0)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

        for module in (self.norm1, self.norm2, self.norm_out):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        gate_conv = self.gate[0]
        nn.init.zeros_(gate_conv.weight)
        if gate_conv.bias is not None:
            nn.init.zeros_(gate_conv.bias)

    def tokenize(self, x):
        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).transpose(1, 2)
        pos = self._positional_encoding(B, H, W, x.device, tokens.dtype)
        return tokens + pos, (H, W)

    @staticmethod
    def _rotate_half(x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        x_rot = torch.stack((-x2, x1), dim=-1)
        return x_rot.flatten(start_dim=-2)

    def _apply_rope_single(self, tensor, sin, cos):
        return (tensor * cos) + (self._rotate_half(tensor) * sin)

    def _get_2d_rope_embed(self, Hp, Wp, device, dtype):
        key = (Hp, Wp, str(dtype))
        if key not in self._rope_cache:
            half = self.head_dim // 2
            inv_freq = 1.0 / (10000 ** (torch.arange(0, half, 2, device=device, dtype=dtype) / half))

            y_pos = torch.arange(Hp, device=device, dtype=dtype)
            theta_y = torch.einsum('i,j->ij', y_pos, inv_freq)
            sin_y = torch.repeat_interleave(theta_y.sin(), 2, dim=-1)
            cos_y = torch.repeat_interleave(theta_y.cos(), 2, dim=-1)
            sin_y = sin_y[:, None, :].expand(Hp, Wp, half).reshape(-1, half)
            cos_y = cos_y[:, None, :].expand(Hp, Wp, half).reshape(-1, half)

            x_pos = torch.arange(Wp, device=device, dtype=dtype)
            theta_x = torch.einsum('i,j->ij', x_pos, inv_freq)
            sin_x = torch.repeat_interleave(theta_x.sin(), 2, dim=-1)
            cos_x = torch.repeat_interleave(theta_x.cos(), 2, dim=-1)
            sin_x = sin_x[None, :, :].expand(Hp, Wp, half).reshape(-1, half)
            cos_x = cos_x[None, :, :].expand(Hp, Wp, half).reshape(-1, half)

            cls_sin = torch.zeros(1, half, device=device, dtype=dtype)
            cls_cos = torch.ones(1, half, device=device, dtype=dtype)

            sin_y_full = torch.cat([cls_sin, sin_y], dim=0).unsqueeze(0).unsqueeze(0)
            cos_y_full = torch.cat([cls_cos, cos_y], dim=0).unsqueeze(0).unsqueeze(0)
            sin_x_full = torch.cat([cls_sin, sin_x], dim=0).unsqueeze(0).unsqueeze(0)
            cos_x_full = torch.cat([cls_cos, cos_x], dim=0).unsqueeze(0).unsqueeze(0)

            self._rope_cache[key] = (
                sin_y_full,
                cos_y_full,
                sin_x_full,
                cos_x_full,
            )

        sin_y_full, cos_y_full, sin_x_full, cos_x_full = self._rope_cache[key]
        return (
            sin_y_full.to(device=device, dtype=dtype),
            cos_y_full.to(device=device, dtype=dtype),
            sin_x_full.to(device=device, dtype=dtype),
            cos_x_full.to(device=device, dtype=dtype),
        )

    def _apply_rotary(self, q, k, sin_y, cos_y, sin_x, cos_x):
        half = self.head_dim // 2
        q_y, q_x = q[..., :half], q[..., half:]
        k_y, k_x = k[..., :half], k[..., half:]

        q_y = self._apply_rope_single(q_y, sin_y, cos_y)
        k_y = self._apply_rope_single(k_y, sin_y, cos_y)
        q_x = self._apply_rope_single(q_x, sin_x, cos_x)
        k_x = self._apply_rope_single(k_x, sin_x, cos_x)

        q = torch.cat([q_y, q_x], dim=-1)
        k = torch.cat([k_y, k_x], dim=-1)
        return q, k

    def _self_attention(self, x, Hp, Wp):
        B, tokens, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        sin_y, cos_y, sin_x, cos_x = self._get_2d_rope_embed(Hp, Wp, x.device, x.dtype)
        q, k = self._apply_rotary(q, k, sin_y, cos_y, sin_x, cos_x)

        # Use manual attention computation to avoid kernel issues
        # (Flash Attention can have compatibility issues with some configurations)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(B, tokens, self.channels)
        out = self.proj(context)
        out = self.proj_drop(out)
        return out

    def _attn_mlp_block(self, seq, Hp, Wp):
        seq = seq + self._self_attention(self.norm1(seq), Hp, Wp)
        seq = seq + self.mlp(self.norm2(seq))
        return seq

    def forward(self, x_prev, x_init, cls_prev=None):
        """Returns refined spatial state and CLS embedding."""
        B, C, H, W = x_prev.shape
        tokens, (Hp, Wp) = self.tokenize(x_prev)

        if cls_prev is None:
            cls_token = self.cls_token.expand(B, -1, -1)
        else:
            cls_token = cls_prev.unsqueeze(1)

        seq = torch.cat([cls_token, tokens], dim=1)
        seq = self._attn_mlp_block(seq, Hp, Wp)
        seq = self.norm_out(seq)

        cls_out = seq[:, 0]
        tokens_out = seq[:, 1:]
        x_tokens = tokens_out.transpose(1, 2).view(B, C, Hp, Wp)

        gate = self.gate(torch.cat([x_tokens, x_init], dim=1))
        x_next = gate * x_tokens + (1.0 - gate) * x_prev

        return x_next, cls_out


class SubjectiveHead(nn.Module):
    """Estimates subjective contribution τ_i at each step"""
    
    def __init__(self, channels):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        hidden_dim = max(32, channels)
        mid_dim = max(16, channels // 2)
        
        # Feed-forward network g for computing contribution from state/CLS differences
        self.contribution_net = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
            nn.Sigmoid()  # Bounded [0, 1] to prevent explosion
        )
        
    def reset_parameters(self):
        for module in self.contribution_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_curr, x_prev, cls_curr, cls_prev):
        """Compute subjective contribution τ_i based on spatial and CLS deltas."""
        diff = x_curr - x_prev
        cls_diff = cls_curr - cls_prev

        spatial_change = torch.norm(diff, p=2, dim=1).mean(dim=(1, 2))  # [B]

        feat_prev = self.pool(x_prev).flatten(1)
        feat_curr = self.pool(x_curr).flatten(1)
        feat_prev_norm = F.normalize(feat_prev, p=2, dim=1)
        feat_curr_norm = F.normalize(feat_curr, p=2, dim=1)
        cosine_sim = torch.sum(feat_prev_norm * feat_curr_norm, dim=1)
        feature_change = 1.0 - (cosine_sim + 1.0) / 2.0

        diff_vec = self.pool(diff).flatten(1)
        combined = torch.cat([diff_vec, cls_diff], dim=1)
        g_diff = self.contribution_net(combined).squeeze(-1)  # [B] in [0, 1]
        # Scale to reasonable range (0 to 0.3) to prevent explosion
        g_diff = g_diff * 0.3

        tau = spatial_change * (1.0 + feature_change) * g_diff

        return tau


class ObjectiveHead(nn.Module):
    """Estimates objective computation quality threshold o(x^n)"""
    
    def __init__(self, channels):
        super().__init__()

        hidden_dim = max(32, channels)
        mid_dim = max(16, channels // 2)

        self.threshold_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
            nn.Sigmoid()  # Bounded [0, 1] to prevent explosion
        )

        self.min_threshold = nn.Parameter(torch.tensor(0.1))  # Smaller initial threshold

    def reset_parameters(self):
        for module in self.threshold_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        with torch.no_grad():
            self.min_threshold.copy_(torch.tensor(1.0))

    def forward(self, cls_token):
        """Compute objective threshold o(x^n) from the CLS embedding."""
        threshold = self.threshold_net(cls_token).squeeze(-1)  # [B] in [0, 1]
        # Scale to reasonable range (0.1 to 0.6) to prevent explosion
        threshold = 0.1 + threshold * 0.5
        return threshold


class ACTBottleneck(nn.Module):
    """
    ACT-enhanced bottleneck with adaptive computational depth
    
    Operates on encoded CNN features from U-Net encoder (not raw images!)
    Input: Encoded features of shape [B, C, H/16, W/16] where C=512 or 1024
    Output: Refined features after adaptive iterations
    """
    
    def __init__(self, channels, max_iterations=5, base_channels=None, patch_size=4):
        super().__init__()
        self.channels = channels
        self.max_iterations = max_iterations
        self.patch_size = patch_size
        
        if base_channels is None:
            base_channels = channels
            
        # Initial projection if needed
        self.input_proj = nn.Conv2d(base_channels, channels, 1) if base_channels != channels else nn.Identity()

        # Patch embedding handled before recurrent iterations
        self.patch_embed = nn.Conv2d(
            channels,
            channels,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=False,
        )
        
        # Recurrent processing block (shared weights) operates on patch features
        self.recurrent_block = RecurrentTransformerBlock(channels)
        
        # Value function heads for halting policy
        self.subjective_head = SubjectiveHead(channels)
        self.objective_head = ObjectiveHead(channels)
        
        # Output projection
        self.output_proj = nn.Conv2d(channels, base_channels, 1) if base_channels != channels else nn.Identity()

        self.reset_parameters()
        
    def reset_parameters(self):
        if isinstance(self.input_proj, nn.Conv2d):
            nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_out', nonlinearity='relu')
            if self.input_proj.bias is not None:
                nn.init.zeros_(self.input_proj.bias)

        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')

        self.recurrent_block.reset_parameters()
        self.subjective_head.reset_parameters()
        self.objective_head.reset_parameters()
        
        # Extra conservative initialization to prevent gradient explosion
        with torch.no_grad():
            # Scale down recurrent block weights to prevent accumulation
            for m in self.recurrent_block.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data *= 0.5  # Reduce magnitude
            
            # Ensure heads start with very small outputs
            for m in self.subjective_head.modules():
                if isinstance(m, nn.Linear) and m.out_features == 1:
                    m.weight.data *= 0.1
                    if m.bias is not None:
                        m.bias.data.zero_()
                        
            for m in self.objective_head.modules():
                if isinstance(m, nn.Linear) and m.out_features == 1:
                    m.weight.data *= 0.1
                    if m.bias is not None:
                        m.bias.data.zero_()

        if isinstance(self.output_proj, nn.Conv2d):
            nn.init.kaiming_normal_(self.output_proj.weight, mode='fan_out', nonlinearity='relu')
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)

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
        x_proj = self.input_proj(x)

        # Patch embedding (encoder responsibility)
        x_patch_init = self.patch_embed(x_proj)
        
        # Initialize trajectory storage
        contributions = []
        thresholds = []
        
        # Initialize state and CLS summaries in patch space
        x_patch_curr = x_patch_init
        cls_curr = self.recurrent_block.cls_token.expand(batch_size, -1, -1).squeeze(1)
        if return_trajectory:
            cls_history = [cls_curr]
            patch_trajectory = []
            feature_trajectory = []
        else:
            cls_history = None
        cumulative_contribution = torch.zeros(batch_size, device=device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        halt_iterations = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Iterate up to max_iterations
        for i in range(self.max_iterations):
            # Apply recurrent transformation
            cls_prev = cls_curr
            x_patch_next, cls_next = self.recurrent_block(x_patch_curr, x_patch_init, cls_prev=cls_prev)
            
            if i > 0:  # Skip first iteration (no previous state difference)
                # Compute contribution and threshold
                tau_i = self.subjective_head(x_patch_next, x_patch_curr, cls_next, cls_prev)
                o_i = self.objective_head(cls_next)
                
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
                cls_history.append(cls_next)
                patch_trajectory.append(x_patch_next.detach())
                feature_trajectory.append(
                    F.interpolate(
                        x_patch_next,
                        size=x_proj.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    ).detach()
                )
            
            # Update states
            x_patch_curr = x_patch_next
            cls_curr = cls_next
            
            # Early exit if all samples have halted (during inference)
            if not return_trajectory and halted.all():
                break
        
        # Set remaining halt iterations for samples that never halted
        halt_iterations[~halted] = self.max_iterations - 1
        
        # Select output based on halt iterations
        if return_trajectory:
            patch_trajectory = torch.stack(patch_trajectory, dim=0) if patch_trajectory else None
            feature_trajectory = torch.stack(feature_trajectory, dim=0) if feature_trajectory else None
            cls_tokens = torch.stack(cls_history, dim=0) if cls_history is not None else None

            batch_indices = torch.arange(batch_size, device=device)
            if patch_trajectory is not None:
                output_patch = patch_trajectory[halt_iterations, batch_indices]
            else:
                output_patch = x_patch_curr

            info = {
                'patch_trajectory': patch_trajectory,
                'feature_trajectory': feature_trajectory,
                'cls_tokens': cls_tokens,
                'contributions': torch.stack(contributions) if contributions else None,
                'thresholds': torch.stack(thresholds) if thresholds else None,
                'halt_iterations': halt_iterations,
                'cumulative_contributions': cumulative_contribution
            }
        else:
            output_patch = x_patch_curr
            info = {
                'halt_iterations': halt_iterations,
                'cumulative_contributions': cumulative_contribution
            }
        
        # Restore spatial resolution for decoder
        output = F.interpolate(
            output_patch,
            size=x_proj.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        output = self.output_proj(output)
        
        return output, info
