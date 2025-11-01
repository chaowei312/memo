"""Transformer denoiser for DLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# Try to import flash attention
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Flash Attention not available. Install with: pip install flash-attn --no-build-isolation")


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale_shift = nn.Linear(d_model, d_model * 2)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply AdaLN.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            t_emb: Timestep embeddings [batch_size, d_model]
        """
        x = self.ln(x)
        scale, shift = self.scale_shift(t_emb).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module with optional Flash Attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_flash: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_flash = use_flash and FLASH_AVAILABLE
        
        # For Flash Attention, we use a combined QKV projection
        if self.use_flash:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        else:
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout) if not self.use_flash else None
        self.dropout_p = dropout
        self.scale = math.sqrt(self.d_head)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional Flash Attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        if self.use_flash:
            # Flash Attention path
            qkv = self.qkv_proj(x)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
            
            # Flash attention expects [batch, seq_len, num_heads, head_dim]
            q, k, v = qkv.unbind(dim=2)
            
            # Flash attention computation
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=1.0 / self.scale,
                causal=False
            )
            
            # Reshape output
            out = out.reshape(batch_size, seq_len, self.d_model)
        else:
            # Standard attention path
            q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
            k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
            v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
            
            # Transpose for attention computation
            q = q.transpose(1, 2)  # [batch, heads, seq, d_head]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            
            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
            
            # Apply softmax
            attn = F.softmax(scores, dim=-1)
            if self.dropout is not None:
                attn = self.dropout(attn)
            
            # Apply attention to values
            out = torch.matmul(attn, v)
            
            # Reshape back
            out = out.transpose(1, 2).contiguous()
            out = out.view(batch_size, seq_len, self.d_model)
        
        # Final projection
        out = self.out_proj(out)
        if not self.use_flash and self.dropout is not None:
            out = self.dropout(out)
        
        return out


class TransformerBlock(nn.Module):
    """Single transformer block with AdaLN."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()
        
        # Self-attention
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, use_flash)
        self.ln1 = AdaptiveLayerNorm(d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = AdaptiveLayerNorm(d_model)
        
        # For timestep conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            t_emb: Timestep embeddings [batch_size, d_model]
            mask: Attention mask
        """
        # Process timestep embeddings
        t_emb = self.time_mlp(t_emb)
        
        # Self-attention with residual
        x_norm = self.ln1(x, t_emb)
        x = x + self.attn(x_norm, mask)
        
        # Feed-forward with residual
        x_norm = self.ln2(x, t_emb)
        x = x + self.ff(x_norm)
        
        return x


class TransformerDenoiser(nn.Module):
    """Transformer-based denoiser for DLM."""
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_flash: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_flash = use_flash and FLASH_AVAILABLE
        
        if self.use_flash:
            print(f"Using Flash Attention in TransformerDenoiser")
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, self.use_flash)
            for _ in range(n_layers)
        ])
        
        # Timestep embedding (shared across layers)
        self.time_emb = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through all transformer blocks.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            t: Timesteps [batch_size]
            mask: Attention mask
            
        Returns:
            Output embeddings [batch_size, seq_len, d_model]
        """
        # Get timestep embeddings
        t_emb = get_timestep_embedding(t, self.d_model, x.device)
        t_emb = self.time_emb(t_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, mask)
        
        return x


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    device: torch.device,
    max_period: int = 10000
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: Timestep indices [batch_size]
        embedding_dim: Dimension of the embeddings
        device: Device to create embeddings on
        max_period: Maximum period for sinusoidal embeddings
        
    Returns:
        Timestep embeddings [batch_size, embedding_dim]
    """
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * 
        torch.arange(half_dim, device=device) / half_dim
    )
    
    args = timesteps[:, None].float() * freqs[None]
    embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if embedding_dim % 2:
        embeddings = torch.cat(
            [embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1
        )
    
    return embeddings
