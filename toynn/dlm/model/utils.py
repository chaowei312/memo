"""Utility functions for DLM."""

import torch
import torch.nn.functional as F
import math
from typing import Optional


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


def compute_confidence_scores(probs: torch.Tensor) -> torch.Tensor:
    """Compute confidence scores for token predictions.
    
    Args:
        probs: Probability distributions [batch_size, seq_len, vocab_size]
        
    Returns:
        Confidence scores [batch_size, seq_len]
    """
    # Use max probability as confidence
    confidence, _ = probs.max(dim=-1)
    
    # Alternative: use entropy-based confidence
    # entropy = -(probs * probs.log()).sum(dim=-1)
    # confidence = 1 - entropy / math.log(probs.shape[-1])
    
    return confidence


def nucleus_sampling(
    logits: torch.Tensor,
    top_p: float = 0.95,
    temperature: float = 1.0
) -> torch.Tensor:
    """Apply nucleus (top-p) sampling.
    
    Args:
        logits: Logits [batch_size, vocab_size]
        top_p: Cumulative probability threshold
        temperature: Sampling temperature
        
    Returns:
        Sampled token indices [batch_size]
    """
    # Apply temperature
    logits = logits / temperature
    
    # Sort probabilities
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # Compute cumulative probabilities
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for nucleus
    mask = cumsum > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    
    # Zero out tokens outside nucleus
    sorted_probs[mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # Sample from filtered distribution
    sampled_sorted_idx = torch.multinomial(sorted_probs, 1)
    sampled_token_idx = torch.gather(
        sorted_indices, -1, sampled_sorted_idx
    ).squeeze(-1)
    
    return sampled_token_idx


def create_padding_mask(
    x: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    """Create attention mask for padding tokens.
    
    Args:
        x: Input tokens [batch_size, seq_len]
        pad_token_id: ID of padding token
        
    Returns:
        Padding mask [batch_size, 1, 1, seq_len]
    """
    mask = (x == pad_token_id).unsqueeze(1).unsqueeze(1)
    return mask


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a schedule with linear warmup and linear decay.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: The index of last epoch
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch
    )
