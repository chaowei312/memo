"""Main Diffusion Language Model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np
from tqdm import tqdm

from .transformer import TransformerDenoiser
from .utils import get_timestep_embedding, compute_confidence_scores


class DiffusionSchedule:
    """Manages the masking schedule for forward and reverse diffusion."""
    
    def __init__(self, num_timesteps: int = 100, schedule_type: str = "cosine"):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        # Create masking probability schedule
        if schedule_type == "linear":
            self.betas = np.linspace(0.0001, 0.02, num_timesteps)
        elif schedule_type == "cosine":
            steps = np.arange(num_timesteps + 1) / num_timesteps
            alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            self.betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute cumulative products
        self.alphas = 1 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)
        
        # Convert to masking probabilities (higher timestep = more masking)
        self.mask_prob = 1 - self.alpha_bar
    
    def get_mask_prob(self, t: int) -> float:
        """Get masking probability for timestep t."""
        return self.mask_prob[t]
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class DiffusionLM(nn.Module):
    """Diffusion Language Model with masked token diffusion."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        num_timesteps: int = 100,
        dropout: float = 0.1,
        mask_token_id: int = None,
        pad_token_id: int = 0,
        use_flash: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id or vocab_size  # Use last token as mask
        self.pad_token_id = pad_token_id
        
        # Diffusion schedule
        self.schedule = DiffusionSchedule(num_timesteps)
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size + 1, d_model)  # +1 for mask token
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Timestep embedding
        self.time_emb = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Transformer denoiser
        self.denoiser = TransformerDenoiser(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_flash=use_flash
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward_diffusion(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion (masking) process.
        
        Args:
            x: Input token ids [batch_size, seq_len]
            t: Timesteps [batch_size]
            
        Returns:
            x_noised: Masked tokens [batch_size, seq_len]
            mask: Binary mask indicating masked positions [batch_size, seq_len]
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Get masking probability for each timestep
        mask_probs = torch.tensor(
            [self.schedule.get_mask_prob(ti.item()) for ti in t],
            device=device
        ).unsqueeze(1)
        
        # Sample mask for each position
        rand = torch.rand(batch_size, seq_len, device=device)
        mask = rand < mask_probs
        
        # Don't mask padding tokens
        padding_mask = (x == self.pad_token_id)
        mask = mask & ~padding_mask
        
        # Apply mask
        x_noised = x.clone()
        x_noised[mask] = self.mask_token_id
        
        return x_noised, mask
    
    def forward(
        self, 
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.
        
        Args:
            x: Input token ids [batch_size, seq_len]
            t: Timesteps [batch_size] (if None, sample randomly)
            mask: Pre-computed mask [batch_size, seq_len] (if None, compute from t)
            
        Returns:
            Dictionary containing:
                - logits: Token predictions [batch_size, seq_len, vocab_size]
                - x_noised: Masked input tokens
                - mask: Binary mask
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Sample timesteps if not provided
        if t is None:
            t = self.schedule.sample_timesteps(batch_size, device)
        
        # Apply forward diffusion if mask not provided
        if mask is None:
            x_noised, mask = self.forward_diffusion(x, t)
        else:
            x_noised = x.clone()
            x_noised[mask] = self.mask_token_id
        
        # Embed tokens
        token_emb = self.token_emb(x_noised)
        
        # Add position embeddings
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        
        # Add timestep embeddings
        t_emb = get_timestep_embedding(t, self.d_model, device)
        t_emb = self.time_emb(t_emb).unsqueeze(1)
        
        # Combine embeddings
        h = self.dropout(token_emb + pos_emb + t_emb)
        
        # Apply transformer denoiser
        h = self.denoiser(h, t)
        
        # Final layer norm and output projection
        h = self.ln_f(h)
        logits = self.output_proj(h)
        
        return {
            "logits": logits,
            "x_noised": x_noised,
            "mask": mask
        }
    
    @torch.no_grad()
    def reverse_diffusion(
        self,
        x_T: torch.Tensor,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        confidence_threshold: float = 0.8
    ) -> torch.Tensor:
        """Reverse diffusion for generation.
        
        Args:
            x_T: Starting tokens (can be fully masked) [batch_size, seq_len]
            num_steps: Number of denoising steps (default: num_timesteps)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            confidence_threshold: Minimum confidence for unmasking
            
        Returns:
            Generated tokens [batch_size, seq_len]
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        
        batch_size, seq_len = x_T.shape
        device = x_T.device
        x = x_T.clone()
        
        # Reverse diffusion loop
        for i in tqdm(range(num_steps - 1, -1, -1), desc="Reverse diffusion"):
            t = torch.full((batch_size,), i, device=device)
            
            # Get current mask
            mask = (x == self.mask_token_id)
            if not mask.any():
                break  # All tokens unmasked
            
            # Predict tokens
            out = self.forward(x, t, mask)
            logits = out["logits"] / temperature
            
            # Compute confidence scores
            probs = F.softmax(logits, dim=-1)
            confidence = compute_confidence_scores(probs)
            
            # Determine which tokens to unmask
            # Unmask high-confidence masked tokens
            unmask_candidates = mask & (confidence > confidence_threshold)
            
            if unmask_candidates.any():
                # Sample tokens for positions to unmask
                # Apply nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask_cumsum = cumsum > top_p
                mask_cumsum[..., 1:] = mask_cumsum[..., :-1].clone()
                mask_cumsum[..., 0] = False
                sorted_probs[mask_cumsum] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                
                # Sample from filtered distribution
                sampled_indices = torch.multinomial(
                    sorted_probs.view(-1, self.vocab_size), 1
                ).view(batch_size, seq_len)
                sampled_tokens = torch.gather(
                    sorted_indices, -1, sampled_indices.unsqueeze(-1)
                ).squeeze(-1)
                
                # Unmask selected positions
                x[unmask_candidates] = sampled_tokens[unmask_candidates]
            
            # Progressively unmask based on schedule
            target_mask_ratio = self.schedule.get_mask_prob(i)
            current_mask_ratio = mask.float().mean()
            
            if current_mask_ratio > target_mask_ratio:
                # Need to unmask more tokens
                num_to_unmask = int((current_mask_ratio - target_mask_ratio) * seq_len)
                if num_to_unmask > 0 and mask.any():
                    # Unmask highest confidence positions
                    confidence_masked = confidence.clone()
                    confidence_masked[~mask] = -float('inf')
                    _, top_indices = confidence_masked.view(batch_size, -1).topk(
                        min(num_to_unmask, mask.sum().item())
                    )
                    
                    for b in range(batch_size):
                        for idx in top_indices[b]:
                            if mask[b, idx]:
                                x[b, idx] = sampled_tokens[b, idx]
        
        return x
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[torch.Tensor] = None,
        max_length: int = 100,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """Generate text from prompt or from scratch.
        
        Args:
            prompt: Starting tokens [batch_size, prompt_len] or None
            max_length: Maximum sequence length
            num_steps: Number of diffusion steps
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated tokens [batch_size, max_length]
        """
        device = next(self.parameters()).device
        
        if prompt is None:
            # Start from fully masked sequence
            batch_size = 1
            x = torch.full(
                (batch_size, max_length), 
                self.mask_token_id, 
                device=device
            )
        else:
            batch_size, prompt_len = prompt.shape
            # Pad prompt and mask remaining positions
            x = torch.full(
                (batch_size, max_length),
                self.mask_token_id,
                device=device
            )
            x[:, :prompt_len] = prompt
        
        # Run reverse diffusion
        generated = self.reverse_diffusion(
            x, 
            num_steps=num_steps,
            temperature=temperature,
            top_p=top_p
        )
        
        return generated
    
    def compute_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked language modeling loss.
        
        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            mask: Binary mask [batch_size, seq_len]
            
        Returns:
            Loss scalar
        """
        # Only compute loss on masked positions
        loss = F.cross_entropy(
            logits[mask],
            targets[mask],
            reduction='mean'
        )
        return loss
    
    @classmethod
    def from_pretrained(cls, path: str) -> "DiffusionLM":
        """Load pretrained model."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'max_seq_len': self.max_seq_len,
                'num_timesteps': self.num_timesteps
            },
            'model_state_dict': self.state_dict()
        }, path)
