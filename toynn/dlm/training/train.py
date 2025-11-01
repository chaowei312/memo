"""Training script for DLM."""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.diffusion import DiffusionLM
from model.utils import get_linear_schedule_with_warmup
from data.dataset import create_dataloader
from training.config import TrainingConfig


def train_step(
    model: DiffusionLM,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig
) -> Dict[str, float]:
    """Single training step.
    
    Args:
        model: DLM model
        batch: Batch of data
        optimizer: Optimizer
        config: Training configuration
        
    Returns:
        Dictionary of metrics
    """
    model.train()
    
    # Move batch to device
    input_ids = batch['input_ids'].to(config.device)
    attention_mask = batch['attention_mask'].to(config.device)
    
    # Forward pass
    outputs = model(input_ids)
    
    # Compute loss (only on masked tokens)
    loss = model.compute_loss(
        outputs['logits'],
        input_ids,
        outputs['mask']
    )
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    if config.gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Compute accuracy on masked tokens
    with torch.no_grad():
        mask = outputs['mask']
        if mask.any():
            preds = outputs['logits'][mask].argmax(dim=-1)
            targets = input_ids[mask]
            accuracy = (preds == targets).float().mean().item()
        else:
            accuracy = 0.0
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'mask_ratio': mask.float().mean().item()
    }


@torch.no_grad()
def evaluate_model(
    model: DiffusionLM,
    dataloader: DataLoader,
    config: TrainingConfig,
    num_samples: int = 100
) -> Dict[str, float]:
    """Evaluate model on validation set.
    
    Args:
        model: DLM model
        dataloader: Validation dataloader
        config: Training configuration
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    total_loss = 0
    total_accuracy = 0
    total_samples = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples // config.batch_size:
            break
        
        # Move batch to device
        input_ids = batch['input_ids'].to(config.device)
        
        # Forward pass
        outputs = model(input_ids)
        
        # Compute loss
        loss = model.compute_loss(
            outputs['logits'],
            input_ids,
            outputs['mask']
        )
        
        # Compute accuracy
        mask = outputs['mask']
        if mask.any():
            preds = outputs['logits'][mask].argmax(dim=-1)
            targets = input_ids[mask]
            accuracy = (preds == targets).float().mean().item()
        else:
            accuracy = 0.0
        
        total_loss += loss.item()
        total_accuracy += accuracy
        total_samples += 1
    
    return {
        'val_loss': total_loss / total_samples,
        'val_accuracy': total_accuracy / total_samples
    }


@torch.no_grad()
def generate_samples(
    model: DiffusionLM,
    tokenizer: AutoTokenizer,
    num_samples: int = 3,
    max_length: int = 100,
    config: Optional[TrainingConfig] = None
) -> List[str]:
    """Generate text samples.
    
    Args:
        model: DLM model
        tokenizer: Tokenizer
        num_samples: Number of samples to generate
        max_length: Maximum length of generated text
        config: Training configuration
        
    Returns:
        List of generated texts
    """
    model.eval()
    
    samples = []
    
    for _ in range(num_samples):
        # Generate from scratch (fully masked)
        generated_ids = model.generate(
            prompt=None,
            max_length=max_length,
            num_steps=config.num_generation_steps if config else 50,
            temperature=config.generation_temperature if config else 1.0,
            top_p=config.generation_top_p if config else 0.95
        )
        
        # Decode to text
        generated_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        samples.append(generated_text)
    
    return samples


def train_model(config: TrainingConfig):
    """Main training function.
    
    Args:
        config: Training configuration
    """
    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    # Update vocab size based on tokenizer
    config.vocab_size = len(tokenizer)
    
    # Create dataloaders
    print("Loading datasets...")
    train_dataloader = create_dataloader(
        config.data_path,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
        max_length=config.max_length,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_dataloader = None
    if config.val_data_path:
        val_dataloader = create_dataloader(
            config.val_data_path,
            batch_size=config.batch_size,
            tokenizer=tokenizer,
            max_length=config.max_length,
            shuffle=False,
            num_workers=config.num_workers
        )
    
    # Initialize model
    print("Initializing model...")
    model = DiffusionLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_length,
        num_timesteps=config.num_timesteps,
        dropout=config.dropout,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_flash=config.use_flash  # Use config setting for Flash Attention
    ).to(config.device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Initialize scheduler
    num_training_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize wandb if requested
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.to_dict()
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            config.use_wandb = False
    
    # Training loop
    print("Starting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{config.num_epochs}"
        )
        
        for batch in progress_bar:
            # Training step
            metrics = train_step(model, batch, optimizer, config)
            scheduler.step()
            
            # Update metrics
            epoch_loss += metrics['loss']
            epoch_accuracy += metrics['accuracy']
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'mask': f"{metrics['mask_ratio']:.2f}"
            })
            
            # Log metrics
            if global_step % config.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                avg_acc = epoch_accuracy / num_batches
                
                if config.use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/accuracy': avg_acc,
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'step': global_step
                    })
            
            # Evaluate
            if global_step % config.eval_interval == 0 and val_dataloader:
                val_metrics = evaluate_model(model, val_dataloader, config)
                print(f"\nValidation - Loss: {val_metrics['val_loss']:.4f}, "
                      f"Accuracy: {val_metrics['val_accuracy']:.4f}")
                
                if config.use_wandb:
                    import wandb
                    wandb.log({
                        'val/loss': val_metrics['val_loss'],
                        'val/accuracy': val_metrics['val_accuracy'],
                        'step': global_step
                    })
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    model.save_pretrained(
                        os.path.join(config.checkpoint_dir, 'best_model.pt')
                    )
            
            # Save checkpoint
            if global_step % config.save_interval == 0:
                checkpoint_path = os.path.join(
                    config.checkpoint_dir,
                    f'checkpoint_step_{global_step}.pt'
                )
                model.save_pretrained(checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
                
                # Generate samples
                samples = generate_samples(
                    model, tokenizer, num_samples=3, config=config
                )
                print("\nGenerated samples:")
                for i, sample in enumerate(samples):
                    print(f"{i+1}: {sample[:200]}...")
        
        # End of epoch
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_accuracy / num_batches
        print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, 'final_model.pt')
    model.save_pretrained(final_path)
    print(f"\nTraining complete! Saved final model to {final_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Diffusion Language Model')
    
    # Add arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--num_timesteps', type=int, default=100,
                       help='Number of diffusion timesteps')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--no_flash', action='store_true',
                       help='Disable Flash Attention (enabled by default)')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        num_timesteps=args.num_timesteps,
        device=args.device,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        use_flash=not args.no_flash  # Enable Flash unless --no_flash is specified
    )
    
    # Train model
    train_model(config)


if __name__ == '__main__':
    main()
