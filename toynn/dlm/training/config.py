"""Training configuration for DLM."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for DLM training."""
    
    # Data
    data_path: str = "data/processed"
    val_data_path: Optional[str] = None
    max_length: int = 512
    
    # Model
    vocab_size: int = 30000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    use_flash: bool = True  # Use Flash Attention by default
    
    # Diffusion
    num_timesteps: int = 100
    schedule_type: str = "cosine"
    
    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "linear"
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device
    device: str = "cuda"
    num_workers: int = 0
    
    # Experiment tracking
    use_wandb: bool = False
    project_name: str = "dlm-training"
    run_name: Optional[str] = None
    
    # Generation
    num_generation_steps: int = 50
    generation_temperature: float = 1.0
    generation_top_p: float = 0.95
    
    def to_dict(self):
        """Convert config to dictionary."""
        return self.__dict__
