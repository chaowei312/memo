"""
Simple training loop for ACT on MNIST
Much simpler than the UNet version for easier debugging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


class SimpleACTTrainer:
    """Simplified trainer for ACT on MNIST"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        act_weight: float = 0.1
    ):
        self.model = model.to(device)
        self.device = device
        self.act_weight = act_weight
        
        # Import ACT loss
        try:
            from ..model import ACTLoss
        except ImportError:
            # Fallback for direct execution
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from model import ACTLoss
        self.act_loss_fn = ACTLoss(efficiency_penalty=0.01)
        
        # Single optimizer for simplicity (no complex alternating)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Track metrics
        self.train_history = []
        self.val_history = []
        self.update_critic = True
        self.global_step = 0
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        metrics = {
            'loss': [], 'accuracy': [], 'act_loss': [],
            'optimal_k': [], 'policy_k': [], 'agreement': []
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Flatten images
            images = images.view(images.size(0), -1)
            
            # Forward pass
            logits, act_info = self.model(images, return_act_info=True)
            
            # Classification loss
            class_loss = F.cross_entropy(logits, labels)
            
            # ACT loss (RL component)
            act_loss, act_metrics = self.act_loss_fn(
                act_info, labels, logits,
                update_critic=self.update_critic
            )
            
            # Combined loss
            total_loss = class_loss + self.act_weight * act_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == labels).float().mean()
            
            # Update metrics
            metrics['loss'].append(class_loss.item())
            metrics['accuracy'].append(accuracy.item())
            metrics['act_loss'].append(act_loss.item())
            metrics['optimal_k'].append(act_metrics['optimal_k'])
            metrics['policy_k'].append(act_metrics['policy_k'])
            metrics['agreement'].append(act_metrics['agreement'])
            
            # Alternate between actor and critic every 50 steps
            self.global_step += 1
            if self.global_step % 50 == 0:
                self.update_critic = not self.update_critic
                mode = "Critic" if self.update_critic else "Actor"
                pbar.write(f"  Switching to {mode} updates")
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{class_loss.item():.4f}",
                'acc': f"{accuracy.item():.2%}",
                'k': f"{act_metrics['policy_k']:.1f}",
                'agree': f"{act_metrics['agreement']:.1%}"
            })
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        return avg_metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict:
        """Validate model"""
        self.model.eval()
        
        metrics = {
            'loss': [], 'accuracy': [],
            'avg_iterations': [], 'computation_saved': []
        }
        
        for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            images = images.view(images.size(0), -1)
            
            # Forward pass
            logits, act_info = self.model(images, return_act_info=True)
            
            # Loss and accuracy
            loss = F.cross_entropy(logits, labels)
            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == labels).float().mean()
            
            metrics['loss'].append(loss.item())
            metrics['accuracy'].append(accuracy.item())
            
            # ACT metrics
            if act_info and 'halt_iterations' in act_info:
                avg_iter = act_info['halt_iterations'].float().mean()
                metrics['avg_iterations'].append(avg_iter.item())
                
                # Computation saved vs always using max iterations
                saved = 1 - (avg_iter / self.model.max_iterations)
                metrics['computation_saved'].append(saved.item())
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
        return avg_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10
    ) -> Tuple[list, list]:
        """Full training loop"""
        
        print("Starting ACT training on MNIST...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Max iterations: {self.model.max_iterations}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_history.append(val_metrics)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.2%}, "
                  f"Avg k: {train_metrics['policy_k']:.1f}, "
                  f"Agreement: {train_metrics['agreement']:.1%}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.2%}, "
                  f"Avg iter: {val_metrics.get('avg_iterations', 0):.1f}, "
                  f"Saved: {val_metrics.get('computation_saved', 0):.1%}")
            print("-" * 50)
        
        return self.train_history, self.val_history


def get_mnist_loaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Get MNIST data loaders"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def visualize_act_behavior(model: nn.Module, test_loader: DataLoader, device: str = 'cuda'):
    """Visualize how ACT behaves on different digits"""
    model.eval()
    
    # Collect iterations per digit class
    iterations_per_class = {i: [] for i in range(10)}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device).view(images.size(0), -1)
            labels = labels.to(device)
            
            _, act_info = model(images, return_act_info=True)
            halt_iters = act_info['halt_iterations']
            
            for label, halt in zip(labels, halt_iters):
                iterations_per_class[label.item()].append(halt.item() + 1)
    
    # Plot average iterations per digit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Average iterations per digit
    avg_iters = [np.mean(iterations_per_class[i]) for i in range(10)]
    std_iters = [np.std(iterations_per_class[i]) for i in range(10)]
    
    ax1.bar(range(10), avg_iters, yerr=std_iters, capsize=5)
    ax1.set_xlabel('Digit Class')
    ax1.set_ylabel('Average Iterations')
    ax1.set_title('ACT Iterations by Digit Class')
    ax1.set_xticks(range(10))
    ax1.grid(axis='y', alpha=0.3)
    
    # Distribution of iterations
    all_iters = []
    for iters in iterations_per_class.values():
        all_iters.extend(iters)
    
    ax2.hist(all_iters, bins=model.max_iterations, edgecolor='black')
    ax2.set_xlabel('Number of Iterations')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of ACT Iterations')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nACT Behavior Summary:")
    print("Average iterations per digit:")
    for i in range(10):
        print(f"  Digit {i}: {avg_iters[i]:.2f} ± {std_iters[i]:.2f}")
    
    print(f"\nOverall average: {np.mean(all_iters):.2f} iterations")
    print(f"Computation saved: {(1 - np.mean(all_iters)/model.max_iterations)*100:.1f}%")


if __name__ == "__main__":
    # Quick test
    try:
        from ..model import SimpleACTNet
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model import SimpleACTNet
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleACTNet(max_iterations=5)
    
    # Get data
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    # Create trainer
    trainer = SimpleACTTrainer(model, device=device, learning_rate=1e-3, act_weight=0.1)
    
    # Train
    train_history, val_history = trainer.train(train_loader, test_loader, num_epochs=5)
    
    # Visualize ACT behavior
    visualize_act_behavior(model, test_loader, device)
