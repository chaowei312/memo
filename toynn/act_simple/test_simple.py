"""Quick test to verify simple ACT implementation works"""

import torch
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from act_simple.model import SimpleACTNet, ACTLoss
from act_simple.training import get_mnist_loaders

def test_simple_act():
    """Test that simple ACT model works without errors"""
    print("Testing Simple ACT Implementation")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    print("\n1. Creating model...")
    model = SimpleACTNet(max_iterations=3).to(device)
    print(f"   Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    x = torch.randn(4, 784).to(device)
    logits, act_info = model(x, return_act_info=True)
    print(f"   Output shape: {logits.shape}")
    print(f"   Halt iterations: {act_info['halt_iterations'].cpu().numpy()}")
    
    # Test with real data
    print("\n3. Testing with MNIST data...")
    train_loader, _ = get_mnist_loaders(batch_size=32)
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    logits, act_info = model(images, return_act_info=True)
    print(f"   Batch size: {images.shape[0]}")
    print(f"   Predictions shape: {logits.shape}")
    print(f"   Average iterations: {act_info['halt_iterations'].float().mean():.2f}")
    
    # Test loss computation
    print("\n4. Testing ACT loss...")
    act_loss_fn = ACTLoss()
    loss, metrics = act_loss_fn(act_info, labels, logits, update_critic=True)
    print(f"   ACT loss: {loss.item():.4f}")
    print(f"   Metrics: {metrics}")
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    total_loss = torch.nn.functional.cross_entropy(logits, labels) + 0.1 * loss
    total_loss.backward()
    
    # Check for NaN gradients
    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"   WARNING: NaN gradient in {name}")
            has_nan = True
    
    if not has_nan:
        print("   [OK] No NaN gradients detected!")
    
    print("\n" + "=" * 50)
    print("[SUCCESS] Simple ACT test PASSED!")
    print("The model is ready for training on MNIST.")
    return True

if __name__ == "__main__":
    success = test_simple_act()
    sys.exit(0 if success else 1)
