"""
Simple ACT (Adaptive Computation Time) Test Implementation
Testing the ACT theory on MNIST before complex UNet integration
"""

from .model import SimpleACTNet, ACTLoss
from .training import SimpleACTTrainer, get_mnist_loaders, visualize_act_behavior

__all__ = ['SimpleACTNet', 'ACTLoss', 'SimpleACTTrainer', 'get_mnist_loaders', 'visualize_act_behavior']
