"""
ACT-Enhanced U-Net combining base architecture with adaptive bottleneck
"""

import torch
import torch.nn as nn
from .unet_base import DoubleConv, Down, Up
from .act_bottleneck import ACTBottleneck


class ACTUNet(nn.Module):
    """U-Net with ACT-enhanced adaptive bottleneck"""
    
    def __init__(self, n_channels, n_classes, max_iterations=5, bilinear=True):
        super(ACTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.max_iterations = max_iterations
        self.bilinear = bilinear
        
        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # ACT Bottleneck with adaptive depth
        self.bottleneck = ACTBottleneck(
            channels=512,  # Internal processing channels
            max_iterations=max_iterations,
            base_channels=1024 // factor  # Input/output channels
        )
        
        # Decoder path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x, return_act_info=False):
        """
        Forward pass through ACT-UNet
        
        Args:
            x: Input image [B, C, H, W]
            return_act_info: If True, return ACT trajectory info
            
        Returns:
            logits: Segmentation logits [B, n_classes, H, W]
            act_info: ACT information dictionary (if return_act_info=True)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # ACT Bottleneck
        x_bottleneck, act_info = self.bottleneck(x5, return_trajectory=return_act_info)
        
        # Decoder
        x = self.up1(x_bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        if return_act_info:
            return logits, act_info
        return logits
    
    def get_average_iterations(self, dataloader, device='cuda'):
        """Compute average number of iterations used on a dataset"""
        self.eval()
        total_iterations = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                _, act_info = self.forward(images, return_act_info=True)
                total_iterations += act_info['halt_iterations'].sum().item()
                total_samples += images.shape[0]
        
        return total_iterations / total_samples if total_samples > 0 else 0
