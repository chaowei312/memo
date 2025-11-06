"""
Generic dataset utilities for segmentation tasks
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import random
from pathlib import Path
from typing import Optional, Callable, Tuple


class SegmentationDataset(Dataset):
    """Generic segmentation dataset"""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform
        
        # Get all image files
        self.images = sorted(list(self.image_dir.glob('*.png')) + 
                            list(self.image_dir.glob('*.jpg')) +
                            list(self.image_dir.glob('*.jpeg')))
        
        # Verify masks exist
        self.masks = []
        for img_path in self.images:
            mask_name = img_path.stem + '.png'  # Assume masks are PNG
            mask_path = self.mask_dir / mask_name
            if not mask_path.exists():
                mask_name = img_path.name  # Try same name
                mask_path = self.mask_dir / mask_name
            self.masks.append(mask_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')  # Grayscale mask
        
        # Apply joint transforms (same random params for both)
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        
        # Apply individual transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        return {
            'image': image,
            'mask': mask,
            'image_path': str(self.images[idx]),
            'mask_path': str(self.masks[idx])
        }


class JointTransform:
    """Apply same random transform to image and mask"""
    
    def __init__(self, size: Tuple[int, int], augment: bool = True):
        self.size = size
        self.augment = augment
    
    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)
        
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.randint(-30, 30)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, fill=0)
            
            # Random crop
            if random.random() > 0.5:
                i, j, h, w = T.RandomCrop.get_params(
                    image, output_size=(int(self.size[0] * 0.8), int(self.size[1] * 0.8))
                )
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
                # Resize back
                image = TF.resize(image, self.size)
                mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)
        
        return image, mask


def get_transforms(
    size: Tuple[int, int] = (256, 256),
    augment: bool = True,
    normalize: bool = True
):
    """Get standard transforms for segmentation"""
    
    # Joint transform for geometric augmentations
    joint_transform = JointTransform(size, augment)
    
    # Image transforms
    image_transforms = [T.ToTensor()]
    if normalize:
        # ImageNet normalization
        image_transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    image_transform = T.Compose(image_transforms)
    
    # Mask transform (just to tensor, values should be 0 or 1)
    def mask_transform(mask):
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        mask = mask.unsqueeze(0)  # Add channel dimension
        return mask
    
    return {
        'joint': joint_transform,
        'image': image_transform,
        'mask': mask_transform
    }
