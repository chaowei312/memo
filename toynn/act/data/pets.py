"""
Oxford-IIIT Pets dataset handler for segmentation
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet
import numpy as np
from PIL import Image
from typing import Optional, Callable


class PetsDataset(Dataset):
    """Oxford Pets dataset wrapper for segmentation"""
    
    def __init__(
        self,
        root: str = './datasets',
        split: str = 'trainval',
        size: tuple = (256, 256),
        transform: Optional[Callable] = None,
        download: bool = True
    ):
        """
        Args:
            root: Root directory for datasets
            split: 'trainval' or 'test'
            size: Image size (height, width)
            transform: Optional transform pipeline
            download: Whether to download if not present
        """
        self.root = root
        self.split = split
        self.size = size
        
        # Load dataset with segmentation masks
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types='segmentation',
            download=download
        )
        
        # Default transforms if not provided
        if transform is None:
            self.transform = T.Compose([
                T.Resize(size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        self.mask_transform = T.Compose([
            T.Resize(size, interpolation=Image.NEAREST),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get image and trimap from dataset
        image, trimap = self.dataset[idx]
        
        # Convert trimap to binary mask (1: foreground, 2: background, 3: boundary)
        # We'll treat boundary as foreground for binary segmentation
        trimap = self.mask_transform(trimap)
        mask = np.array(trimap)
        binary_mask = (mask == 1) | (mask == 3)  # Pet pixels (foreground + boundary)
        binary_mask = binary_mask.astype(np.float32)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert mask to tensor
        mask = torch.from_numpy(binary_mask).unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'trimap': torch.from_numpy(np.array(trimap)).long(),  # Original trimap
            'idx': idx
        }
    
    @staticmethod
    def get_data_loaders(
        batch_size: int = 8,
        num_workers: int = 4,
        root: str = './datasets',
        size: tuple = (256, 256),
        download: bool = True
    ):
        """Convenience method to get train and test loaders"""
        
        # Training dataset with augmentation
        train_transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = PetsDataset(
            root=root,
            split='trainval',
            size=size,
            transform=train_transform,
            download=download
        )
        
        # Test dataset without augmentation
        test_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = PetsDataset(
            root=root,
            split='test',
            size=size,
            transform=test_transform,
            download=download
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader
