"""Utility functions for preparing ACT training dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

try:  # When imported as part of the toynn.act package
    from ..data import SegmentationDataset, PetsDataset, get_transforms
except ImportError:
    try:  # Fallback for absolute import when package root is on sys.path
        from toynn.act.data import SegmentationDataset, PetsDataset, get_transforms
    except ImportError:  # Final fallback for running notebooks/scripts from repo root
        from data import SegmentationDataset, PetsDataset, get_transforms


@dataclass
class DataConfig:
    """Configuration describing how to build training/validation dataloaders."""

    dataset: str = "custom"  # "custom" | "pets"
    image_dir: Optional[str] = None
    mask_dir: Optional[str] = None
    root: str = "./datasets"
    input_size: Tuple[int, int] = (256, 256)
    batch_size: int = 4
    num_workers: int = 4
    val_split: float = 0.2
    seed: int = 42
    augment: bool = True
    normalize: bool = True
    download: bool = True


def build_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders according to the supplied configuration."""

    if config.dataset.lower() == "pets":
        train_loader, val_loader = PetsDataset.get_data_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            root=config.root,
            size=config.input_size,
            download=config.download,
        )
        return train_loader, val_loader

    if config.image_dir is None or config.mask_dir is None:
        raise ValueError("`image_dir` and `mask_dir` must be provided for custom datasets")

    transforms = get_transforms(
        size=config.input_size,
        augment=config.augment,
        normalize=config.normalize,
    )

    full_dataset: Dataset = SegmentationDataset(
        image_dir=config.image_dir,
        mask_dir=config.mask_dir,
        transform=transforms["image"],
        mask_transform=transforms["mask"],
        joint_transform=transforms["joint"],
    )

    if not 0.0 < config.val_split < 1.0:
        raise ValueError("`val_split` must be between 0 and 1 for custom datasets")

    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("Dataset too small for the requested validation split")

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


__all__ = ["DataConfig", "build_dataloaders"]

