"""Image data loading via HuggingFace datasets: CIFAR-10, MNIST."""

from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

# HuggingFace dataset IDs
HF_DATASETS: dict[str, str] = {
    "CIFAR10": "uoft-cs/cifar10",
    "MNIST": "ylecun/mnist",
}

# Column names in HF datasets
IMG_COLUMN: dict[str, str] = {
    "CIFAR10": "img",
    "MNIST": "image",
}

# Backward-compatible routing dict used by loader.py
image_data_name_dict: dict[str, str] = {
    "CIFAR10": "CIFAR10",
    "MNIST": "MNIST",
}


class _ImageTransform:
    """Picklable transform for ``Dataset.with_transform`` (supports multiprocess workers)."""

    def __init__(self, col: str, n_levels: int, train: bool) -> None:
        self.col = col
        self.n_levels = n_levels
        self.train = train

    def __call__(self, batch: dict[str, list[Any]]) -> dict[str, list[torch.Tensor]]:
        tensors: list[torch.Tensor] = []
        for img in batch[self.col]:
            x = TF.to_tensor(img)
            if self.train:
                x = TF.hflip(x) if torch.rand(1).item() < 0.5 else x
            x = (x * (self.n_levels - 1)).round().long().clamp(0, self.n_levels - 1)
            tensors.append(x)
        batch[self.col] = tensors
        return batch


class _CollateFn:
    """Picklable collate function for multiprocess DataLoader workers."""

    def __init__(self, col: str) -> None:
        self.col = col

    def __call__(self, batch: list[dict[str, Any]]) -> torch.Tensor:
        return torch.stack([item[self.col] for item in batch])


def get_img_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train/test dataloaders for CIFAR-10 or MNIST."""
    data_name: str = cfg.data.data
    n_levels: int = cfg.data.N
    batch_size: int = cfg.train.batch_size

    # Allow overriding the HF dataset ID via config
    hf_id: str = getattr(cfg.data, "hf_dataset", HF_DATASETS[data_name])
    col = IMG_COLUMN[data_name]

    ds = load_dataset(hf_id)

    train_ds = ds["train"].with_transform(
        _ImageTransform(col, n_levels, train=True),
    )
    test_ds = ds["test"].with_transform(
        _ImageTransform(col, n_levels, train=False),
    )

    if torch.cuda.is_available():
        num_workers = 16 // max(1, torch.cuda.device_count())
    elif torch.backends.mps.is_available():
        num_workers = 4  # Apple Silicon: moderate parallelism
    else:
        num_workers = 0

    collate = _CollateFn(col)

    train_dataloader = DataLoader(
        train_ds,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    test_dataloader = DataLoader(
        test_ds,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataloader, test_dataloader
