"""Image data loading: MNIST, CIFAR10 dataloaders."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

image_data_name_dict: dict[str, type] = {
    "CIFAR10": CIFAR10,
    "MNIST": MNIST,
}


def get_img_dataloaders(
    cfg: object,
) -> tuple[DataLoader, DataLoader]:
    batch_size = cfg.train.batch_size

    train_dataset = image_data_name_dict[cfg.data.data](
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    test_dataset = image_data_name_dict[cfg.data.data](
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )

    def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> torch.Tensor:
        x, cond = zip(*batch)
        x = torch.stack(x)
        x = (x * (cfg.data.N - 1)).round().long().clamp(0, cfg.data.N - 1)
        return x

    # multiprocessing.cpu_count()
    num_workers = 16 // max([1, torch.cuda.device_count()])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_dataloader, test_dataloader
