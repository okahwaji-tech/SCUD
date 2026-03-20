"""Protein data loading: UniRef50 dataloaders."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from evodiff.utils import Tokenizer
from scud.utils import _pad
from sequence_models.datasets import UniRefDataset

protein_data_name_dict: dict[str, None] = {"uniref50": None}


def get_protein_dataloaders(
    cfg: object,
) -> tuple[DataLoader, DataLoader]:
    batch_size = cfg.train.batch_size

    max_len = 1024
    tokenizer = Tokenizer()
    print("Getting Uniref.")
    data_dir = (
        cfg.data.data_dir
        if hasattr(cfg.data, "data_dir")
        else "data/uniref_2020/uniref50/"
    )
    train_dataset = UniRefDataset(
        data_dir, "train", structure=False, max_len=max_len
    )
    test_dataset = UniRefDataset(
        data_dir, "test", structure=False, max_len=max_len
    )

    def mask_pad(
        tokenized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = tokenized != tokenizer.pad_id
        return tokenized.long(), masks.float()

    def collate_fn(
        batch: list[tuple[str, ...]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = [torch.tensor(tokenizer.tokenize(s)) for s in batch]
        tokenized = _pad(tokenized, tokenizer.pad_id)
        return mask_pad(tokenized)

    # multiprocessing.cpu_count()
    print("Setting N workers.")
    num_workers = 16 // torch.cuda.device_count()
    if hasattr(cfg.train, "pack") and cfg.train.pack:
        block_size = 13

        def collate_fn_pack(
            batch: list[tuple[str, ...]],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch = [string[0] + tokenizer.pad for string in batch]
            if len(batch) % block_size != 0:
                batch = batch + (block_size - len(batch) % block_size) * [""]
            strings = np.array(batch).reshape(-1, block_size)
            strings[0, 1:] = ""
            strings = [("".join(strs)[:max_len],) for strs in strings]
            return collate_fn(strings)

        print("Building dataloader.")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size * block_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=collate_fn_pack,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    print("Building test dataloader.")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_dataloader, test_dataloader
