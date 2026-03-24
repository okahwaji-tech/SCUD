"""Protein data loading via HuggingFace datasets: UniRef50."""

from __future__ import annotations

import logging

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from scud.utils import _pad

# HuggingFace dataset ID
HF_UNIREF50 = "fredzzp/Uniref50"

protein_data_name_dict: dict[str, str] = {"uniref50": HF_UNIREF50}

logger = logging.getLogger(__name__)


def get_protein_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train/test DataLoaders from HuggingFace UniRef50.

    Downloads the dataset on first use (cached by HF afterwards).
    Each item is a dict with keys ``"sequence"`` and ``"length"``.

    Args:
        cfg: Hydra config with ``data``, ``train`` sections.

    Returns:
        Tuple of (train_dataloader, test_dataloader).
    """
    from evodiff.utils import Tokenizer

    batch_size: int = int(cfg.train.batch_size)
    max_len = 1024

    tokenizer = Tokenizer()

    logger.info("Loading UniRef50 from HuggingFace.")
    hf_dataset = getattr(cfg.data, "hf_dataset", HF_UNIREF50)
    ds = load_dataset(hf_dataset)

    train_dataset = ds["train"]
    test_dataset = ds["test"]

    def mask_pad(
        tokenized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = tokenized != tokenizer.pad_id
        return tokenized.long(), masks.float()

    def collate_fn(
        batch: list[dict[str, object]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [str(item["sequence"])[:max_len] for item in batch]
        tokenized_list = [torch.tensor(tokenizer.tokenize(s)) for s in seqs]
        tokenized_padded = _pad(tokenized_list, tokenizer.pad_id)
        return mask_pad(tokenized_padded)

    logger.info("Setting N workers.")
    num_workers = 16 // torch.cuda.device_count()

    if hasattr(cfg.train, "pack") and cfg.train.pack:
        block_size = 13

        def collate_fn_pack(
            batch: list[dict[str, object]],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            seqs = [item["sequence"] + tokenizer.pad for item in batch]
            if len(seqs) % block_size != 0:
                seqs = seqs + (block_size - len(seqs) % block_size) * [""]
            strings = np.array(seqs).reshape(-1, block_size)
            strings[0, 1:] = ""
            packed: list[dict[str, object]] = [
                {"sequence": "".join(row)[:max_len]} for row in strings
            ]
            return collate_fn(packed)

        logger.info("Building dataloader.")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size * block_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=collate_fn_pack,
            pin_memory=True,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    logger.info("Building test dataloader.")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader
