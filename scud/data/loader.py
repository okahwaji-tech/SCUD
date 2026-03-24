"""Main dispatcher: routes to the correct dataloader based on config."""

from __future__ import annotations

from omegaconf import DictConfig
from torch.utils.data import DataLoader


def get_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader] | tuple[DataLoader | None, DataLoader | None]:
    """Build train/test dataloaders based on ``cfg.data.data``."""
    from scud.data.image import get_img_dataloaders, image_data_name_dict
    from scud.data.protein import get_protein_dataloaders, protein_data_name_dict
    from scud.data.text import get_text_dataloaders, get_tokenizer, text_data_name_dict

    if cfg.data.data in image_data_name_dict:
        return get_img_dataloaders(cfg)
    elif cfg.data.data in text_data_name_dict:
        tokenizer = get_tokenizer(cfg)
        return get_text_dataloaders(cfg, tokenizer)
    elif cfg.data.data in protein_data_name_dict:
        return get_protein_dataloaders(cfg)
    else:
        all_supported = (
            list(image_data_name_dict) + list(text_data_name_dict) + list(protein_data_name_dict)
        )
        raise ValueError(f"Unknown dataset '{cfg.data.data}'. Supported: {all_supported}")
