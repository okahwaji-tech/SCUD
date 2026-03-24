"""Data preparation pipeline: auto-download datasets before training.

With HuggingFace datasets, most data downloads are handled automatically
by the `datasets` library. This module handles auxiliary files (BLOSUM62
matrix) that are not available as HF datasets.

Usage:
    Called automatically at the start of training, or manually via:
        scud-prepare --config-name=basic_protein
"""

from __future__ import annotations

import logging
import os
import urllib.request

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# BLOSUM62 matrix from evodiff GitHub repository
BLOSUM_URL = (
    "https://raw.githubusercontent.com/microsoft/evodiff/main/data/blosum62-special-MSA.mat"
)


def prepare_data(cfg: DictConfig) -> None:
    """Prepare auxiliary data files for the configured experiment.

    HuggingFace datasets (CIFAR-10, UniRef50, LM1B) are downloaded
    automatically by the `datasets` library on first use. This function
    only handles files not available through HF, such as the BLOSUM62
    substitution matrix needed for protein forward processes.

    Args:
        cfg: Hydra DictConfig with data and model configuration.
    """
    data_name = cfg.data.data

    if data_name == "uniref50":
        forward_type = cfg.model.forward_kwargs.get("type", "uniform")
        if forward_type == "blosum":
            data_dir = getattr(cfg.data, "data_dir", "data")
            prepare_blosum(data_dir)

    # All other datasets (CIFAR-10, MNIST, LM1B) are handled by HF datasets
    logger.info("Data preparation complete for '%s'.", data_name)


def prepare_blosum(data_dir: str) -> None:
    """Download BLOSUM62 substitution matrix from evodiff repository.

    Args:
        data_dir: Directory to save the matrix file.
    """
    blosum_path = os.path.join(data_dir, "blosum62-special-MSA.mat")

    if os.path.exists(blosum_path):
        logger.info("BLOSUM62 matrix already exists at %s, skipping.", blosum_path)
        return

    os.makedirs(data_dir, exist_ok=True)
    logger.info("Downloading BLOSUM62 matrix from evodiff repository...")

    try:
        urllib.request.urlretrieve(BLOSUM_URL, blosum_path)
        logger.info("BLOSUM62 matrix saved to %s", blosum_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download BLOSUM62 matrix from {BLOSUM_URL}. "
            f"Please download it manually and place at {blosum_path}. "
            f"Error: {e}"
        ) from e
