"""CLI entry points for SCUD training, sampling, and evaluation."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="basic")
def train(cfg: DictConfig) -> None:
    """Train a SCUD/Masking/Classical diffusion model."""
    from scud._train import run_training

    run_training(cfg)


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="basic")
def sample(cfg: DictConfig) -> None:
    """Generate samples from a trained checkpoint."""
    from scud._sample import run_sampling

    run_sampling(cfg)


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="basic")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained checkpoint on test data."""
    from scud._evaluate import run_evaluation

    run_evaluation(cfg)


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="basic")
def prepare(cfg: DictConfig) -> None:
    """Download and prepare datasets for the configured experiment."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from scud.data.prepare import prepare_data

    prepare_data(cfg)
