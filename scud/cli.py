"""CLI entry points for SCUD training, sampling, and evaluation."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="basic")
def train(cfg: DictConfig) -> None:
    """Train a SCUD/Masking/Classical diffusion model."""
    from scud._train import run_training

    run_training(cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="basic")
def sample(cfg: DictConfig) -> None:
    """Generate samples from a trained checkpoint."""
    raise NotImplementedError(
        "Sampling CLI not yet implemented. Use model.sample_sequence() directly."
    )


@hydra.main(version_base=None, config_path="../configs", config_name="basic")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained checkpoint on test data."""
    raise NotImplementedError("Evaluation CLI not yet implemented.")
