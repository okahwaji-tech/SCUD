"""Sampling script for trained SCUD/Masking/Classical diffusion models."""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image

from scud.classical_diffusion import ClassicalDiffusion
from scud.masking_diffusion import MaskingDiffusion
from scud.scud import SCUD
from scud.scud_tcr import SCUD_TCR

MODEL_CLASS_MAP: dict[str, type] = {
    "SCUD": SCUD,
    "SCUD_TCR": SCUD_TCR,
    "Masking": MaskingDiffusion,
    "Classical": ClassicalDiffusion,
}


def _resolve_checkpoint(restart_path: str) -> str:
    """Find the most recent .ckpt file in a checkpoint folder."""
    ckpt_dir = f"checkpoints/{restart_path}"
    candidates = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    return max(candidates, key=os.path.getmtime)


def _load_model(cfg: DictConfig) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    if not cfg.model.restart:
        raise ValueError("model.restart must be set to a checkpoint folder name.")
    ckpt_path = _resolve_checkpoint(cfg.model.restart)
    model_cls = MODEL_CLASS_MAP[cfg.model.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: torch.nn.Module = model_cls.load_from_checkpoint(ckpt_path, map_location=device)  # type: ignore[attr-defined,assignment]
    model.eval()
    model.to(device)
    return model


def _save_image_samples(samples: torch.Tensor, num_classes: int, output_dir: Path) -> None:
    """Save generated image samples as a PNG grid."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Normalize to [0, 1]
    images = samples.float() / (num_classes - 1)
    out_path = output_dir / "samples.png"
    nrow = int(math.ceil(math.sqrt(images.shape[0])))
    save_image(images, str(out_path), nrow=nrow)
    print(f"Saved image samples to {out_path}")


def _save_protein_samples(samples: torch.Tensor, output_dir: Path) -> None:
    """Save generated protein samples as a FASTA file."""
    from evodiff.utils import Tokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer()
    out_path = output_dir / "samples.fasta"
    lines: list[str] = []
    for i, seq_tensor in enumerate(samples):
        header = f">sample_{i}"
        aa_seq = tokenizer.untokenize(seq_tensor)
        lines.append(header)
        lines.append(aa_seq)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved protein samples to {out_path}")


def run_sampling(cfg: DictConfig) -> None:
    """Generate samples from a trained diffusion model checkpoint.

    Config keys used:
        model.restart: Checkpoint folder name (required).
        model.model: Model type (SCUD, Masking, Classical).
        sampling.num_samples: Number of samples to generate (default 16).
        sampling.output_dir: Output directory (default "samples/").
        sampling.temperature: Sampling temperature (default 1.0).
        sampling.gen_trans_step: Number of denoising steps.
        data.data: Dataset name (determines output format).
    """
    num_samples = OmegaConf.select(cfg, "sampling.num_samples", default=16)
    output_dir = Path(OmegaConf.select(cfg, "sampling.output_dir", default="samples/"))
    temperature = OmegaConf.select(cfg, "sampling.temperature", default=1.0)
    gen_trans_step = OmegaConf.select(cfg, "sampling.gen_trans_step", default=2048)

    model = _load_model(cfg)
    device = next(model.parameters()).device

    # Get a reference sample shape from the model's stored sample_x or infer from config
    # For now, we need to get the data shape from the dataset
    from scud.data import get_dataloaders

    _, test_dataloader = get_dataloaders(cfg)
    assert test_dataloader is not None
    sample_batch = next(iter(test_dataloader))
    if isinstance(sample_batch, tuple):
        sample_x, sample_a = sample_batch
    elif isinstance(sample_batch, dict):
        sample_x = sample_batch["input_ids"]
        sample_a = sample_batch["attention_mask"]
    else:
        sample_x = sample_batch
        sample_a = None

    # Generate initial noise from stationary distribution
    with torch.inference_mode():
        p = model.get_stationary()  # type: ignore[operator]
        n_elements = sample_x.shape[1:].numel()
        noise_samples = torch.multinomial(p, num_samples=num_samples * n_elements, replacement=True)
        init_noise = noise_samples.reshape((num_samples,) + sample_x.shape[1:]).to(device)

        attn_mask = None
        if sample_a is not None:
            attn_mask = sample_a[:1].repeat(num_samples, *[1] * (sample_a.dim() - 1)).to(device)

        # Run sampling
        print(f"Generating {num_samples} samples with {gen_trans_step} denoising steps...")
        images = model.sample_sequence(  # type: ignore[operator]
            init_noise,
            attn_mask,
            n_T=gen_trans_step,
            stride=1,
            temperature=temperature,
        )

    if not images:
        print("Warning: sampling returned no results.")
        return

    final_samples = images[-1]

    # Save based on data type
    is_protein = cfg.data.data in ("uniref50",)
    if is_protein:
        _save_protein_samples(final_samples, output_dir)
    else:
        _save_image_samples(final_samples, model.num_classes, output_dir)  # type: ignore[arg-type]
