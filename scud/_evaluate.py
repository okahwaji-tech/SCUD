"""Evaluation script for trained SCUD/Masking/Classical diffusion models."""

from __future__ import annotations

import math

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from scud._sample import MODEL_CLASS_MAP, _resolve_checkpoint
from scud.data import get_dataloaders


def run_evaluation(cfg: DictConfig) -> None:
    """Evaluate a trained checkpoint on the test set.

    Computes:
        - VB loss (variational bound on NLL)
        - KL at t=1 (terminal KL divergence)
        - NLL = mean(vb_loss) + mean(kl_t1)
        - BPD = NLL / (D * log(2)) for image data

    Config keys used:
        model.restart: Checkpoint folder name (required).
        model.model: Model type (SCUD, Masking, Classical).
    """
    if not cfg.model.restart:
        raise ValueError("model.restart must be set to a checkpoint folder name.")

    ckpt_path = _resolve_checkpoint(cfg.model.restart)
    model_cls = MODEL_CLASS_MAP[cfg.model.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {ckpt_path}...")
    model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)  # type: ignore[attr-defined]
    model.eval()
    model.to(device)

    print("Loading test data...")
    _, test_dataloader = get_dataloaders(cfg)
    assert test_dataloader is not None

    vb_losses: list[float] = []
    kl_t1s: list[float] = []
    n_batches = 0

    print("Evaluating...")
    with torch.inference_mode():
        for batch in tqdm(test_dataloader, desc="Eval"):
            if isinstance(batch, tuple):
                x, attn_mask = batch
            elif isinstance(batch, dict):
                x, attn_mask = batch["input_ids"], batch["attention_mask"]
            else:
                x = batch
                attn_mask = None

            x = x.to(device)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            loss, info = model(x, attn_mask)
            kl_t1 = model.get_kl_t1(x)

            vb_losses.append(info["vb_loss"])
            kl_t1s.append(kl_t1.detach().item())
            n_batches += 1

    mean_vb = sum(vb_losses) / n_batches
    mean_kl_t1 = sum(kl_t1s) / n_batches
    nll = mean_vb + mean_kl_t1

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  VB Loss (E[L_01]):    {mean_vb:.4f}")
    print(f"  KL at t=1 (L_1):     {mean_kl_t1:.4f}")
    print(f"  NLL:                  {nll:.4f}")

    # Compute BPD for image data
    is_image = cfg.data.data not in ("uniref50",)
    if is_image:
        # Get data dimensions from a sample batch
        assert test_dataloader is not None
        sample_batch = next(iter(test_dataloader))
        if isinstance(sample_batch, tuple):
            sample_x = sample_batch[0]
        elif isinstance(sample_batch, dict):
            sample_x = sample_batch["input_ids"]
        else:
            sample_x = sample_batch
        total_dims = sample_x[0].numel()
        bpd = nll / (total_dims * math.log(2))
        print(f"  BPD:                  {bpd:.4f}  (D={total_dims})")

    print("=" * 50)
