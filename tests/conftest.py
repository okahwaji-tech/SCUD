"""Shared test fixtures for SCUD test suite."""

from __future__ import annotations

from omegaconf import OmegaConf


def make_config(
    model: str = "SCUD",
    data: str = "CIFAR10",
    N: int = 4,
    x0_model_class: str = "KingmaUNet",
    **overrides: object,
) -> OmegaConf:
    """Create a minimal test config."""
    cfg = OmegaConf.create(
        {
            "data": {"data": data, "N": N},
            "model": {
                "model": model,
                "n_T": 10,
                "gamma": 0,
                "schedule_type": "cos",
                "forward_kwargs": {"type": "uniform"},
                "logistic_pars": False,
                "t_max": 0.999,
                "restart": False,
                "seed": 0,
            },
            "architecture": {
                "x0_model_class": x0_model_class,
                "s_dim": 8,
                "width": 8,
                "nn_params": {
                    "s_lengthscale": 50,
                    "time_lengthscale": 1,
                    "n_layers": 1,
                    "time_embed_dim": 0,
                    "not_logistic_pars": True,
                    "semb_style": "u_inject",
                    "s_embed_dim": 16,
                    "film": False,
                    "input_logits": False,
                    "first_mult": False,
                },
            },
            "train": {
                "batch_size": 2,
                "n_epoch": 1,
                "lr": 0.001,
                "grad_clip_val": 1,
                "weight_decay": 0,
                "accumulate": 1,
                "ema": False,
            },
            "sampling": {"gen_trans_step": 10},
        }
    )
    for key, val in overrides.items():
        OmegaConf.update(cfg, key, val)
    return cfg
