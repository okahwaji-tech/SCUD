"""Neural network model registry and configuration.

Maps architecture names from config files to their implementing classes
and prepares constructor keyword arguments. Supports image models
(KingmaUNet) and protein sequence models (ByteNetLMTimeNew).

Reference: "Why Masking Diffusion Works" (NeurIPS 2025).
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from scud.protein_convnet import ByteNetLMTimeNew
from scud.unet import KingmaUNet

image_nn_name_dict = {
    "KingmaUNet": KingmaUNet,
}

protein_nn_name_dict = {"ConvNew": ByteNetLMTimeNew}


def get_model_setup(
    cfg: DictConfig, tokenizer: object | None = None
) -> tuple[type, dict[str, object]]:
    """Resolve the neural network class and its constructor parameters from config.

    Args:
        cfg: Hydra/OmegaConf configuration object containing model, data,
            and architecture sections.
        tokenizer: Optional tokenizer for text/protein models.

    Returns:
        Tuple of (nn_class, nn_params) where nn_class is the neural network
        class and nn_params is a dict of constructor keyword arguments.
    """
    schedule_conditioning = cfg.model.model in [
        "SCUD",
        "SCUD_TCR",
        "ScheduleCondition",
        "DiscreteScheduleCondition",
        "MaskingDiffusion",
    ]
    raw_params = cfg.architecture.nn_params
    nn_params: dict[str, object] = (
        dict(OmegaConf.to_container(raw_params, resolve=True))  # type: ignore[arg-type]
        if raw_params is not None
        else {}
    )
    if cfg.architecture.x0_model_class in image_nn_name_dict:  # noqa: RET503
        nn_params = {
            "n_channel": 1 if cfg.data.data == "MNIST" else 3,
            "N": cfg.data.N + (cfg.model.model == "MaskingDiffusion"),
            "n_T": cfg.model.n_T,
            "schedule_conditioning": schedule_conditioning,
            "s_dim": cfg.architecture.s_dim,
            **nn_params,
        }

        return image_nn_name_dict[cfg.architecture.x0_model_class], nn_params

    elif cfg.architecture.x0_model_class in protein_nn_name_dict:
        nn_params = {
            "n_tokens": cfg.data.N + (cfg.model.model == "MaskingDiffusion"),
            "schedule_conditioning": schedule_conditioning,
            **nn_params,
        }
        return protein_nn_name_dict[cfg.architecture.x0_model_class], nn_params
    raise ValueError(f"Unknown model class: {cfg.architecture.x0_model_class}")
