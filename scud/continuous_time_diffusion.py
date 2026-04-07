"""Base class for continuous-time discrete diffusion models.

Implements shared functionality for all continuous-time diffusion variants
(SCUD, ClassicalDiffusion, MaskingDiffusion): schedule construction,
model prediction with optional logistic parameterization, forward-process
sampling, and checkpoint loading.

Reference: "Why Masking Diffusion Works" (NeurIPS 2025), Section 3.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from scud.mutual_info_schedule import get_a_b_func_mi

from .schedule_sample import sample_n_transitions_cont
from .trainer import DiffusionTrainer

logger = logging.getLogger(__name__)


def get_betas(schedule_type: str) -> Callable[..., tuple[Callable[..., torch.Tensor], ...]]:
    """Return a factory function that produces log_alpha and beta schedule functions.

    Args:
        schedule_type: One of 'cos', 'linear', or 'mutual_information'.

    Returns:
        A callable that, given rate matrix and data distribution, returns
        (log_alpha, beta) schedule functions.
    """
    if schedule_type in ["cos", "linear"]:

        def get_funcs(L, p0, model="SEDD", scale=1, type_=None):
            if schedule_type == "cos":
                alpha = lambda t: 1 - torch.cos((1 - t) * torch.pi / 2)
                alpha_prime = lambda t: -torch.sin((1 - t) * torch.pi / 2) * torch.pi / 2
            if schedule_type == "linear":
                alpha = lambda t: 1 - t
                alpha_prime = lambda t: -1
            beta = lambda t: -scale * alpha_prime(t) / alpha(t)
            log_alpha = lambda t: scale * torch.log(alpha(t))
            return log_alpha, beta

    elif schedule_type in ["mutual_information"]:
        return get_a_b_func_mi
    return get_funcs


class ContinuousTimeDiffusion(DiffusionTrainer):
    """Base class for continuous-time discrete diffusion models.

    Provides the shared interface and utilities that SCUD, ClassicalDiffusion,
    and MaskingDiffusion build upon: noise schedule construction, neural
    network prediction (with optional logistic parameterization), and
    forward-process time/schedule sampling.

    Args:
        x0_model_class: Neural network class for the denoiser.
        nn_params: Keyword arguments passed to x0_model_class constructor.
        num_classes: Number of discrete token classes.
        schedule_type: Noise schedule type ('cos', 'linear', 'mutual_information').
        logistic_pars: If True, use logistic parameterization for predictions.
        t_max: Maximum diffusion time (slightly less than 1 for stability).
    """

    def __init__(
        self,
        x0_model_class: type,
        nn_params: dict[str, object],
        num_classes: int = 10,
        schedule_type: str = "cos",
        logistic_pars: bool = False,
        t_max: float = 0.999,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.save_hyperparameters(ignore=["x0_model_class"])
        self.hparams.update(x0_model_class=x0_model_class.__name__)
        self.x0_model = x0_model_class(**nn_params)
        self.eps = 1e-9
        self.num_classes = num_classes
        self.t_max = t_max
        self.logistic_pars = logistic_pars

        # Precalculate betas
        self.get_beta_func = get_betas(schedule_type)

    def get_stationary(self) -> torch.Tensor:
        """Return the stationary distribution of the forward process."""
        raise NotImplementedError

    def base_predict(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.Tensor | None,
        S: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the denoiser network on noisy data."""
        if tau is not None:
            result: torch.Tensor = self.x0_model(x_t, t, attn_mask, S, tau=tau).to(torch.float32)
        else:
            result = self.x0_model(x_t, t, attn_mask, S).to(torch.float32)
        return result

    def model_predict(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.Tensor | None,
        S: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = self.base_predict(x_t, t, attn_mask, S, tau=tau)
        if not self.logistic_pars:
            return pred
        else:
            loc = pred[..., 0].unsqueeze(-1)
            log_scale = pred[..., 1].unsqueeze(-1)
            inv_scale = torch.exp(-(log_scale - 2.0))
            bin_width = 2.0 / (self.num_classes - 1.0)
            bin_centers = torch.linspace(-1.0, 1.0, self.num_classes).to(pred.device)
            bin_centers = bin_centers - loc
            log_cdf_min = torch.nn.LogSigmoid()(inv_scale * (bin_centers - 0.5 * bin_width))
            log_cdf_max = torch.nn.LogSigmoid()(inv_scale * (bin_centers + 0.5 * bin_width))
            logits = log_cdf_max + torch.log1p(-torch.exp(log_cdf_min - log_cdf_max) + self.eps)
            result: torch.Tensor = logits
            return result

    def q_posterior_logits(
        self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, S: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute log-probabilities of the denoising posterior."""
        raise NotImplementedError

    def x_t_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, S: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample from the forward process at time t."""
        raise NotImplementedError

    log_alpha: Callable[..., torch.Tensor]  # set dynamically in subclasses
    beta: Callable[..., torch.Tensor]  # set dynamically in subclasses

    def sample_point(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, rand_shape: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.rand(x.shape[0], device=x.device) * self.t_max
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)  # type: ignore[arg-type]
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        x_t = self.x_t_sample(
            x,
            t,
            torch.rand(
                (*x.shape, rand_shape if rand_shape is not None else self.num_classes),
                device=x.device,
            ),
            S,
        )
        return t, S, x_t

    def load_state_dict(  # type: ignore[override]
        self, state_dict: dict[str, torch.Tensor], strict: bool = False
    ) -> tuple[list[str], list[str]]:
        # Call the parent class's load_state_dict method
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)

        # Load the additional state dict variables
        for key in [
            "p0_inds",
            "p0_rank",
            "K",
            "L",
            "K_coo",
            "K_csc",
            "K_T",
            "L_T",
            "stat",
            "stationary",
        ]:
            if key in state_dict:
                setattr(self, key, state_dict[key])
                if key in unexpected_keys:
                    unexpected_keys.remove(key)

        if strict:
            error_msgs = []
            if len(unexpected_keys) > 0:
                error_msgs.append(
                    "unexpected key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in unexpected_keys)
                    )
                )
            if len(missing_keys) > 0:
                error_msgs.append(
                    "missing key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in missing_keys)
                    )
                )

            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        self.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )

        return missing_keys, unexpected_keys

    @classmethod
    def load_from_checkpoint(  # type: ignore[override]
        cls, checkpoint_path: str, map_location: str | torch.device | None = None, **kwargs: object
    ) -> ContinuousTimeDiffusion:
        logger.info("Loading checkpoint ...")
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        hparams = checkpoint["hyper_parameters"]

        # Get the x0_model_class
        x0_model_class_map: dict[str, type] = {}
        try:
            from scud.unet import KingmaUNet

            x0_model_class_map["KingmaUNet"] = KingmaUNet
        except ImportError:
            pass
        try:
            from scud.protein_convnet import ByteNetLMTimeNew

            x0_model_class_map["ByteNetLMTimeNew"] = ByteNetLMTimeNew
        except ImportError:
            pass
        try:
            from scud.dit_vision import DiT_Llama

            x0_model_class_map["DiT_Llama"] = DiT_Llama
        except ImportError:
            pass
        try:
            from scud.dit_text import DIT

            x0_model_class_map["DIT"] = DIT
        except ImportError:
            pass
        cls_name = hparams["x0_model_class"]
        if cls_name not in x0_model_class_map:
            available = list(x0_model_class_map.keys())
            raise ImportError(
                f"x0_model_class '{cls_name}' not found. "
                f"Available classes: {available}. "
                f"Ensure the corresponding module is installed."
            )
        x0_model_class = x0_model_class_map[cls_name]
        hparams["x0_model_class"] = x0_model_class

        # Create model
        logger.info("Setting up class ...")
        model = cls(**hparams)
        logger.info("Loading params ...")
        model.load_state_dict(checkpoint["state_dict"])
        return model
