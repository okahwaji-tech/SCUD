"""Classical continuous-time discrete diffusion model.

Implements the standard (non-schedule-conditioned) continuous-time discrete
diffusion baseline that uses the matrix exponential exp(L * t) as the
transition operator. Serves as the gamma -> 1 limit of SCUD.

Reference: "Why Masking Diffusion Works" (NeurIPS 2025), Section 3.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .continuous_time_diffusion import ContinuousTimeDiffusion
from .utils import convert_to_probs, get_inf_gen, kls

logger = logging.getLogger(__name__)


class ClassicalDiffusion(ContinuousTimeDiffusion):
    """Classical continuous-time discrete diffusion (CTDD/SEDD baseline).

    Uses the matrix exponential of the infinitesimal generator L to define
    transition probabilities, without schedule conditioning. This corresponds
    to the gamma -> 1 (code convention) limit of SCUD.

    Args:
        x0_model_class: Neural network class for the denoiser.
        nn_params: Constructor kwargs for the denoiser network.
        num_classes: Number of discrete token classes.
        forward_kwargs: Forward process specification (type, bandwidth, etc.).
        schedule_type: Noise schedule type.
        logistic_pars: If True, use logistic parameterization.
    """

    # Type annotations for register_buffer tensors
    L: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    eigenvectors_inv: torch.Tensor

    def __init__(
        self,
        x0_model_class: type,
        nn_params: dict[str, object],
        num_classes: int = 10,
        forward_kwargs: dict[str, object] | None = None,
        schedule_type: str = "cos",
        logistic_pars: bool = False,
        **kwargs: object,
    ) -> None:
        if forward_kwargs is None:
            forward_kwargs = {"type": "uniform"}
        # Precalculate betas, define model_predict, p_sample
        super().__init__(
            x0_model_class, nn_params, num_classes, schedule_type, logistic_pars, **kwargs  # type: ignore[arg-type]
        )
        self.save_hyperparameters(ignore=["x0_model_class"])

        # Get L
        L = get_inf_gen(forward_kwargs, num_classes)
        self.register_buffer("L", L)
        eigenvalues, eigenvectors = torch.linalg.eig(L.double())
        eigenvalues[torch.real(eigenvalues) > -self.eps] = 0
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        self.register_buffer("eigenvalues", eigenvalues)
        self.register_buffer("eigenvectors", eigenvectors)
        self.register_buffer("eigenvectors_inv", eigenvectors_inv)

    def pre_configure_model(self, dataloader: object) -> None:
        self.calc_p0(dataloader)
        self.log_alpha, self.beta, *_ = self.get_beta_func(self.L.cpu(), self.p0.cpu(), "SEDD")

    def get_stationary(self) -> torch.Tensor:
        evals, evecs = torch.linalg.eig(self.L.T)
        norms_sq = torch.real(evals)
        stationary = evecs[:, torch.argmax(norms_sq)]
        assert torch.allclose(torch.imag(stationary), torch.tensor(0, dtype=self.L.dtype))
        stationary = torch.real(stationary)
        stationary = stationary * torch.sign(stationary)
        assert torch.all(stationary >= 0)
        return stationary / stationary.sum()

    def get_trans_mats_mvp(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute exp(L * log_alpha(t)) @ v via eigendecomposition.

        Args:
            t: Time values, shape (B,).
            v: Positive probability vectors, shape (B, ..., C).

        Returns:
            Result of matrix exponential applied to v, shape (B, ..., C).
        """
        dv = v.to(dtype=self.eigenvectors.dtype).reshape(v.shape[0], -1, v.shape[-1])
        diag = torch.exp(-self.log_alpha(t)[..., None] * self.eigenvalues)
        dv = dv @ self.eigenvectors
        dv = dv * diag.unsqueeze(-2)
        dv = dv @ self.eigenvectors_inv
        return F.relu(dv.double()).to(t.dtype).reshape(v.shape)  # negative values are errors

    def get_trans_mats_index(self, t: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
        """Compute rows of exp(L * log_alpha(t)) indexed by ind.

        Args:
            t: Time values, shape (B,).
            ind: Integer indices, shape (B, ...), values in [0, C).

        Returns:
            Transition probabilities, shape (B, ..., C).
        """
        dind = ind.reshape(ind.shape[0], -1)
        diag = torch.exp(-self.log_alpha(t)[..., None] * self.eigenvalues)
        dv = self.eigenvectors[dind, :]
        dv = dv * diag.unsqueeze(-2)
        dv = dv @ self.eigenvectors_inv
        return F.relu(dv.double()).to(t.dtype).reshape(ind.shape + (self.num_classes,))

    def get_kl_t1(self, x: torch.Tensor) -> torch.Tensor:
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        x_1 = torch.log(self.get_trans_mats_index(t, x) + self.eps)
        kl = kls(x_1, torch.log(self.get_stationary() + self.eps))
        return kl.mean()

    def x_t_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, S: torch.Tensor | None = None
    ) -> torch.Tensor:
        # forward process, x_0 is the clean input.
        probs = self.get_trans_mats_index(t, x_0)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = 1 / (-torch.log(noise))
        x_t = torch.argmax(probs * gumbel_noise, dim=-1)
        return x_t

    def r_posterior(
        self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, S: torch.Tensor | None
    ) -> torch.Tensor:  # returns backward inf_gen
        softmaxed = convert_to_probs(x_0, self.num_classes)  # bs, ..., num_classes
        p_y = self.get_trans_mats_mvp(t, softmaxed)
        p_xt = p_y.gather(-1, x_t.unsqueeze(-1)).squeeze(-1)
        if not torch.all(p_xt > 1000 * self.eps):
            err = torch.any(p_xt <= self.eps, dim=-1)
            logger.warning("Small p_xt: %s %s %s", t[err], p_xt[err], x_0[err])
        ratios = p_y / (p_xt[..., None] + self.eps)
        bwd_inf_gen = ((ratios * self.L.T[x_t, :]).transpose(0, -1) * self.beta(t)).transpose(0, -1)
        bwd_inf_gen.scatter_(-1, x_t.unsqueeze(-1), 0)  # set diag to 0
        return bwd_inf_gen

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, *args: object
    ) -> tuple[torch.Tensor, dict[str, float]]:
        t, _, x_t = self.sample_point(x, attn_mask)
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, attn_mask, None).to(torch.float32)
        true_r_posterior = self.r_posterior(x, x_t, t, None)
        pred_r_posterior = self.r_posterior(predicted_x0_logits, x_t, t, None)

        # get kls and loss
        kl = -(torch.xlogy(true_r_posterior, pred_r_posterior + self.eps) - pred_r_posterior) + (
            torch.xlogy(true_r_posterior, true_r_posterior) - true_r_posterior
        )
        kl = kl.sum(-1)
        if attn_mask is not None:
            kl = kl * attn_mask
        vb_loss = kl.mean() * self.t_max
        if attn_mask is not None:
            vb_loss = vb_loss / attn_mask.mean()

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss(reduction="none")(predicted_x0_logits, x)
        if attn_mask is not None:
            ce_loss = (ce_loss * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            ce_loss = ce_loss.mean()
        return vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.Tensor | None,
        noise: torch.Tensor,
        delta_t: float,
        S: torch.Tensor | None = None,
        temperature: float = 1,
    ) -> torch.Tensor:
        # predict prev(x_t) or x_{t-1}
        predicted_x0_logits = self.model_predict(x, t, attn_mask, None) / temperature
        bwd_inf_gen = self.r_posterior(predicted_x0_logits, x, t, None)
        bwd_inf_gen.scatter_(-1, x.unsqueeze(-1), 0)
        bwd_inf_gen.scatter_(-1, x.unsqueeze(-1), -bwd_inf_gen.sum(-1).unsqueeze(-1))

        x_t = F.one_hot(x, self.num_classes)
        trans_mat = x_t + delta_t * bwd_inf_gen
        # sample
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = 1 / (-torch.log(noise))
        sample = torch.argmax(trans_mat * gumbel_noise, dim=-1)
        return sample

    def sample_sequence(  # type: ignore[override]
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        n_T: int = 200,
        stride: int = 10,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        n_T = 1
        steps = 0  ## TODO fix sampling when there are masks
        images = []
        pbar = tqdm(
            torch.flip(torch.linspace(0, self.t_max, n_T, dtype=torch.float32), (0,)),
            position=0,
            leave=True,
        )
        for t_val in pbar:
            t = torch.tensor([t_val] * x.shape[0], device=x.device)
            x_next = self.p_sample(
                x, t, attn_mask, torch.rand((*x.shape, self.num_classes), device=x.device), 1 / n_T
            )
            x = x_next

            steps += 1
            if steps % stride == 0:
                images.append(x)
        if steps % stride != 0:
            images.append(x)
        return images
