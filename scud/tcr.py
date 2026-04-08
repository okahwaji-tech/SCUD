"""Trajectory Consistency Regularization (TCR) utilities for SM-DFM training.

Pure functions for detached prediction, SCUD backward transitions (Eq. 21),
and event-count sampling used by the trajectory consistency loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _eigenvector_mvp(
    S: torch.Tensor,
    v: torch.Tensor,
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors_inv: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute v @ K^S via eigendecomposition.

    Args:
        S: Per-element exponents, shape (...).
        v: Probability vectors, shape (..., C).
        eigenvectors: Eigenvector matrix V, shape (C, C).
        eigenvalues: Eigenvalues lambda, shape (C,).
        eigenvectors_inv: Inverse eigenvector matrix V^{-1}, shape (C, C).
        eps: Unused, kept for interface consistency.

    Returns:
        Result of v @ K^S, shape (..., C), clamped to non-negative.
    """
    dv = v.to(dtype=eigenvectors.dtype).reshape(-1, v.shape[-1])
    diag = eigenvalues ** F.relu(S.flatten()[..., None])
    dv = dv @ eigenvectors
    dv = dv * diag
    dv = dv @ eigenvectors_inv
    return F.relu(dv.double()).to(torch.float32).reshape(v.shape)


def detached_predict(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    attn_mask: torch.Tensor | None,
    S: torch.Tensor,
    tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a detached discrete prediction from the model via Gumbel-max.

    Args:
        model: SCUD model with a ``model_predict`` method.
        x_t: Noisy tokens, shape (B, ...).
        t: Diffusion time, shape (B,).
        attn_mask: Optional attention mask.
        S: Schedule counts, shape (B, ...).
        tau: Optional temperature tensor.

    Returns:
        Tuple of (x0_hat, x0_logits), both detached.
    """
    with torch.no_grad():
        x0_logits = model.model_predict(x_t, t, attn_mask, S, tau=tau)
    probs = F.softmax(x0_logits, dim=-1)
    noise = torch.rand_like(probs).clamp(min=1e-9)
    gumbel_noise = 1.0 / (-torch.log(noise))
    x0_hat = torch.argmax(probs * gumbel_noise, dim=-1)
    return x0_hat.detach(), x0_logits.detach()


def scud_backward_transition(
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    K_powers: torch.Tensor,
    S: torch.Tensor,
    k: torch.Tensor,
    num_classes: int,
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors_inv: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Sample x_{t-k} from the SCUD backward transition (Eq. 21).

    Computes p(x_{t-k} | x_t, x_0) proportional to K^k[x_t, :] * K^{S-k} @ one_hot(x0_hat).

    Args:
        x_t: Current noisy tokens, shape (B, ...).
        x0_hat: Predicted clean tokens, shape (B, ...).
        K_powers: Precomputed K^n matrices, shape (max_pow, C, C).
        S: Total transitions per element, shape (B, ...).
        k: Number of transitions to reverse, shape (B, ...).
        num_classes: Vocabulary size C.
        eigenvectors: Eigenvector matrix V, shape (C, C).
        eigenvalues: Eigenvalues, shape (C,).
        eigenvectors_inv: V^{-1}, shape (C, C).
        eps: Small constant for numerical stability.

    Returns:
        Sampled intermediate tokens x_{t-k}, shape (B, ...).
    """
    # Likelihood: K^k[x_t, :]
    fact1 = K_powers.swapaxes(1, 2)[k, x_t, :]

    # Prior: K^{S-k} @ one_hot(x0_hat)
    x0_probs = F.one_hot(x0_hat.long(), num_classes).float()
    fact2 = _eigenvector_mvp(S - k, x0_probs, eigenvectors, eigenvalues, eigenvectors_inv, eps)

    # Posterior: normalised product
    probs = (fact1 * fact2).clamp(min=0.0)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + eps)

    # Gumbel-max sampling
    noise = torch.rand_like(probs).clamp(min=1e-9)
    gumbel_noise = 1.0 / (-torch.log(noise))
    return torch.argmax(probs * gumbel_noise, dim=-1)


def sample_k(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the number of events to reverse per position.

    Draws k ~ Uniform(1, S) where S > 0, and k = 0 where S == 0.

    Args:
        S: Total transitions per element, shape (B, ...).

    Returns:
        Tuple of (s_low, k) where s_low = S - k, both long tensors.
    """
    s_low = (torch.rand_like(S.float()) * S.float()).long()
    k = S - s_low
    return s_low, k
