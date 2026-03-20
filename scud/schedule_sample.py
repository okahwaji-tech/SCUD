"""Schedule sampling utilities for discrete diffusion.

Provides functions to simulate the number of forward-process transitions
(jumps) at given times, for both discrete-time (Bernoulli) and
continuous-time (Poisson) formulations.

Reference: "Why Masking Diffusion Works" (NeurIPS 2025), Section 4.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def sample_n_transitions(beta_t: torch.Tensor, batch_size: int, times: torch.Tensor) -> torch.Tensor:
    """For a bunch of betas and times, simulate # transitions before
    time. Repeat batch_size # of times to get [times_dim] + batch_size.
    Note t=0 gives the number of transitions after 1 timestep.

    Example usage:
    sample_n_transitions(d3pm.beta_t, 7, torch.tensor([500, 999, 102]))
    >tensor([[ 1.,  6.,  0.],
             [ 0.,  8.,  0.],
             [ 1., 10.,  0.],
             [ 0.,  7.,  0.],
             [ 0.,  3.,  0.],
             [ 0.,  2.,  0.],
             [ 0.,  7.,  0.]], dtype=torch.float64)
    """
    t_shape = times.shape
    times = times.reshape(1, -1)
    beta_t = beta_t.reshape(1, 1, -1).repeat(batch_size, times.shape[1], 1)
    transitions = torch.bernoulli(beta_t)
    transitions = transitions.cumsum(-1)
    transitions = transitions[
        torch.arange(batch_size)[:, None],
        torch.arange(times.shape[1])[None, :],
        times.repeat(batch_size, 1),
    ]
    return transitions.reshape((batch_size,) + t_shape)


def sample_full_transitions(beta_t: torch.Tensor, batch_size: int) -> torch.Tensor:
    """For a bunch of betas, simulate # transitions at each timestep.

    Example usage:
    sample_n_transitions(d3pm.beta_t, 7)
    > 7 x 1000 tensor
    """
    beta_t = beta_t.reshape(1, -1).repeat(batch_size, 1)
    transitions = torch.bernoulli(beta_t)
    return transitions.bool()


def sample_n_transitions_cont(
    log_alpha: Callable[[torch.Tensor], torch.Tensor], batch_size: int, times: torch.Tensor
) -> torch.Tensor:
    """Continuous-time transition sampling via Poisson distribution.

    Samples S ~ Poisson(-log_alpha(t)) for each time in times, repeated
    batch_size times.

    Args:
        log_alpha: Function mapping time tensor to log-alpha values.
        batch_size: Number of independent samples per time.
        times: 1-D tensor of time values.

    Returns:
        Tensor of shape (batch_size, len(times)) with sampled transition counts.
    """
    times = times.reshape(-1)
    log_alpha_t = log_alpha(times).reshape(1, -1).repeat(batch_size, 1)
    transitions = torch.poisson(-log_alpha_t)
    return transitions
