"""
Trajectory Consistency Regularization (TCR) utilities.

Provides the building blocks for Path B training:
  - Detached prediction and token sampling
  - SCUD backward transition (Eq. 21) for faithful state generation
  - Target noise level sampling
  - Gradient normalization between losses
"""

import torch
import torch.nn.functional as F

def detached_predict(model, x_t, t, attn_mask, S, tau=None):
    """
    Run a no-grad forward pass and sample discrete tokens.

    Args:
        model: SCUD model (or subclass) with model_predict method
        x_t: noisy input tokens [batch, ...]
        t: time values [batch]
        attn_mask: attention mask or None
        S: event counts [batch, ...]
        tau: holding times [batch, ...] or None (defaults to zeros)

    Returns:
        x0_hat: sampled discrete tokens [batch, ...] (detached)
        x0_logits: raw logits [batch, ..., num_classes] (detached)
    """
    with torch.no_grad():
        x0_logits = model.model_predict(x_t, t, attn_mask, S)
        x0_hat = torch.argmax(
            F.softmax(x0_logits, dim=-1)
            * (1 / (-torch.log(torch.rand_like(x0_logits).clamp(min=1e-9)))),
            dim=-1,
        )
    return x0_hat, x0_logits

def compute_kernel_consistency_score(x0_hat, x_t, S, K_powers, log_K_powers_max, clamp_min=-10.0):
    """
    Compute kernel-consistent self-correction score.

    Measures how consistent the model's prediction x0_hat is with the observed
    corruption x_t given event count S, using the kernel's own transition
    probabilities. Normalized by the best possible prediction at each (S, x_t).

    Called with Path A's (x0_hat, x_t, S) to produce conditioning and loss
    weights for Path B, where the model sees x0_hat at s=0 everywhere.

    Args:
        x0_hat: model predictions [batch, seq_len] (long)
        x_t: corrupted tokens [batch, seq_len] (long)
        S: event counts [batch, seq_len] (long) — from Path A corruption
        K_powers: precomputed K^i [num_powers, num_classes, num_classes]
        log_K_powers_max: precomputed log(max_v K^i[v, :]) [num_powers, num_classes]
        clamp_min: lower bound for normalized log score (default -10)

    Returns:
        score: normalized log consistency [batch, seq_len], in [clamp_min, 0]
        loss_weight: -score [batch, seq_len], in [0, -clamp_min]
    """
    S = S.long()
    x0_hat = x0_hat.long()
    x_t = x_t.long()

    # Clamp S to valid index range for K_powers
    S_clamped = S.clamp(0, K_powers.shape[0] - 1)

    # Raw transition probability: K^s[x0_hat, x_t] per position
    log_p = torch.log(K_powers[S_clamped, x0_hat, x_t].clamp(min=1e-9))

    # Best possible prediction at this (s, x_t) pair
    log_p_max = log_K_powers_max[S_clamped, x_t]

    # Normalize: 0 = best possible, negative = worse
    score = (log_p - log_p_max).clamp(min=clamp_min)

    loss_weight = -score

    return score, loss_weight

def grad_norm_weight(loss_main, loss_aux, parameters):
    """
    Compute scaling factor so both losses contribute equal gradient magnitude.

    Used for SCUD + TCR where Path A (ELBO) and Path B (unweighted CE) are
    on different scales. Not needed for SCUM + TCR where both paths use CE.

    Args:
        loss_main: primary loss scalar (Path A)
        loss_aux: auxiliary loss scalar (Path B)
        parameters: model parameters to compute gradients w.r.t.

    Returns:
        weight: scalar to multiply loss_aux by (detached)
    """
    params = [p for p in parameters if p.requires_grad]

    grad_main = torch.autograd.grad(
        loss_main, params, retain_graph=True, allow_unused=True
    )
    grad_aux = torch.autograd.grad(
        loss_aux, params, retain_graph=True, allow_unused=True
    )

    norm_main = torch.sqrt(
        sum((g.norm() ** 2 for g in grad_main if g is not None))
    )
    norm_aux = torch.sqrt(
        sum((g.norm() ** 2 for g in grad_aux if g is not None))
    )

    return (norm_main / (norm_aux + 1e-8)).detach()


