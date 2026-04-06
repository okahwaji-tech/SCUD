"""
Trajectory Consistency Regularization (TCR) utilities.

Provides the building blocks for Path B training:
  - Detached prediction and token sampling
  - SCUD backward transition (Eq. 21) for faithful state generation
  - Target noise level sampling
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

def scud_backward_transition(x_t, x0_hat, K_powers, S, k, num_classes, eigenvectors, eigenvalues, eigenvectors_inv, eps=1e-9):
    """
    Eq. 21 from SCUD paper: sample x_{t-k} from the posterior.

    Produces states from the same distribution as inference by combining
    the forward likelihood K^k @ x_t with the backward prior K^{s-k,T} @ x̂_0.

    Args:
        x_t: current noisy tokens [batch, ...] (integer indices)
        x0_hat: detached model prediction [batch, ...] (integer indices)
        K_powers: precomputed K^i matrices [num_powers, num_classes, num_classes]
        S: current event counts [batch, ...]
        k: events to reverse [batch, ...]
        num_classes: vocabulary size
        eigenvectors, eigenvalues, eigenvectors_inv: eigen-decomposition of K
            for computing K^n @ v via spectral method
        eps: numerical stability

    Returns:
        x_unrolled: sampled tokens at noise level S-k [batch, ...]
    """
    # Likelihood: p(x_t | x_{t-k}) = K^k[x_t, :] (transpose convention)
    # K_powers is [num_powers, num_classes, num_classes]
    # We need K_powers[k, x_t, :] with per-position k
    fact1 = K_powers.swapaxes(1, 2)[k, x_t, :]  # [batch, ..., num_classes]

    # Prior: p(x_{t-k} | x_0) = K^{S-k} @ one_hot(x0_hat)
    x0_probs = F.one_hot(x0_hat.long(), num_classes).float()
    Smk = S - k  # remaining events after reversal
    fact2 = _eigenvector_mvp(Smk, x0_probs, eigenvectors, eigenvalues, eigenvectors_inv, eps)

    # Posterior: element-wise product, then normalize
    probs = fact1 * fact2
    probs = probs.clamp(min=0)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + eps)

    # Sample via Gumbel-max
    noise = torch.rand_like(probs).clamp(min=eps)
    gumbel_noise = 1 / (-torch.log(noise))
    x_unrolled = torch.argmax(probs * gumbel_noise, dim=-1)

    return x_unrolled

def _eigenvector_mvp(S, v, eigenvectors, eigenvalues, eigenvectors_inv, eps=1e-9):
    """
    Compute K^S @ v using eigen-decomposition: V @ diag(λ^S) @ V^{-1} @ v.

    This mirrors SCUD.get_trans_mats_mvp but operates on arbitrary inputs.
    """
    dv = v.to(dtype=eigenvectors.dtype).reshape(-1, v.shape[-1])
    diag = eigenvalues ** F.relu(S.flatten()[..., None])
    dv = dv @ eigenvectors
    dv = dv * diag
    dv = dv @ eigenvectors_inv
    return F.relu(dv.double()).to(torch.float32).reshape(v.shape)

def sample_s_low(S):
    """
    Sample target noise levels uniformly: s_low^d ~ Uniform(0, s_high^d).

    Args:
        S: current event counts [batch, ...] (integer)

    Returns:
        s_low: target event counts [batch, ...], 0 <= s_low < S per position
        k: number of events to reverse, k = S - s_low
    """
    # For positions where S=0, s_low=0 and k=0 (no reversal)
    s_low = (torch.rand_like(S.float()) * S.float()).long()
    k = S - s_low
    return s_low, k


