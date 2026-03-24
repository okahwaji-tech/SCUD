"""Utility functions for discrete diffusion models.

Provides helpers for KL divergence computation, distribution conversion,
infinitesimal generator construction, schedule sorting, and numerical
stability. Used throughout the SCUD codebase.

Reference: "Why Masking Diffusion Works" (NeurIPS 2025).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F


def _at(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Index into a 3-D tensor using time indices and class indices.

    Args:
        a: Tensor of shape (T, C, C) to index into.
        t: 1-D time index tensor of shape (B,).
        x: Integer class indices, shape (B, ...), values in [0, num_classes).

    Returns:
        Indexed tensor of shape (B, ..., C).
    """
    # t is 1-d, x is integer value of 0 to num_classes - 1
    bs = t.shape[0]
    t = t.reshape((bs, *[1] * (x.dim() - 1)))
    return a[t, x, :]


def kls(dist1: torch.Tensor, dist2: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Compute KL divergence between two distributions along the last dimension.

    Args:
        dist1: Logits for the target distribution, shape (..., C).
        dist2: Logits for the predicted distribution, shape (..., C).
        eps: Unused, kept for API compatibility.

    Returns:
        KL divergence per element, shape (...).
    """
    out = F.kl_div(
        torch.log_softmax(dist2, dim=-1),
        torch.log_softmax(dist1, dim=-1),
        log_target=True,
        reduction="none",
    ).sum(-1)
    return out


def convert_to_distribution(x_0: torch.Tensor, num_classes: int, eps: float) -> torch.Tensor:
    """Convert data to log-probability representation.

    Args:
        x_0: Input data, either integer class indices or logits.
        num_classes: Number of discrete classes.
        eps: Small constant for numerical stability in log.

    Returns:
        Log-probabilities tensor of shape (..., num_classes).
    """
    # returns log probs of x_0 as a distribution
    if x_0.dtype in (torch.int64, torch.int32):
        x_0_logits = torch.log(torch.nn.functional.one_hot(x_0, num_classes) + eps)
    else:
        x_0_logits = x_0.clone()
    return x_0_logits


def convert_to_probs(x_0: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert data to probability representation.

    Args:
        x_0: Input data, either integer class indices or logits.
        num_classes: Number of discrete classes.

    Returns:
        Probability tensor of shape (..., num_classes).
    """
    # returns probs of x_0 as a distribution. input is either indices or logits
    if x_0.dtype in (torch.int64, torch.int32):
        x_0_probs = torch.nn.functional.one_hot(x_0, num_classes)
    else:
        x_0_probs = torch.softmax(x_0.clone(), dim=-1)
    return x_0_probs


def get_inf_gen(
    forward_kwargs: dict[str, object], num_classes: int, data_dir: str = "data"
) -> torch.Tensor:  # noqa: C901
    """Construct the infinitesimal generator matrix L for the forward process.

    Builds the rate matrix that defines the continuous-time Markov chain used
    as the forward noising process. Supports uniform, Gaussian, and BLOSUM
    transition structures (Section 3 of the paper).

    Args:
        forward_kwargs: Dictionary specifying the forward process type and
            parameters. Must contain 'type' key ('uniform', 'gaussian', or
            'blosum'). May contain 'bandwidth', 'beta', 'alpha', 'make_sym',
            'normalize'/'normalized' depending on type.
        num_classes: Number of discrete classes (vocabulary size).
        data_dir: Directory containing auxiliary data files (e.g., BLOSUM matrix).

    Returns:
        Rate matrix L of shape (num_classes, num_classes) with rows summing to zero.
    """
    if forward_kwargs["type"] == "uniform":
        L = torch.ones(num_classes, num_classes) / (num_classes - 1)
        L.diagonal().fill_(-1)
    elif forward_kwargs["type"] == "gaussian":
        bandwidth = forward_kwargs["bandwidth"]
        range_ = torch.arange(num_classes)
        diff_mat = (range_[:, None] - range_[None, :]) ** 2
        bw = float(bandwidth)  # type: ignore[arg-type]
        L = torch.exp(-diff_mat / (2 * (bw * num_classes) ** 2))
        L = L / (L.sum(-1).max() - 1)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    elif forward_kwargs["type"] == "blosum":
        from evodiff.utils import Tokenizer

        tokenizer = Tokenizer()
        # from https://web.expasy.org/protscale/pscale/A.A.Swiss-Prot.html
        aa_freq = (
            np.array(
                [
                    8.25,
                    5.53,
                    4.06,
                    5.45,
                    1.37,
                    3.93,
                    6.75,
                    7.07,
                    2.27,
                    5.96,
                    9.66,
                    5.84,
                    2.42,
                    3.86,
                    4.70,
                    6.56,
                    5.34,
                    1.08,
                    2.92,
                    6.87,
                ]
                + 11 * [0]
            )
            / 100
        )
        blosum_alphabet = np.array(list("ARNDCQEGHILKMFPSTWYVBZXJOU-"))
        tok_alphabet = np.array(tokenizer.alphabet)
        import os

        with open(os.path.join(data_dir, "blosum62-special-MSA.mat")) as f:
            load_matrix = np.array(
                [line.split()[1:] for line in f if line[0] in blosum_alphabet], dtype=int
            )
        map_ = blosum_alphabet[:, None] == tok_alphabet[None, :]
        blosum_matrix = np.zeros((len(tok_alphabet), len(tok_alphabet)))
        for i, ind_i in enumerate(np.argmax(map_, axis=1)):
            for j, ind_j in enumerate(np.argmax(map_, axis=1)):
                blosum_matrix[ind_i, ind_j] = load_matrix[i, j]
        # X_ij = BLOSUM_ij * p(aa_j) = p(aa_j | aa_i)
        cond_liks = (2.0 ** (blosum_matrix / 2)) * aa_freq[None, :]
        cond_liks = cond_liks ** float(forward_kwargs["beta"])  # type: ignore[arg-type]
        cond_liks = cond_liks / cond_liks.sum(-1)[:, None]
        L = cond_liks - np.eye(len(cond_liks))
        # break up
        eig_vals, V = np.linalg.eig(cond_liks[:20, :20])
        V_inv = np.linalg.inv(V)

        # alpha
        alpha = float(forward_kwargs["alpha"])  # type: ignore[arg-type]
        evals = (eig_vals**alpha - 1)[None, :] / alpha if alpha > 0 else np.log(eig_vals)
        L[:20, :20] = (V * evals) @ V_inv
        L[20:] *= -np.diagonal(L).min()
        L[L < 0] = 0
        L = torch.tensor(L).float()
        range_ = torch.arange(num_classes)
        L[range_, range_] = -L.sum(-1)
    if forward_kwargs.get("make_sym"):
        L = (L + L.T) / 2
        range_ = torch.arange(num_classes)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    if ("normalize" in forward_kwargs and forward_kwargs["normalize"]) or (
        "normalized" in forward_kwargs and forward_kwargs["normalized"]
    ):
        L = L / (-L.diagonal()[:, None])
        range_ = torch.arange(num_classes)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    return L


def get_sort_S(
    S: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort a schedule tensor in descending order and return inverse permutation.

    Args:
        S: Schedule tensor of arbitrary shape.

    Returns:
        Tuple of (S_sort, sort_indices, unsort_indices) where S_sort is the
        sorted tensor reshaped to original shape, sort_indices maps original
        to sorted positions, and unsort_indices inverts the sort.
    """
    S_flat, sort = torch.sort(S.flatten(), descending=True)
    S_sort = S_flat.reshape(S.shape)
    unsort = torch.zeros_like(sort)
    unsort[sort] = torch.arange(len(S_flat), device=S_flat.device)
    return S_sort, sort, unsort


def get_counts_S_flat(S_flat: torch.Tensor) -> torch.Tensor:
    """Compute cumulative counts of unique values in a flattened schedule.

    Args:
        S_flat: 1-D tensor of non-negative integer schedule values.

    Returns:
        Cumulative count tensor where entry i gives the number of elements >= i.
    """
    unique, counts = torch.unique(torch.clamp(S_flat, min=0), return_counts=True)
    full_counts = torch.zeros(unique.max() + 1, device=unique.device, dtype=torch.long)
    full_counts[unique] = counts
    return full_counts.flip(0).cumsum(0)


def _pad(tokenized: list[torch.Tensor], value: float, dim: int = 2) -> torch.Tensor:
    """Pad a list of tokenized sequences to the same length.

    Args:
        tokenized: List of tokenized sequence tensors.
        value: Padding value to fill shorter sequences.
        dim: Dimensionality of output (2 for indices, 3 for one-hot).

    Returns:
        Padded tensor of shape (batch_size, max_len) or (batch_size, max_len, C).
    """
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    if dim == 3:  # dim = 3 (one hot)
        categories = tokenized[0].shape[-1]
        output = torch.zeros((batch_size, max_len, categories)) + value
        for row, t in enumerate(tokenized):
            output[row, : len(t), :] = t
    elif dim == 2:  # dim = 2 (tokenized)
        output = torch.zeros((batch_size, max_len)) + value
        for row, t in enumerate(tokenized):
            output[row, : len(t)] = t
    else:
        print("padding not supported for dim > 3")
    return output


def sample_index_S(S: torch.Tensor) -> tuple[int, ...]:
    """Sample a multidimensional index from a schedule tensor as a probability.

    Args:
        S: Non-negative schedule tensor used as sampling weights.

    Returns:
        Tuple of integer indices into S.

    Raises:
        ValueError: If any entry in S is negative.
    """
    # Flatten the array
    S_flat = S.flatten()

    # Ensure all values are non-negative
    if torch.any(S_flat < 0):
        raise ValueError("All entries in S must be non-negative for probability sampling")

    # Sample an index
    sampled_flat_index = torch.multinomial(S_flat, num_samples=1)

    # Convert the flat index back to multidimensional index
    sampled_index = np.unravel_index(int(sampled_flat_index.item()), S.shape)

    return tuple(int(x) for x in sampled_index)


def log1p(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 + x) for values near -1.

    Uses an alternative formula for x < -0.7 to avoid catastrophic
    cancellation in torch.log1p.

    Args:
        x: Input tensor.

    Returns:
        log(1 + x) computed with improved numerical stability.
    """
    result = torch.log1p(x)
    mask = x < -0.7
    if mask.any():
        x_neg = x[mask]
        neg_x = -x_neg
        inv_xp1 = 1.0 / (neg_x - 1.0)
        result[mask] = torch.log1p(inv_xp1) + torch.log(neg_x)
    return result
