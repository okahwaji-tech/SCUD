"""Mutual information-based noise schedule construction.

Computes noise schedules (log_alpha, beta) by inverting a mutual information
criterion: the schedule is chosen so that MI(x_0; x_t) decreases linearly
in t. Supports both classical (continuous-time) and SCUD (schedule-conditioned)
formulations.

Reference: "Why Masking Diffusion Works" (NeurIPS 2025), Section 5.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from collections.abc import Callable

import numpy as np
import torch
from tqdm import tqdm

from scud.root_finder import newton_root_finder

logger = logging.getLogger(__name__)


def hash_matrix(matrix: torch.Tensor) -> str:
    """Compute SHA-256 hash of a tensor for caching purposes.

    Args:
        matrix: Tensor to hash.

    Returns:
        Hex digest string.
    """
    byte_string = matrix.cpu().numpy().tobytes()
    hash_object = hashlib.sha256()
    hash_object.update(byte_string)
    return hash_object.hexdigest()


def try_load(
    func: Callable[[], torch.Tensor], fname: str, cache_dir: str = "data/save_alphas/"
) -> torch.Tensor:
    """Load a cached tensor from disk, or compute and save it.

    Args:
        func: Zero-argument callable that computes the tensor if not cached.
        fname: Filename for the cached .npy file.
        cache_dir: Directory for cached files.

    Returns:
        The loaded or computed tensor.
    """
    os.makedirs(cache_dir, exist_ok=True)
    if fname in os.listdir(cache_dir):
        logger.info("Loading alphas. Note: I hope p0 is similar to before!")
        val = np.load(cache_dir + fname)
        val = torch.tensor(val)
    else:
        val = func()  # - log_alpha_naive(base_ts)
        np.save(cache_dir + fname, val.cpu().numpy())
    return val


def get_a_b_func_cont(L: torch.Tensor, p0: torch.Tensor, **kwargs: object) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Construct MI-based noise schedule for classical continuous-time diffusion.

    Inverts the mutual information I(x_0; x_t) = 1 - t criterion by
    computing the matrix exponential exp(L * alpha) and finding alpha(t)
    via Newton root finding. Caches results for large vocabularies.

    Args:
        L: Infinitesimal generator matrix, shape (N, N).
        p0: Data distribution, shape (N,).

    Returns:
        Tuple of (log_alpha, beta, mi) functions mapping time tensors to
        their respective schedule values.
    """
    N = len(p0)
    ent_p0 = -torch.xlogy(p0, p0).sum()
    evals, V = torch.linalg.eig(L.double())
    evals[torch.real(evals) > -1e-6] = 0
    second_eval = torch.real(evals)
    second_eval = -second_eval.sort().values[-2]
    V_inv = torch.linalg.inv(V)

    def mi(t, t_shift=1):
        """t_shift is 1-t"""
        evals_skew = torch.exp(t[:, None, None] * evals[None, None, :])
        too_big = torch.real(evals_skew * evals_skew.conj()) > 1
        evals_skew = torch.where(too_big, 1, evals_skew)
        mat = torch.where(
            (t > 1e-5)[:, None, None],
            torch.real((V[None, :, :] * evals_skew) @ V_inv),
            torch.eye(len(L)) + t[:, None, None] * L + 0.5 * (t**2)[:, None, None] * (L @ L),
        )  # stable for small t
        p = p0[None, :, None] * mat
        p = torch.where(p < 1e-12, 0, p)
        mi_m1 = (torch.xlogy(p, p).sum(-1) - torch.xlogy(p.sum(-2), p.sum(-2))).sum(-1) / ent_p0
        return mi_m1 + t_shift

    base_ts = torch.linspace(0.0, 0.9991, 1000000)

    def log_alpha_naive(ts, bs=1000):
        batches = [ts.double()[i * bs : (i + 1) * bs] for i in range(math.ceil(len(ts) / bs))]
        guess_ts = torch.tensor([batch[len(batch) // 2] for batch in batches])
        guesses = newton_root_finder(mi, 1 / second_eval, guess_ts, print_=False)
        out = [
            newton_root_finder(mi, guess, batch, print_=False, max_iter=20)
            for batch, guess in tqdm(list(zip(batches, guesses, strict=True)))
        ]
        out = torch.concat(out)
        return -torch.where(out > 1e-8, out, 1e-8)

    hash_mat = hash_matrix(L)
    if N > 200:
        try_load(lambda: L, f"{hash_mat}_mat.npy")
        base_alphas = try_load(lambda: -log_alpha_naive(base_ts), f"{hash_mat}.npy")
    else:
        base_alphas = -log_alpha_naive(base_ts)

    def log_alpha(ts):
        closest_index = torch.searchsorted(base_ts, ts.to("cpu"))
        best_guess_l = base_alphas[closest_index]
        out = best_guess_l
        return -torch.where(out > 1e-8, out, 1e-8).to(ts.device).to(ts.dtype)

    def naive_beta(ts, bs=10000):
        batches = [ts.double()[i * bs : (i + 1) * bs] for i in range(math.ceil(len(ts) / bs))]
        grad = []
        for batch in batches:
            out_tensor = -log_alpha(batch.to("cpu"))
            grad.append(-1 / torch.func.grad(lambda o: mi(o).sum())(out_tensor))
        return torch.cat(grad).to(ts.device).to(ts.dtype)

    if N > 100:
        base_betas = try_load(lambda: naive_beta(base_ts), f"{hash_mat}_beta.npy")
    else:
        base_betas = naive_beta(base_ts)

    def beta(ts):
        closest_index = torch.searchsorted(base_ts, ts.to("cpu"))
        best_guess_l = base_betas[closest_index]
        out = best_guess_l
        return out.to(ts.device).to(ts.dtype)

    return log_alpha, beta, mi


def get_a_b_func_sc(
    K: torch.Tensor,
    p0: torch.Tensor,
    precompute_mis: list[float] | torch.Tensor | None = None,
    second_eval: float | torch.Tensor | None = None,
    **kwargs: object,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    torch.Tensor,
]:
    """Construct MI-based noise schedule for SCUD (schedule-conditioned diffusion).

    Computes the Poisson-weighted mutual information I(x_0; x_S) where
    S ~ Poisson(lambda), and inverts I = 1 - t to find the schedule
    lambda(t). Uses precomputed MI values for each integer n and Newton
    root finding for inversion.

    Args:
        K: Transition kernel matrix, shape (N, N).
        p0: Data distribution, shape (N,).
        precompute_mis: Optional precomputed MI values for each n.
        second_eval: Optional second-largest eigenvalue magnitude of K - I.

    Returns:
        Tuple of (log_alpha, beta, mi, precompute_mis) where the first three
        are callable schedule functions and precompute_mis can be reused.
    """
    if second_eval is None:
        L_ish = K.double() - torch.eye(len(K), dtype=torch.float64)
        evals, V = torch.linalg.eig(L_ish)
        evals[torch.real(evals) > -1e-6] = 0
        second_eval = torch.real(evals)
        second_eval = -second_eval.sort().values[-2]
    max_n = int(40 / second_eval)
    ent_p0 = -torch.xlogy(p0, p0).sum()
    if precompute_mis is None:
        V_inv = torch.linalg.inv(V)

        def mi_p(n):
            if n > 0:
                mat = torch.real((V * ((1 + evals[None, :]) ** n)) @ V_inv)
            else:
                mat = torch.eye(len(K), dtype=K.dtype)
            p = p0[:, None] * mat
            p = torch.where(p < 0, 0, p)
            return (torch.xlogy(p, p).sum(-1) - torch.xlogy(p.sum(-2), p.sum(-2))).sum(
                -1
            ) / ent_p0 + 1

        precompute_mis = [mi_p(n) for n in range(max_n)]
    precompute_mis = torch.tensor(precompute_mis).double()
    precompute_mis = torch.maximum(precompute_mis, torch.zeros(1))
    max_n = int((precompute_mis > 1e-7).sum())
    precompute_mis = torch.cummin(precompute_mis[:max_n], 0)[0]
    range_ = torch.arange(max_n, dtype=p0.dtype)
    precompute_log_factorial = torch.lgamma(1 + range_)

    def mi(t, t_shift=1.0):
        log_probs = range_ * torch.log(t[:, None] + 1e-6) - t[:, None] - precompute_log_factorial
        return torch.exp(log_probs) @ precompute_mis - 1.0 + t_shift

    base_ts = torch.linspace(0.0, 0.9991, 1000000)

    def log_alpha_naive(ts):
        out = newton_root_finder(mi, max_n / 20, ts.double())
        return -torch.where(out > 1e-6, out, 1e-6).float()

    hash_mat = hash_matrix(K)
    if len(K) > 200:
        try_load(lambda: K, f"{hash_mat}_mat.npy")
        base_alphas = try_load(lambda: -log_alpha_naive(base_ts), f"scud_{hash_mat}.npy")
    else:
        base_alphas = -log_alpha_naive(base_ts)
    base_alphas = torch.cummax(base_alphas, 0)[0]  # fix any errors

    def log_alpha(ts):
        closest_index = torch.searchsorted(base_ts, ts.to("cpu"))
        best_guess_l = base_alphas[closest_index]
        out = best_guess_l
        return -torch.where(out > 1e-6, out, 1e-6).to(ts.device).to(ts.dtype)

    def naive_beta(ts, bs=10000):
        out_tensor = -log_alpha(ts.to("cpu").double())
        batches = [ts[i * bs : (i + 1) * bs].double() for i in range(math.ceil(len(ts) / bs))]
        grad = []
        for batch in batches:
            out_tensor = -log_alpha(batch.to("cpu"))
            grad.append(-1 / torch.func.grad(lambda o: mi(o).sum())(out_tensor))
        return torch.cat(grad).to(ts.device).to(ts.dtype)

    if len(K) > 100:
        base_betas = try_load(lambda: naive_beta(base_ts), f"scud_{hash_mat}_beta.npy")
    else:
        base_betas = naive_beta(base_ts)

    def beta(ts):
        closest_index = torch.searchsorted(base_ts, ts.to("cpu"))
        best_guess_l = base_betas[closest_index]
        out = best_guess_l
        return out.to(ts.device).to(ts.dtype)

    return log_alpha, beta, mi, precompute_mis


def get_a_b_func_mi(
    mat: torch.Tensor, p0: torch.Tensor, type_: str, **kwargs: object
) -> tuple[Callable[..., torch.Tensor], ...]:
    if type_ == "schedule_condition":
        return get_a_b_func_sc(mat, p0, **kwargs)  # type: ignore[arg-type,return-value]
    elif type_ == "SEDD":
        return get_a_b_func_cont(mat, p0, **kwargs)
    raise ValueError(f"Unknown type_: {type_}")
