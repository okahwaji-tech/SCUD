"""Root-finding algorithms for schedule inversion.

Provides Brent-style and Newton root finders used to invert the mutual
information schedule, mapping target MI values to the corresponding
log-alpha (noise level) parameter.

Reference: "Why Masking Diffusion Works" (NeurIPS 2025), Section 5.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def root_finder(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: float,
    x1: float,
    ts: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Vectorized Brent-style root finder combining bisection and secant methods.

    Finds x such that func(x, ts) = 0 for each element of ts.

    Args:
        func: Function f(x, t) -> residual, operating element-wise.
        x0: Left bracket for all roots.
        x1: Right bracket for all roots.
        ts: Target values tensor.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.

    Returns:
        Tensor of roots, NaN where convergence failed.
    """

    def secant_step(x0, x1, f0, f1):
        return x1 - f1 * (x1 - x0) / (
            f1 - f0 + 1e-15
        )  # Add small epsilon to avoid division by zero

    def bisect_step(x0, x1, f0, f1):
        return (x1 + x0) / 2

    x0_t = x0 * torch.ones_like(ts)
    x1_t = x1 * torch.ones_like(ts)
    x2 = x1_t.clone()

    f0 = func(x0_t, ts)
    f1 = func(x1_t, ts)
    f2 = f1.clone()

    bisect_mode = torch.ones_like(ts).bool()
    mask = torch.ones_like(ts).bool()

    for _ in range(max_iter):
        x2[mask] = torch.where(
            bisect_mode, bisect_step(x0_t, x1_t, f0, f1), secant_step(x0_t, x1_t, f0, f1)
        )[mask]
        f2[mask] = func(x2[mask], ts[mask])
        bisect_mode = torch.where(
            torch.abs(f2) < 0.5 * torch.abs(f1 - f0), bisect_mode, ~bisect_mode  # did well
        )

        update_x1 = torch.sign(f2) == torch.sign(f1)
        x0_t[mask] = torch.where(update_x1, x0_t, x1_t)[mask]
        x1_t[mask] = x2[mask]
        f0[mask] = torch.where(update_x1, f0, f1)[mask]
        f1[mask] = f2[mask]

        mask = torch.abs(f2) > tol
        if (~mask).all():
            break

    return torch.where(torch.abs(f2) < tol, x2, torch.full_like(x2, float("nan")))


def newton_root_finder(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: float,
    ts: torch.Tensor,
    min_x: torch.Tensor | None = None,
    max_iter: int = 1000,
    tol: float = 1e-12,
    print_: bool = False,
) -> torch.Tensor:
    """Vectorized Newton root finder with automatic differentiation.

    Finds x such that func(x, ts) = 0 for each element of ts, using
    torch.autograd to compute derivatives.

    Args:
        func: Function f(x, t) -> residual, must be differentiable w.r.t. x.
        x0: Initial guess for all roots.
        ts: Target values tensor.
        min_x: Minimum allowed x value (clamp).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
        print_: If True, print convergence diagnostics.

    Returns:
        Tensor of roots with the same shape as ts.
    """
    if min_x is None:
        min_x = torch.tensor(1e-8)
    ts = ts.detach()
    x = x0 * torch.ones_like(ts).double()
    x = torch.maximum(x, min_x)
    f = torch.ones_like(ts)
    df = torch.ones_like(ts)
    mask = torch.ones_like(ts).bool()

    for _ in range(max_iter):
        with torch.enable_grad():
            x_masked = x[mask].detach().requires_grad_(True)
            f_masked = func(x_masked, ts[mask])
            df_masked = torch.autograd.grad(f_masked.sum(), x_masked)[0]
            f[mask] = f_masked.detach()
            df[mask] = df_masked

        step = -f / df
        x_at_min = (x == min_x) * (step < 0)
        x[mask] = torch.where(mask, x + step, x)[mask]
        x = torch.maximum(x, min_x)

        mask = (torch.abs(f) > tol) * (~x_at_min)
        if (~mask).all():
            break
        if print_:
            print(
                "av errs",
                (torch.abs(f)[torch.abs(f) > tol]).mean(),
                "N:",
                (torch.abs(f) > tol).sum(),
            )

    return x
