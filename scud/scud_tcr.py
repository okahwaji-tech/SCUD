"""SCUD with Trajectory Consistency Regularization (SM-DFM).

Trains using a single unified SCUM Cross-Entropy objective over a mixture
of base states (tau=0, forward-corrupted) and unrolled states (tau=binary,
model-generated trajectories). Evaluates using the strict SCUD ELBO with tau=0.

Key changes from naive TCR:
  - Single-step unrolling (k=1) with binary tau from value comparison
  - Curriculum warmup: lambda ramps from 0 to lambda_max over warmup epochs
  - Path B skipped during validation and when lambda=0

Reference: Semi-Markov Discrete Flow Matching (SM-DFM).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .scud import SCUD
from .tcr import detached_predict, scud_backward_transition
from .utils import kls


class SCUD_TCR(SCUD):
    """SCUD with Trajectory Consistency Regularization (SM-DFM).

    Both training paths use cross-entropy loss, yielding stable gradients
    without ELBO computation or gradient normalization during training.
    The SCUD ELBO is computed without gradients for logging only.

    Args:
        tcr_warmup_epochs: Number of epochs over which to ramp lambda from 0 to lambda_max.
        tcr_lambda_max: Maximum weight for the unrolled (Path B) loss term.
    """

    def __init__(
        self,
        *args: object,
        tcr_warmup_epochs: int = 3,
        tcr_lambda_max: float = 0.5,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tcr_warmup_epochs = tcr_warmup_epochs
        self.tcr_lambda_max = tcr_lambda_max

    def _get_tcr_lambda(self) -> float:
        """Compute curriculum mixing weight for the current epoch.

        Returns:
            Lambda in [0, tcr_lambda_max] that linearly warms up over
            tcr_warmup_epochs, then stays at tcr_lambda_max.
        """
        epoch = getattr(self, "current_epoch", 0)
        warmup = max(self.tcr_warmup_epochs, 1)
        return min(epoch / warmup, 1.0) * self.tcr_lambda_max

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute SM-DFM training loss.

        Two paths, both using cross-entropy:
          Path A: CE on true forward-corrupted data (tau=0)
          Path B: CE on single-step unrolled trajectories (binary tau)

        Path B is only run during training and when tcr_lambda > 0.
        Also computes SCUD ELBO (no gradients) for logging.

        Args:
            x: Clean data tensor, shape (B, ...).
            attn_mask: Optional attention mask, shape (B, L).

        Returns:
            Tuple of (loss, info_dict).
        """
        # === Shared: sample noise level ===
        t, S, x_t = self.sample_point(x, attn_mask)
        tau_zero = torch.zeros_like(S)

        # === Path A: CE on true forward marginals (tau=0) ===
        predicted_x0_logits_A = self.model_predict(x_t, t, attn_mask, S, tau=tau_zero).to(
            torch.float32
        )
        ce_base = self._compute_ce(predicted_x0_logits_A, x, attn_mask)

        # === Path B: single-step unrolled CE (only during training) ===
        lam = self._get_tcr_lambda()
        if self.training and lam > 0:
            x0_hat, _ = detached_predict(self, x_t, t, attn_mask, S, tau=tau_zero)

            # Single-step backward: k=1 where S>0, k=0 where S=0
            k = torch.where(S > 0, torch.ones_like(S), torch.zeros_like(S))
            s_low = S - k

            x_unrolled = scud_backward_transition(
                x_t,
                x0_hat,
                self.K_powers,
                S,
                k,
                self.num_classes,
                self.eigenvectors,
                self.eigenvalues,
                self.eigenvectors_inv,
            )

            # Binary tau: 1 where token survived corruption, 0 where it changed
            tau_binary = (x_unrolled == x0_hat).long()

            predicted_x0_logits_B = self.model_predict(
                x_unrolled, t, attn_mask, s_low, tau=tau_binary
            ).to(torch.float32)
            ce_unroll = self._compute_ce(predicted_x0_logits_B, x, attn_mask)

            loss = (1 - lam) * ce_base + lam * ce_unroll
            ce_unroll_val = ce_unroll.detach().item()
        else:
            loss = ce_base
            ce_unroll_val = 0.0

        # === ELBO for logging (no gradients) ===
        with torch.no_grad():
            true_q = self.q_posterior_logits(x, x_t, t, S)
            pred_q = self.q_posterior_logits(predicted_x0_logits_A, x_t, t, S)
            kl = kls(true_q, pred_q)
            if attn_mask is not None:
                kl = kl * attn_mask
            weight = -self.beta(t) / self.log_alpha(t)
            weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1)
            vb_loss = (kl * weight).mean() * self.t_max
            if attn_mask is not None:
                vb_loss = vb_loss / attn_mask.mean()

        return loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_base.detach().item(),
            "ce_loss_tcr": ce_unroll_val,
            "tcr_lambda": lam,
        }

    def _compute_ce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss with optional masking.

        Args:
            logits: Predicted logits, shape (B, ..., C).
            targets: Target class indices, shape (B, ...).
            attn_mask: Optional mask, shape (B, L).

        Returns:
            Scalar cross-entropy loss.
        """
        flat_logits = logits.flatten(start_dim=0, end_dim=-2)
        flat_targets = targets.flatten(start_dim=0, end_dim=-1)
        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        if attn_mask is not None:
            ce = (ce * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            ce = ce.mean()
        return ce
