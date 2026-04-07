"""SCUD with Trajectory Consistency Regularization (SM-DFM).

Trains using a single unified SCUM Cross-Entropy objective over a mixture
of base states (tau=0, forward-corrupted) and unrolled states (tau>0, model-
generated trajectories). Evaluates using the strict SCUD ELBO with tau=0.

Reference: Semi-Markov Discrete Flow Matching (SM-DFM).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .scud import SCUD
from .tcr import detached_predict, sample_k, scud_backward_transition
from .utils import kls


class SCUD_TCR(SCUD):
    """SCUD with Trajectory Consistency Regularization (SM-DFM).

    Both training paths use cross-entropy loss, yielding stable gradients
    without ELBO computation or gradient normalization during training.
    The SCUD ELBO is computed without gradients for logging only.
    """

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute SM-DFM training loss.

        Two paths, both using cross-entropy:
          Path A: CE on true forward-corrupted data (tau=0)
          Path B: CE on model-generated trajectories (tau=k)

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

        # === Path B: CE on model-generated trajectory ===
        x0_hat, _ = detached_predict(self, x_t, t, attn_mask, S, tau=tau_zero)

        s_low, k = sample_k(S)
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

        predicted_x0_logits_B = self.model_predict(x_unrolled, t, attn_mask, s_low, tau=k).to(
            torch.float32
        )
        ce_unroll = self._compute_ce(predicted_x0_logits_B, x, attn_mask)

        # === Combined loss ===
        loss = (ce_base + ce_unroll) / 2

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
            "ce_loss_tcr": ce_unroll.detach().item(),
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
