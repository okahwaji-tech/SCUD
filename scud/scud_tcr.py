"""
SCUD with Trajectory Consistency Regularization (TCR).

Subclasses SCUD and overrides forward() to add Path B:
  Path A: standard SCUD ELBO loss on forward-corrupted data
  Path B: unweighted CE on model-generated trajectories via Eq. 21

Gradient normalization balances the two losses (different scales),
then combined as (loss_A + w * loss_B) / 2.
"""

import torch
import torch.nn.functional as F

from .scud import SCUD
from .tcr import detached_predict, scud_backward_transition, sample_s_low, grad_norm_weight
from .utils import kls


class SCUD_TCR(SCUD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, attn_mask=None):
        # =============================================
        # Path A: standard SCUD ELBO loss
        # =============================================
        t, S, x_t = self.sample_point(x, attn_mask)
        predicted_x0_logits = self.model_predict(x_t, t, attn_mask, S).to(torch.float32)

        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)

        kl = kls(true_q_posterior_logits, pred_q_posterior_logits)
        if attn_mask is not None:
            kl = kl * attn_mask
        weight = -self.beta(t) / self.log_alpha(t)
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1)
        loss_A = (kl * weight).mean() * self.t_max
        if attn_mask is not None:
            loss_A = loss_A / attn_mask.mean()

        # =============================================
        # Path B: unweighted CE on model-generated trajectory
        # =============================================
        # Reuse Path A logits (detached) to get x̂_0
        x0_hat, _ = detached_predict(self, x_t, t, attn_mask, S)

        # Sample intermediate noise level
        s_low, k = sample_s_low(S)

        # Generate faithful intermediate state via Eq. 21
        x_unrolled = scud_backward_transition(
            x_t, x0_hat, self.K_powers, S, k, self.num_classes,
            self.eigenvectors, self.eigenvalues, self.eigenvectors_inv,
        )

        # Forward pass on unrolled state with gradients
        predicted_x0_logits_B = self.model_predict(x_unrolled, t, attn_mask, s_low).to(torch.float32)

        # Unweighted CE against ground truth
        ce_logits_B = predicted_x0_logits_B.flatten(start_dim=0, end_dim=-2)
        ce_targets_B = x.flatten(start_dim=0, end_dim=-1)
        loss_B = F.cross_entropy(ce_logits_B, ce_targets_B, reduction='none')
        if attn_mask is not None:
            loss_B = (loss_B * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            loss_B = loss_B.mean()

        # =============================================
        # Combined loss with gradient normalization
        # =============================================
        w = grad_norm_weight(loss_A, loss_B, self.parameters())
        loss = (loss_A + w * loss_B) / 2

        # CE loss for logging (from Path A prediction)
        ce_logits_A = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        ce_loss = F.cross_entropy(ce_logits_A, ce_targets_B, reduction='none')
        if attn_mask is not None:
            ce_loss = (ce_loss * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            ce_loss = ce_loss.mean()

        return loss, {
            "vb_loss": loss_A.detach().item(),
            "ce_loss_tcr": loss_B.detach().item(),
            "ce_loss": ce_loss.detach().item(),
            "grad_norm_w": w.item(),
        }
