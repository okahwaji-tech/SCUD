"""
SCUD with Kernel-Consistent Self-Correction (KCSC).

Subclasses SCUD and overrides forward() to add Path B:
  Path A: standard SCUD ELBO loss on forward-corrupted data (S > 0, score = 0)
  Path B: CE on model predictions in correction mode (S = 0, score conditioning)

Optionally weights Path B loss by normalized kernel consistency score,
focusing gradient on positions where the model's predictions are least
consistent with the observed corruption.

Gradient normalization balances the two losses (different scales),
then combined as (loss_A + w * loss_B) / 2.
"""

import torch
import torch.nn.functional as F

from .scud import SCUD
from .tcr import compute_kernel_consistency_score, detached_predict, grad_norm_weight
from .utils import kls


class SM_SCUD_TCR(SCUD):
    def __init__(self, *args, score_weighted_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_weighted_loss = score_weighted_loss
        for p in self.x0_model.score_map.parameters(): p.requires_grad_(True)
        for p in self.x0_model.score_zero_linear.parameters(): p.requires_grad_(True)

        
    def base_predict(self, x_t, t, attn_mask, S=None):
        assert S is not None
        if not hasattr(self, '_score') or self._score.shape != S.shape:
            self._score = torch.zeros(S.shape, device=S.device, dtype=torch.float32)
        return self.x0_model(x_t, t, attn_mask, S, self._score).to(torch.float32)

    def forward(self, x, attn_mask=None):
        # =============================================
        # Path A: standard SCUD ELBO loss
        # =============================================
        t, S, x_t = self.sample_point(x, attn_mask)
        self._score = torch.zeros(S.shape, device=S.device, dtype=torch.float32)
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

        score, loss_weight = compute_kernel_consistency_score(
            x0_hat, x_t, S, self.K_powers, self.log_K_powers_max,
        )

        # Forward pass in correction mode: S=0, score as conditioning
        S_zero = torch.zeros_like(S)
        self._score = score
        predicted_x0_logits_B = self.model_predict(x0_hat, t, attn_mask, S_zero).to(torch.float32)

        # CE against ground truth
        ce_logits_B = predicted_x0_logits_B.flatten(start_dim=0, end_dim=-2)
        ce_targets_B = x.flatten(start_dim=0, end_dim=-1)
        per_position_ce = F.cross_entropy(ce_logits_B, ce_targets_B, reduction='none')
        if self.score_weighted_loss:
            # Normalize loss_weight to mean 1 so it reweights relative importance
            # without inflating total loss magnitude
            norm_weight = loss_weight / (loss_weight.mean() + 1e-8)
            norm_weight_flat = norm_weight.flatten()
            if attn_mask is not None:
                mask_flat = attn_mask.flatten()
                loss_B = (per_position_ce * norm_weight_flat * mask_flat).sum() / mask_flat.sum()
            else:
                loss_B = (per_position_ce * norm_weight_flat).mean()
        else:
            if attn_mask is not None:
                mask_flat = attn_mask.flatten()
                loss_B = (per_position_ce * mask_flat).sum() / mask_flat.sum()
            else:
                loss_B = per_position_ce.mean()

        # =============================================
        # Combined loss
        # =============================================
        # TODO: re-enable gradient normalization if loss scales diverge
        # if self.training:
        #     w = grad_norm_weight(loss_A, loss_B, self.parameters())
        # else:
        #     w = 1.0
        w = 1.0
        loss = (loss_A + loss_B) / 2

        # CE loss for logging (from Path A prediction)
        ce_logits_A = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        ce_loss = F.cross_entropy(ce_logits_A, ce_targets_B, reduction='none')
        if attn_mask is not None:
            ce_loss = (ce_loss * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            ce_loss = ce_loss.mean()

        # Unweighted correction CE for monitoring (independent of score weighting)
        if attn_mask is not None:
            ce_loss_correction_unweighted = (per_position_ce * mask_flat).sum() / mask_flat.sum()
        else:
            ce_loss_correction_unweighted = per_position_ce.mean()

        score_weight_norm = self.x0_model.score_zero_linear.weight.data.norm().item()
        if self.training:
            self.log('score_weight_norm', score_weight_norm, sync_dist=True)
        return loss, {
            "vb_loss": loss_A.detach().item(),
            "ce_loss_correction": loss_B.detach().item(),
            "ce_loss_correction_unweighted": ce_loss_correction_unweighted.detach().item(),
            "ce_loss": ce_loss.detach().item(),
            "grad_norm_w": w.item() if isinstance(w, torch.Tensor) else w,
            "score_weight_norm": score_weight_norm,
            "mean_score": score.mean().item(),
        }