"""
SCUD pretrain on the SMSCUD architecture.

Trains standard SCUD (Path A only, ELBO loss) with τ=0 forced everywhere
and the τ modules (tau_map, tau_gate) frozen. Produces a checkpoint
compatible with SM_SCUD_TCR fine-tuning — the τ pathway is dormant at
the end of pretraining and available to be unfrozen downstream.
"""

import torch
import torch.nn.functional as F

from .scud import SCUD
from .utils import kls


class SM_SCUD_PT(SCUD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for p in self.x0_model.tau_map.parameters():
            p.requires_grad_(False)
        for p in self.x0_model.tau_zero_linear.parameters():
            p.requires_grad_(False)

    def base_predict(self, x_t, t, attn_mask, S=None):
        assert S is not None
        self._tau = torch.zeros_like(S)
        return self.x0_model(x_t, t, attn_mask, S, self._tau).to(torch.float32)

    def forward(self, x, attn_mask=None):
        t, S, x_t = self.sample_point(x, attn_mask)
        self._tau = torch.zeros_like(S)
        predicted_x0_logits = self.model_predict(x_t, t, attn_mask, S).to(torch.float32)

        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)

        kl = kls(true_q_posterior_logits, pred_q_posterior_logits)
        if attn_mask is not None:
            kl = kl * attn_mask
        weight = -self.beta(t) / self.log_alpha(t)
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1)
        loss = (kl * weight).mean() * self.t_max
        if attn_mask is not None:
            loss = loss / attn_mask.mean()

        # CE loss for logging
        ce_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        ce_targets = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = F.cross_entropy(ce_logits, ce_targets, reduction='none')
        if attn_mask is not None:
            ce_loss = (ce_loss * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            ce_loss = ce_loss.mean()

        # Sanity check: τ pathway weights should stay at zero (modules are frozen)
        tau_weight_norm = self.x0_model.tau_zero_linear.linear.weight.data.norm().item()
        if self.training:
            self.log('tau_weight_norm', tau_weight_norm, sync_dist=True)

        return loss, {
            "vb_loss": loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
            "tau_weight_norm": tau_weight_norm,
        }
