"""Masking Diffusion model -- a special case of SCUD.

Implements the masking (absorbing-state) variant of discrete diffusion
where the forward process replaces tokens with a [MASK] token. This is
shown to be a special case of SCUD with uniform K and gamma = 1/N
(Section 6.2 of the paper).

Reference: "Why Masking Diffusion Works" (NeurIPS 2025), Section 6.2.
"""

from __future__ import annotations

from typing import cast

import torch
from tqdm import tqdm

from scud.scud import SCUD

from .utils import kls


class MaskingDiffusion(SCUD):
    """Masking (absorbing-state) discrete diffusion as a SCUD special case.

    Uses a uniform transition kernel K with gamma = 1/N, which makes the
    forward process equivalent to independently masking each token. The
    posterior simplifies: S > 1 yields uniform, S == 1 yields the x_0
    prediction, so S is always treated as binary (masked / unmasked).

    Args:
        x0_model_class: Neural network class for the denoiser.
        nn_params: Constructor kwargs for the denoiser network.
        num_classes: Number of discrete token classes (excluding mask token).
        schedule_type: Noise schedule type.
        logistic_pars: If True, use logistic parameterization.
    """

    def __init__(
        self,
        x0_model_class: type,
        nn_params: dict[str, object],
        num_classes: int = 10,
        schedule_type: str = "cos",
        logistic_pars: bool = False,
        **kwargs: object,
    ) -> None:
        forward_kwargs: dict[str, object] = {"type": "uniform"}
        gamma = 1 / num_classes
        if "gamma" in kwargs:
            del kwargs["gamma"]
        if "forward_kwargs" in kwargs:
            del kwargs["forward_kwargs"]
        super().__init__(
            x0_model_class,
            nn_params,
            num_classes,
            forward_kwargs,
            schedule_type,
            gamma,
            logistic_pars,
            **kwargs,
        )
        self.use_bad_model_predict = ~logistic_pars
        # with this choice, x_t_sample is uniform and
        # q_posterior_logits returns uniform if S>1 and x_0 pred if S==1
        # The only differences is the predictions and marginalizing over S>1 in the weight
        # so we always assume S==1.
        # in principle we could also speed up sampling by ignoring S>1

    def base_predict(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.Tensor | None,
        S: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert S is not None
        masked_pos = S > 0
        masked_x_t = torch.where(masked_pos, self.num_classes, x_t)
        masked_x_t = torch.where(
            cast(torch.Tensor, attn_mask) == 1, masked_x_t, x_t
        )  # don't mask pos that are already masked
        result: torch.Tensor = self.x0_model(masked_x_t, t, attn_mask, S=S)[..., :-1]
        return result

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        t, S, x_t = self.sample_point(x, attn_mask)
        S = (S > 0).long()
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, attn_mask, S).float()
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)

        # get kls and loss
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits)  # shape x
        if attn_mask is not None:
            kl = kl * attn_mask
        alpha_t = torch.exp(self.log_alpha(t))
        weight = self.beta(t) * alpha_t / (1 - alpha_t)  # mult by p(S=1|t)
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1)
        vb_loss = (kl * weight).mean() * self.t_max
        if attn_mask is not None:
            vb_loss = vb_loss / attn_mask.mean()

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss(reduction="none")(predicted_x0_logits, x)
        if attn_mask is not None:
            ce_loss = (ce_loss * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            ce_loss = ce_loss.mean()

        return vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def sample_sequence(  # type: ignore[override]
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        n_T: int = 200,
        stride: int = 10,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        t = t * 0 + 1e-6
        S = (1.0 + 0.0 * x).long()  # this is the only line changed
        steps = 0
        images = []
        n_steps = int(torch.tensor([S[b].sum() for b in range(len(S))]).max().item())
        pbar = tqdm(total=n_steps, unit="iteration", position=0, leave=True)
        trans_step = max([n_steps // n_T, 1])
        while S.sum() > 0:
            # predict what comes next
            x_next = self.p_sample(
                x,
                t,
                attn_mask,
                torch.rand((*x.shape, self.num_classes), device=x.device),
                S,
                temperature=1,
            )
            for b in range(len(x)):
                trans_indices = torch.argwhere(S[b] > 0)
                trans_indices = trans_indices[torch.randperm(len(trans_indices))]
                if len(trans_indices) > 0:
                    # randomly transiiton
                    for idx in trans_indices[:trans_step]:
                        idx_tuple = (b,) + tuple(idx)
                        x[idx_tuple] = x_next[idx_tuple]
                        S[idx_tuple] -= 1
            pbar.update(trans_step)
            steps += 1
            if steps % stride == 0:
                images.append(torch.clone(x))
        pbar.close()
        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images
