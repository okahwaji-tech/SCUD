"""Schedule-Conditioned Discrete Diffusion (SCUD) model.

Implements the core SCUD diffusion model that conditions its denoising
network on the realized jump schedule S, enabling tighter variational
bounds and unifying masking and classical discrete diffusion.

Reference: "Why Masking Diffusion Works" (NeurIPS 2025), Sections 4-6.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .continuous_time_diffusion import ContinuousTimeDiffusion
from .schedule_sample import sample_n_transitions_cont
from .utils import convert_to_probs, get_inf_gen, kls


class SCUD(ContinuousTimeDiffusion):
    """Schedule-Conditioned Discrete Diffusion (SCUD) model.

    Implements the SCUD framework (Sections 4-6) where the denoiser is
    conditioned on the realized jump schedule S ~ Poisson(-log_alpha(t)).
    The transition kernel K is derived from the infinitesimal generator L,
    and matrix powers K^S are precomputed via eigendecomposition for
    efficient posterior computation.

    Key hyperparameters:
        gamma: Controls how much schedule information is provided to the
            model. gamma=0 (code) = full schedule conditioning (paper gamma=1).
        forward_kwargs: Specifies the forward process type (uniform, gaussian, blosum).
        schedule_type: Noise schedule parameterization.

    See the NOTE below about the gamma convention difference between code and paper.
    """

    MAX_K_POWERS = 5000
    MAX_CLASSES_FOR_PRECOMPUTE = 512

    # Type annotations for register_buffer tensors
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    eigenvectors_inv: torch.Tensor
    K: torch.Tensor
    K_powers: torch.Tensor

    def __init__(
        self,
        x0_model_class: type,
        nn_params: dict[str, object],
        num_classes: int = 10,
        forward_kwargs: dict[str, object] | None = None,
        schedule_type: str = "cos",
        gamma: float = 0,
        logistic_pars: bool = False,
        **kwargs: object,
    ) -> None:
        if forward_kwargs is None:
            forward_kwargs = {"type": "uniform"}
        # Precalculate betas, define model_predict, p_sample
        super().__init__(
            x0_model_class, nn_params, num_classes, schedule_type, logistic_pars, **kwargs  # type: ignore[arg-type]
        )
        self.save_hyperparameters(ignore=["x0_model_class"])
        assert gamma >= 0  # full schedule and classical resp.
        assert gamma < 1

        # Precalculate Ls
        L = get_inf_gen(forward_kwargs, num_classes)
        # Get Ks
        # NOTE: Gamma convention differs from paper.
        # Paper: r = r* / gamma, where gamma in (0, 1], gamma=1 is full schedule conditioning.
        # Code:  r = r* / (1-gamma), where gamma in [0, 1), gamma=0 is full schedule conditioning.
        # Relationship: code_gamma = 1 - paper_gamma.
        # gamma=0 here -> rate=r* (slowest events, most schedule info) = paper's gamma=1
        # gamma->1 here -> rate->inf (fastest events, classical diffusion) = paper's gamma->0
        rate = -(L.diagonal().min()) / (1 - gamma)  # L^* in sec 6.6 of the notes
        K = L / rate + torch.eye(num_classes)
        self.rate = rate
        eigenvalues, eigenvectors = torch.linalg.eig(K.double())
        eigenvalues[torch.real(eigenvalues) > 1 - self.eps] = 1
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        self.register_buffer("eigenvalues", eigenvalues)
        self.register_buffer("eigenvectors", eigenvectors)
        self.register_buffer("eigenvectors_inv", eigenvectors_inv)

        # Precalculate K_powers
        assert num_classes <= self.MAX_CLASSES_FOR_PRECOMPUTE
        assert forward_kwargs["type"] != "bert_embed"
        K_powers = torch.stack([torch.linalg.matrix_power(K, i) for i in range(self.MAX_K_POWERS)])
        self.register_buffer("K", K)
        self.register_buffer("K_powers", K_powers)

    def pre_configure_model(self, dataloader: object) -> None:
        self.calc_p0(dataloader)
        self.log_alpha, self.beta, *_ = self.get_beta_func(
            self.K.cpu(), self.p0.cpu(), type_="schedule_condition", scale=self.rate.cpu()
        )

    def get_stationary(self) -> torch.Tensor:
        evals, evecs = torch.linalg.eig(self.K.T)
        norms_sq = torch.real(evals * evals.conj())
        assert torch.isclose(evals[torch.argmax(norms_sq)], torch.tensor(1, dtype=torch.complex64))
        stationary = evecs[:, torch.argmax(norms_sq)]
        assert torch.allclose(torch.imag(stationary), torch.tensor(0, dtype=self.K.dtype))
        stationary = torch.real(stationary)
        stationary = stationary * torch.sign(stationary)
        assert torch.all(stationary >= 0)
        return stationary / stationary.sum()

    def get_trans_mats_mvp(self, Smk: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute K^{S-k} @ v via eigendecomposition-based matrix-vector product.

        Efficiently computes the matrix-vector product of the transition matrix
        raised to a per-element power with a probability vector, using the
        precomputed eigendecomposition of K: K^n = V diag(lambda^n) V^{-1}.

        Args:
            Smk: Per-element exponents (S - k), shape (...).
            v: Probability vectors, shape (..., C) where C = num_classes.

        Returns:
            Result of K^{Smk} @ v, shape (..., C), clamped to non-negative.
        """
        dv = v.to(dtype=self.eigenvectors.dtype).reshape(-1, v.shape[-1])
        diag = self.eigenvalues ** F.relu(Smk.flatten()[..., None])
        dv = dv @ self.eigenvectors
        dv = dv * diag
        dv = dv @ self.eigenvectors_inv
        return F.relu(dv.double()).to(torch.float32).reshape(v.shape)

    def get_kl_t1(self, x: torch.Tensor) -> torch.Tensor:
        # sample S
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        softmaxed = convert_to_probs(x, self.num_classes)  # bs, ..., num_classes
        trans = self.get_trans_mats_mvp(S, softmaxed)
        x_1 = torch.log(trans + self.eps)
        kl = kls(x_1, torch.log(self.get_stationary() + self.eps))
        return kl.mean()

    def x_t_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, S: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample from the forward process x_t ~ K^S(x_t | x_0) via Gumbel trick.

        Applies S steps of the transition kernel K to the clean data x_0
        by looking up precomputed K^S powers and sampling with Gumbel noise.

        Args:
            x_0: Clean data tensor of integer class indices, shape (B, ...).
            t: Diffusion time (unused here, schedule is encoded in S).
            noise: Uniform noise for Gumbel sampling, shape (B, ..., C).
            S: Number of transitions per element, shape (B, ...).

        Returns:
            Noisy data x_t with same shape as x_0.
        """
        # forward process, x_0 is the clean input.
        probs = self.K_powers[S, x_0, :]
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = 1 / (-torch.log(noise))
        x_t = torch.argmax(probs * gumbel_noise, dim=-1)
        return x_t

    def q_posterior_logits(  # type: ignore[override]
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        S: torch.Tensor | None = None,
        k: int | torch.Tensor = 1,
        log: bool = True,
    ) -> torch.Tensor:
        """Compute the denoising posterior q(x_{t-k} | x_t, x_0) (Eq. 5).

        Factorizes as: q(x_{t-k} | x_t, x_0) proportional to
        K^k(x_t | x_{t-k}) * K^{S-k}(x_{t-k} | x_0).

        Args:
            x_0: Clean data or predicted logits, shape (B, ...) or (B, ..., C).
            x_t: Current noisy data, shape (B, ...).
            t: Diffusion time (unused, schedule encoded in S).
            S: Total transitions per element, shape (B, ...).
            k: Number of transitions to reverse (default 1).
            log: If True, return log-probabilities; else return probabilities.

        Returns:
            Posterior (log-)probabilities of shape (B, ..., C).
        """
        assert S is not None
        fact1 = self.K_powers.swapaxes(1, 2)[k, x_t, :]  # x_t | x_{t-1}
        softmaxed = convert_to_probs(x_0, self.num_classes)  # bs, ..., num_classes
        fact2 = self.get_trans_mats_mvp(S - k, softmaxed)  # x_{t-1} | x_{0}
        assert torch.all(fact1 >= 0)
        assert torch.all(fact2 >= 0)
        if log:
            return torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        else:
            return fact1 * fact2

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute SCUD training loss (Proposition 4.4, Eq. 6).

        Samples a random time t, computes the schedule S ~ Pois(-log_alpha(t)),
        corrupts data via K^S, and computes the weighted KL divergence between
        true and predicted denoising posteriors.

        Args:
            x: Clean data tensor, shape (B, ...).
            attn_mask: Optional attention mask for sequence data, shape (B, L).

        Returns:
            Tuple of (loss, info_dict) where info_dict contains 'vb_loss' and 'ce_loss'.
        """
        t, S, x_t = self.sample_point(x, attn_mask)
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, attn_mask, S).to(torch.float32)
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)
        # get kls and loss
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits)  # shape x
        if attn_mask is not None:
            kl = kl * attn_mask
        weight = -self.beta(t) / self.log_alpha(t)
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

    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.Tensor | None,
        noise: torch.Tensor,
        S: torch.Tensor | None = None,
        k: int | torch.Tensor = 1,
        temperature: float = 1,
    ) -> torch.Tensor:
        # predict prev(x_t) or x_{t-1}
        predicted_x0_logits = self.model_predict(x, t, attn_mask, S) / temperature
        pred_q_posterior_logits = self.q_posterior_logits(
            predicted_x0_logits, x, t, S, k=k, log=False
        )
        # sample
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = 1 / (-torch.log(noise))
        sample = torch.argmax(pred_q_posterior_logits * gumbel_noise, dim=-1)
        return sample

    def corrector_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.Tensor | None,
        noise: torch.Tensor,
        S: torch.Tensor | None = None,
        k: int | torch.Tensor = 1,
        temperature: float = 1,
    ) -> torch.Tensor:
        # predict prev(x_t) or x_{t-1}
        predicted_x0_logits = self.model_predict(x, t, attn_mask, S) / temperature
        pred_q_posterior_logits = self.q_posterior_logits(
            predicted_x0_logits, x, t, S, k=k, log=False
        )
        # K'_x,.K_.,. denoises and then renoises immediately
        sample_logits = torch.einsum("...i,...ij->...j", pred_q_posterior_logits, self.K_powers[k])
        # sample
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = 1 / (-torch.log(noise))
        sample = torch.argmax(sample_logits * gumbel_noise, dim=-1)
        return sample

    def _build_denoising_schedule(self, S, x, total_steps, n_T, trans_step, use_tau):
        """Construct the ks tensor: how many events to denoise at each step per dimension."""
        ks = torch.zeros(
            [len(x), total_steps, len(S[0].flatten())], device=S.device, dtype=torch.long
        )
        for b in range(len(x)):
            # count how many in each bin
            if use_tau:
                ts = torch.linspace(0, self.t_max, total_steps + 1, device=S.device).to(
                    torch.float32
                )
                weights = -self.log_alpha(ts)
                diffs = weights[1:] - weights[:-1]
                n_steps = torch.bincount(
                    torch.multinomial(diffs, num_samples=S[b].sum(), replacement=True),
                    None,
                    n_T,
                )
                n_steps = torch.cumsum(n_steps, -1)
                n_steps = torch.cat([torch.zeros_like(n_steps[[0]]), n_steps], axis=-1).long()
                assert n_steps[-1] == S[b].sum()
            else:
                n_steps = trans_step * torch.ones(total_steps, device=S.device).to(torch.float32)
                n_steps = torch.cumsum(n_steps, -1)
                n_steps = torch.cat([torch.zeros_like(n_steps[[0]]), n_steps], axis=-1).long()
                assert n_steps[-1] >= S[b].sum()

            indices = torch.argwhere(S[b].flatten() > 0)[:, 0]
            values = S[b].flatten()[indices]
            repeated_indices = torch.repeat_interleave(indices, values.long(), dim=0)
            repeated_indices = repeated_indices[torch.randperm(repeated_indices.size(0))]
            uniq = [
                torch.unique(
                    repeated_indices[n_steps[step] : n_steps[1 + step]],
                    return_counts=True,
                )
                for step in range(total_steps)
            ]
            for (u, c), i in zip(uniq, range(len(uniq)), strict=True):
                if len(u) > 0:
                    ks[b][i][u] += c
        assert torch.all(ks.sum(1) == S.reshape(len(x), -1))
        return ks

    def _run_denoising_loop(
        self,
        x,
        S,
        ks,
        t,
        attn_mask,
        n_corrector_steps,
        temperature,
        trans_step,
        trans_corrector_k,
        stride,
        images,
    ):
        """Run the denoising loop, iterating through steps and calling p_sample/corrector_sample."""
        steps = 0
        n_steps = torch.tensor([S[b].sum() for b in range(len(S))]).max().item()
        pbar = tqdm(total=n_steps, unit="iteration", position=0, leave=True)
        while S.sum() > 0:
            k = ks[:, steps, :].reshape(S.shape)
            S_temp = S - k
            assert torch.all(S_temp >= 0)

            # predict what comes next
            x = self.p_sample(
                x,
                t,
                attn_mask,
                torch.rand((*x.shape, self.num_classes), device=x.device),
                S,
                k=k,
                temperature=temperature,
            )
            assert torch.all(S_temp <= S)
            S = S_temp
            for _l in range(n_corrector_steps):
                x = self.corrector_sample(
                    x,
                    t,
                    attn_mask,
                    torch.rand((*x.shape, self.num_classes), device=x.device),
                    S,
                    k=torch.minimum(S, torch.tensor(trans_corrector_k)),
                    temperature=temperature,
                )
            pbar.update(trans_step)
            steps += 1
            if steps % stride == 0:
                images.append(torch.clone(x))
        pbar.close()
        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)
        return images

    def sample_sequence(  # type: ignore[override]
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        n_T: int = 200,
        stride: int = 10,
        n_corrector_steps: int = 10,
        temperature: float = 1,
        use_tau: bool = False,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        """Generate samples via iterative denoising (Algorithm 2).

        Starts from noise x ~ stationary, samples a schedule S, then
        iteratively denoises by removing k transitions at each step.
        Optionally applies corrector steps that re-noise and denoise.

        Args:
            x: Initial noisy data, shape (B, ...).
            attn_mask: Optional attention mask, shape (B, L).
            n_T: Target number of denoising steps.
            stride: Record intermediate samples every stride steps.
            n_corrector_steps: Number of corrector iterations per step.
            temperature: Sampling temperature.
            use_tau: If True, use time-based schedule for step sizes.

        Returns:
            List of intermediate sample tensors recorded at stride intervals.
        """
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        t = t * 0
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        images: list[torch.Tensor] = []
        n_steps = torch.tensor([S[b].sum() for b in range(len(S))]).max().item()
        trans_step = max([n_steps // n_T, 1]) * (n_corrector_steps + 1)
        total_steps = math.ceil(n_steps / trans_step)
        trans_corrector_k = max([trans_step // np.prod(x[0].shape), 1])
        print("Corrector_k:", trans_corrector_k)

        ks = self._build_denoising_schedule(S, x, total_steps, n_T, trans_step, use_tau)
        result: list[torch.Tensor] = self._run_denoising_loop(
            x,
            S,
            ks,
            t,
            attn_mask,
            n_corrector_steps,
            temperature,
            trans_step,
            trans_corrector_k,
            stride,
            images,
        )
        return result
