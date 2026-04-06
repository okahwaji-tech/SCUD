# Semi-Markov Discrete Flow Matching with Trajectory Consistency Regularization (SM-DFM)

## Project Overview

This project extends SCUD (Schedule-Conditioned Discrete Diffusion, NeurIPS 2025) by addressing the train-inference covariate shift in discrete diffusion models. The core contribution is a Semi-Markov extension that gives the model memory of its own predictions through holding-time embeddings, paired with a trajectory consistency regularization objective that trains the model to correct its own errors.

### The Problem

All discrete diffusion models (SCUD, MDLM, SEDD, D3PM) suffer from a covariate shift between training and inference:

- **During training**: every token at event count `s_t^d` was produced by the forward process applied to ground truth. The event count is a complete description of the token's state.
- **During inference**: tokens are a mixture of forward-process corruption and backward-process predictions. A token at `s_t^d = 4` might be ground truth corrupted 4 times, or a model prediction from 3 steps ago with 4 events remaining. The model cannot distinguish these from `s_t^d` alone.

Additionally, the SCUD ELBO weights each position's loss by `s_t^d`, so positions at `s_t^d = 0` receive zero gradient. During inference, wrong predictions at `s = 0` are treated as ground truth anchors. Errors compound quadratically with inference steps (DAgger/Ross et al., 2011).

This problem is more severe in discrete diffusion than continuous diffusion because there is no smooth interpolation between tokens — a wrong token is fully wrong, not "slightly off manifold."

### The Solution

1. **Semi-Markov extension**: Add holding time `τ_t^d = g_t^d - s_t^d` where `g_t^d` is the event count at which the model last wrote position `d`. This extends the process from CTMC to Semi-Markov Process. The pair `(s, τ)` is a sufficient statistic for token provenance at any noise level.

2. **SCUM formulation**: Use unweighted cross-entropy (discrete flow matching) instead of the ELBO-weighted loss. This removes the `s_t^d` multiplier, giving gradient at every position including `s = 0` with `τ > 0`.

3. **Trajectory Consistency Regularization (TCR)**: Train on detached rollout states where the model encounters its own predictions. The holding time `τ` tells the model how stale each prediction is.

### Why Non-Masking Kernels Are Required

Self-correction through re-corruption requires non-masking kernels (uniform, structured). Masking re-corruption destroys all information (token → ∅). Uniform/structured re-corruption produces tokens that blend prediction with noise — the model can still extract signal. SCUD is the necessary foundation because it showed non-masking kernels can match/beat masking when properly schedule-conditioned.

---

## Theoretical Foundation

### Key Papers

| Paper | Relevance |
|-------|-----------|
| **SCUD** (Amin et al., NeurIPS 2025) — arXiv:2506.08316 | Base framework. Schedule conditioning, SCUM flow matching formulation (Appendix E), structured kernels |
| **SUNDAE** (Savinov et al., 2021) — arXiv:2112.06749 | Unrolled denoising autoencoders. Proved from-scratch training works with detached rollouts. Upper bound proof via Jensen's inequality |
| **CLLMs** (Kou et al., 2024) — arXiv:2403.00835 | Jacobi consistency for LLMs. Parallel decoding with self-correction |
| **MDLM** (Sahoo et al., 2024) — arXiv:2406.07524 | Current SOTA masked discrete diffusion. Primary baseline for language |
| **SEDD** (Lou et al., 2023) | Score entropy discrete diffusion. Baseline for language |
| **DAgger** (Ross et al., 2011) | Theoretical justification: compound error O(T²ε) → O(Tε) |
| **Discrete Flow Matching** (Gat et al., 2024) — arXiv:2407.15595 | General discrete flow matching framework |
| **D3PM** (Austin et al., 2021) | Structured discrete diffusion. Gaussian/BLOSUM kernels |

### SCUD Essentials

**Forward process**: Events occur as Poisson process at rate `r·β_t`. At each event, token transitions via kernel `K`. Infinitesimal generator `L = r(K - I)`.

**SCUD ELBO** (Eq. 6 of paper):
```
-E_{t~Unif(0,1)} E_{p(x_t, x_0, S)} [β_t / ∫β_s ds] × Σ_d s_t^d · KL(p(pr(x_t^d)|x_t^d, s_t^d, x_0^d) || q_θ(pr(x_t^d)|x_t, s_t))
```

**SCUM CE** (Appendix E, flow matching formulation):
```
E_{s, p(x_0), p(x_s|x_0)} x_0^T log x̃_{0,θ}(x_s, s)
```
No `s_t^d` weight. This is what we use for training.

**Backward transition** (Eq. 21):
```
q_θ(pr^k(x_t^d) | x_t, s_t) = K^k · x_t^d ∘ K^{s_t^d - k, T} · x̃_{0,θ}
```
This is used in both inference AND Path B training to generate faithful intermediate states.

**Key property** (Appendix E.2): The x̃_0 predictor can be trained once and the sampling kernel chosen at test time. Training and sampling kernels are decoupled.

### Semi-Markov Extension

**Holding time definition**:
- When model writes position `d` at schedule tick `s`, record `g^d = s`
- At any later time with current count `s_t^d`: `τ_t^d = g^d - s_t^d`
- `τ = 0` everywhere recovers standard SCUD/SCUM exactly

**State space**: Model input is `(x_t, s_t, τ_t)` — three per-token signals fed through FiLM layers.

**Interpretation**:
- `(s > 0, τ = 0)`: Forward-corrupted ground truth. Standard denoising.
- `(s = 0, τ = 0)`: Untouched ground truth. Identity.
- `(s = 0, τ > 0)`: Model prediction, possibly wrong. Self-correction regime.
- `(s > 0, τ > 0)`: Model prediction with remaining corruption. Least trustworthy.
- `τ` large, `g` large: Predicted from high noise long ago. Low reliability.
- `τ` small, `g` small: Predicted from low noise recently. High reliability.

---

## Training Procedure

### Single Training Step (3 forward passes)

```python
def training_step(x_0, model, K, beta_schedule):
    # =========================================
    # Step 1: Forward corruption
    # =========================================
    s_high = sample_event_counts(beta_schedule)  # Poisson per dimension
    x_t = corrupt(x_0, K, s_high)               # Apply K^{s_high^d} per position

    # =========================================
    # PATH A: SCUM Base Loss (τ = 0)
    # =========================================
    tau_A = zeros_like(s_high)
    x0_pred_A = model(x_t, s_high, tau_A)        # Forward pass WITH gradients
    loss_scum = cross_entropy(x0_pred_A, x_0)     # Unweighted CE

    # =========================================
    # PATH B: Trajectory Consistency (τ > 0)
    # =========================================
    # Detached prediction (no gradients)
    with torch.no_grad():
        x0_hat = model(x_t, s_high, tau_A)        # Same input as Path A
        x0_hat = sample_from_logits(x0_hat)        # Sample discrete tokens

    # Generate faithful intermediate state using SCUD backward transition
    s_low = sample_lower_noise(s_high)             # s_low < s_high per position
    k = s_high - s_low                             # Events to reverse
    # Use Eq. 21: K^k x_t ∘ K^{s-k,T} x̃_0 — depends on BOTH x_t and x̂_0
    x_unrolled = scud_backward_transition(x_t, x0_hat, K, s_high, k)

    # Compute holding time
    tau_B = s_high - s_low                         # = k, the events reversed

    # Train on unrolled state WITH gradients
    x0_pred_B = model(x_unrolled, s_low, tau_B)
    loss_tcr = cross_entropy(x0_pred_B, x_0)       # Target is ground truth x_0

    # =========================================
    # Combined loss with gradient normalization
    # =========================================
    loss = loss_scum + grad_norm_weight(loss_scum, loss_tcr) * loss_tcr
    return loss
```

### Key Implementation Details

**Backward transition for Path B** (faithful state generation):
```python
def scud_backward_transition(x_t, x0_hat, K, s, k):
    """
    Eq. 21 from SCUD paper.
    Produces states from the same distribution as inference.
    x_t: current corrupted tokens (one-hot or indices)
    x0_hat: detached model prediction (one-hot or probabilities)
    K: transition kernel matrix
    s: current event counts per position
    k: number of events to reverse per position
    """
    # K^k @ x_t: forward k steps from current state
    # K^{s-k, T} @ x0_hat: backward from prediction to s-k noise level
    # Element-wise product, then normalize to get distribution
    fwd = matrix_power_vec(K, k, x_t)           # K^k x_t per position
    bwd = matrix_power_vec(K.T, s - k, x0_hat)  # K^{s-k,T} x̃_0 per position
    probs = fwd * bwd
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return sample_categorical(probs)
```

**Gradient normalization**:
```python
def grad_norm_weight(loss_main, loss_aux):
    """Scale auxiliary loss so both terms contribute equal gradient magnitude."""
    grad_main = torch.autograd.grad(loss_main, model.parameters(), retain_graph=True)
    grad_aux = torch.autograd.grad(loss_aux, model.parameters(), retain_graph=True)
    norm_main = torch.sqrt(sum(g.norm()**2 for g in grad_main))
    norm_aux = torch.sqrt(sum(g.norm()**2 for g in grad_aux))
    return (norm_main / (norm_aux + 1e-8)).detach()
```

**Sampling `s_low`**: Sample `s_low^d ~ Uniform(0, s_high^d)` per position. This covers all holding times equally. Could also bias toward low `s_low` (large `τ`) to emphasize the self-correction regime.

### FiLM Layer Modification

SCUD replaces time `t` with per-position `s_t^d` in FiLM layers. We add `τ_t^d`:

```python
class SemiMarkovFiLM(nn.Module):
    def __init__(self, hidden_dim, emb_dim):
        super().__init__()
        self.s_embed = SinusoidalEmbedding(emb_dim)
        self.tau_embed = SinusoidalEmbedding(emb_dim)
        self.scale = nn.Linear(emb_dim * 2, hidden_dim)
        self.shift = nn.Linear(emb_dim * 2, hidden_dim)

    def forward(self, h, s_d, tau_d):
        # h: activations [batch, seq_len, hidden_dim]
        # s_d: event counts [batch, seq_len]
        # tau_d: holding times [batch, seq_len]
        emb_s = self.s_embed(s_d)       # [batch, seq_len, emb_dim]
        emb_tau = self.tau_embed(tau_d)  # [batch, seq_len, emb_dim]
        emb = torch.cat([emb_s, emb_tau], dim=-1)
        gamma = self.scale(emb)
        beta = self.shift(emb)
        return gamma * h + beta
```

No additional parameters beyond the embedding and two linear layers per FiLM block. Same overhead pattern as SCUD's modification of classical diffusion.

---

## Inference Procedure

Standard SCUD backward sampling (Algorithm 2 from paper), with holding time tracking:

```python
def sample(model, K, beta_schedule, num_steps=2048):
    D = sequence_length
    B_vocab = vocab_size

    # Initialize from stationary distribution
    x = sample_stationary(K, D)
    s = sample_initial_counts(beta_schedule, D)  # Poisson per position
    tau = torch.zeros(D)  # No predictions yet
    g = torch.zeros(D)    # No prediction timestamps

    events_per_step = math.ceil(s.sum() / num_steps)

    for step in range(num_steps):
        # Select which positions to denoise this step
        k = select_events_to_reverse(s, events_per_step)

        # Predict x_0
        x0_pred = model(x, s, tau)

        # Reverse k events at each position using Eq. 21
        for d in range(D):
            if k[d] > 0:
                x[d] = scud_backward_sample(x[d], x0_pred[d], K, s[d], k[d])
                g[d] = s[d]           # Record prediction timestamp
                s[d] = s[d] - k[d]    # Decrement event count
                tau[d] = g[d] - s[d]  # Update holding time

    return x
```

### Key Inference Properties
- `τ` is naturally maintained — no special computation needed
- `g^d` is stamped when the model writes position `d`
- `τ^d = g^d - s_t^d` grows as the backward process continues past position `d`
- At the end of inference, positions predicted early (high `g`, high `τ`) are flagged as potentially unreliable
- The model was trained on exactly these `(s, τ)` pairs via Path B

---

## Experimental Plan

### Phase 1: Validation (cheap, ~4 GPU-days)

**Dataset**: Text8 (character-level, small, fast iteration)

**Experiment 1 — Does compound error exist measurably?**
- Take a trained baseline model (SCUD or MDLM)
- During inference, track: (a) per-position error rate at each denoising step, (b) whether errors at early steps cause errors at later steps, (c) how often tokens at s=0 are wrong
- This requires no training — just inference with bookkeeping
- **If compound error is not measurable, reconsider the entire direction**

**Experiment 2 — Does unrolled training help?**
- Train on text8: (a) pure SCUM CE baseline, (b) SCUM CE + Path B without τ (τ=0 everywhere)
- Compare: ELBO (computed at eval time via Algorithm 1), sample quality, self-consistency score (fraction of tokens changing between consecutive inference steps)
- **If unrolled training doesn't help, the direction is dead**

**Experiment 3 — Does τ add value?**
- Add τ to the model from Experiment 2
- Compare against Experiment 2 results
- **If τ doesn't help beyond plain unrolling, it's a SUNDAE-for-SCUD paper, not a Semi-Markov paper**

### Phase 2: Scaling (if Phase 1 is positive, ~32 GPU-days)

**Datasets**: LM1B (language, 30K vocab), UniRef50 (proteins, 31 vocab)

**Models**:
1. Full method: SCUM + TCR + τ (SM-DFM)
2. Ablation: SCUM + TCR, no τ
3. Ablation: SCUM only, no TCR, no τ
4. SCUD baseline (reimplementation for fair comparison)

**Architecture**: Match SCUD paper exactly. Same diffusion transformer (SEDD architecture) for language, same CARP architecture for proteins. Only modification: FiLM layers take (s, τ) instead of just s.

**Training**: Match SCUD compute budget (2 days on 2 A100s per run). ~1.5x cost per step due to 3 forward passes.

**Metrics**:
- SCUD ELBO (Algorithm 1, computed at eval with τ=0 — comparable to all prior work)
- SCUM CE loss
- Self-consistency score: fraction of tokens changing between consecutive inference steps
- Convergence speed: number of inference steps to reach stable output
- Error correction rate: intentionally corrupt positions in partially-denoised sequence, measure recovery

**Baselines from literature** (numbers pulled from papers, noted with compute budgets):
- MDLM (33B/327B tokens) for language
- SEDD (33B tokens) for language
- SCUD (11B tokens) for language
- D3PM variants for proteins
- DPLM for proteins (pretrained ESM2)

---

## Codebase

### Starting Point

SCUD codebase: https://github.com/AlanNawzadAmin/SCUD

This provides:
- Forward process implementation (uniform, Gaussian, BLOSUM, graph kernels)
- SCUD loss computation (Algorithm 1)
- Sampling (Algorithm 2)
- FiLM layer architecture
- Training loop for CIFAR-10, LM1B, UniRef50
- Pre-computed rate schedules (β_t via MI-based Newton solver)

### Required Modifications

1. **FiLM layers**: Add τ embedding input alongside s (see SemiMarkovFiLM above)
2. **Training loop**: Add Path B (detached prediction, backward transition, TCR loss)
3. **Gradient normalization**: Implement grad norm scaling between losses
4. **Inference**: Add g and τ tracking to sampling loop
5. **SCUM CE loss**: Implement unweighted CE as alternative to ELBO-weighted loss
6. **Metrics**: Self-consistency score, convergence speed tracking

### File Structure (expected)
```
sm-dfm/
├── models/
│   ├── film.py              # SemiMarkovFiLM layer
│   ├── denoiser.py          # x̃_0 predictor (modified from SCUD)
│   └── architectures/       # DiT for language, CARP for proteins, UNet for images
├── training/
│   ├── scum_loss.py          # Unweighted CE (Path A)
│   ├── tcr_loss.py           # Trajectory consistency (Path B)
│   ├── backward_transition.py # Eq. 21 implementation for faithful state generation
│   ├── grad_norm.py          # Gradient normalization
│   └── train.py              # Main training loop
├── inference/
│   ├── sampler.py            # SCUD backward sampling with τ tracking
│   └── metrics.py            # Self-consistency, convergence speed, error correction
├── kernels/
│   ├── uniform.py
│   ├── gaussian.py           # For images
│   ├── blosum.py             # For proteins
│   └── graph.py              # Nearest-neighbor for language
├── data/
│   ├── text8.py
│   ├── lm1b.py
│   └── uniref50.py
├── experiments/
│   ├── phase1_text8.py       # Cheap validation experiments
│   └── phase2_scale.py       # Full-scale experiments
└── eval/
    ├── elbo.py               # SCUD ELBO computation (Algorithm 1) for evaluation
    └── perplexity.py         # PPL from ELBO
```

---

## Key Design Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| SCUM CE instead of ELBO for training | Removes s_t^d weight, gives gradient at s=0 positions, simpler |
| SCUD ELBO for evaluation only | Comparable to prior work (SCUD, MDLM, SEDD) |
| Holding time τ = g - s, not revision counter | Grounded in SCUD schedule clock, same units as s, consistent meaning train/inference |
| Eq. 21 for Path B state generation | Produces states from exact inference distribution, depends on both x_t and x̂_0 |
| Detached gradient for rollout | Avoids BPTT instability, same approach as SUNDAE |
| Gradient normalization for λ | No hyperparameter, adapts automatically as training progresses |
| Non-masking kernels | Structural requirement — masking re-corruption destroys all information |
| FiLM with (s, τ) | Minimal architecture change from SCUD, no new parameters beyond embeddings |

---

## Potential Issues and Mitigations

**τ might not help beyond plain unrolling**: This is the highest risk. If Experiment 3 shows no benefit from τ, the contribution reduces to "SUNDAE for SCUD." Mitigation: run Phase 1 experiments cheaply before committing to full-scale runs.

**Re-corruption from x̂_0 distribution mismatch**: Resolved — use Eq. 21 backward transition which depends on both x_t and x̂_0, matching inference distribution exactly.

**Compound error might be small in practice**: Existing models produce decent samples despite the covariate shift. Mitigation: Experiment 1 measures this directly before any training.

**SCUM CE might hurt ELBO**: Training with unweighted CE instead of ELBO-weighted loss might produce worse ELBO at evaluation. Mitigation: report both metrics. If ELBO is slightly worse but self-consistency is much better, that's a meaningful tradeoff finding.

**Compute overhead**: 3 forward passes per step (1.5x SCUD). Mitigation: one detached pass has no gradient overhead. Actual wall-clock increase is ~40-50%, not 50%.

---

## Framing for Paper

**Title direction**: "Semi-Markov Discrete Flow Matching with Trajectory Consistency Regularization" or "Closing the Train-Inference Gap in Discrete Diffusion with Holding-Time Embeddings"

**Narrative**: SCUD decomposes discrete diffusion into "when" (schedule conditioning) and "where" (learned transitions). But the schedule `s_t^d` is a complete state description only during training. At inference, it's incomplete because it doesn't distinguish forward-process corruption from backward-process predictions. This covariate shift exists at every noise level and causes quadratic error compounding. The Semi-Markov extension adds the holding time τ — the minimal information needed to restore completeness — and TCR trains the model to use it for self-correction.

**Key related work distinction**:
- vs SUNDAE: adds holding time, uses SCUD's schedule conditioning, works within flow matching framework
- vs CLLMs: operates in diffusion/flow matching setting, not autoregressive Jacobi
- vs continuous self-conditioning: discrete tokens require explicit provenance signal because there's no smooth interpolation
- vs SCUD: extends from CTMC to SMP, addresses covariate shift SCUD doesn't handle

---

## Researcher Context

Jake is a Staff Research Associate at UC Davis and ML Engineer at Allergan Aesthetics. He has experience with discrete diffusion (Stateful SCUD / Semi-Markov SCUD extension, PLM architectures with confidence-gated unmasking), preference optimization (AWDPO paper targeting EMNLP via ARR May 2025 deadline), and deployment on constrained hardware (ClinIQ on Jetson). He uses the UC Davis SLURM cluster (A100 GPUs) and RunPod for compute. His advisors are Jörn and Ashwin Nair. This project builds directly on his prior Stateful SCUD / survival embedding work but with a cleaner theoretical foundation.
