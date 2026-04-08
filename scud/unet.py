"""U-Net architecture for image discrete diffusion.

Implements KingmaUNet, a flat (non-downsampling) U-Net with optional
schedule conditioning via sinusoidal S-embeddings and FiLM modulation.
Adapted from D3PM and VDM codebases.

Reference: "Why Masking Diffusion Works" (NeurIPS 2025).
Code adapted from https://github.com/google-research/google-research/tree/master/d3pm
and https://github.com/google-research/vdm
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import nn

MAX_EMBED_SIZE = 10_000


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


class NormalizationLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        num_groups = 32
        num_groups = min(num_channels, num_groups)
        assert num_channels % num_groups == 0
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.norm(x)


def pad_image(x, target_size):
    """Preprocess image to target size with padding."""
    _, _, h, w = x.shape
    if h == target_size and w == target_size:
        return x

    pad_h = max(target_size - h, 0)
    pad_w = max(target_size - w, 0)
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return F.pad(x, padding, mode="constant", value=0)


class ResnetBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, emb_dim, dropout, semb_dim=0, cond=False, film=False
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = NormalizationLayer(in_channels)
        self.norm2 = NormalizationLayer(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.emb_dim = emb_dim
        self.semb_dim = semb_dim
        self.film = film
        if emb_dim > 0:
            self.temb_proj = nn.Linear(emb_dim, out_channels)
            if self.film:
                self.temb_proj_mult = nn.Linear(emb_dim, out_channels)
        if semb_dim > 0:
            self.semb_proj = nn.Linear(semb_dim, out_channels)
            if self.film:
                self.semb_proj_mult = nn.Linear(semb_dim, out_channels)
        if cond:
            self.y_proj = nn.Linear(emb_dim, out_channels)
            if self.film:
                self.y_proj_mult = nn.Linear(emb_dim, out_channels)

        self.shortcut: nn.Module
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb, y, semb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add in timestep embedding
        if self.emb_dim > 0:
            gam = 1 + self.temb_proj_mult(F.silu(temb))[:, :, None, None] if self.film else 1
            bet = self.temb_proj(F.silu(temb))[:, :, None, None]
            h = gam * h + bet

        if self.semb_dim > 0:
            if self.film:
                gam = 1 + self.semb_proj_mult(F.silu(semb.transpose(-1, -3))).transpose(-1, -3)
            else:
                gam = 1
            bet = self.semb_proj(F.silu(semb.transpose(-1, -3))).transpose(-1, -3)
            h = gam * h + bet

        # Add in class embedding
        if y is not None:
            gam = 1 + self.y_proj_mult(y)[:, :, None, None] if self.film else 1
            bet = self.y_proj(y)[:, :, None, None]
            h = gam * h + bet

        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttnBlock(nn.Module):
    def __init__(self, channels, width, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.height = width
        self.width = width
        self.head_dim = channels // self.num_heads
        self.norm = NormalizationLayer(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        B = x.shape[0]
        h = self.norm(x).view(B, self.channels, self.height * self.width).transpose(1, 2)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, -1)
        q = q.view(B, self.height * self.width, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, self.height * self.width, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, self.height * self.width, self.num_heads, self.head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).view(B, self.height, self.width, self.channels)
        h = self.proj_out(h)
        return x + h.transpose(2, 3).transpose(1, 2)


class KingmaUNet(nn.Module):
    """Flat U-Net for image discrete diffusion with optional schedule conditioning.

    A non-downsampling U-Net that processes images with residual blocks and
    optional self-attention. Supports time embedding, class conditioning, and
    SCUD schedule conditioning via sinusoidal S-embeddings with FiLM layers.

    Args:
        n_channel: Number of image channels (1 for MNIST, 3 for CIFAR).
        N: Number of discrete classes per pixel.
        s_lengthscale: Lengthscale for schedule sinusoidal embeddings.
        time_lengthscale: Lengthscale for time sinusoidal embeddings.
        schedule_conditioning: If True, condition on jump schedule S.
        s_dim: Dimension of per-pixel schedule embedding.
        ch: Base channel width.
        time_embed_dim: Dimension of time embedding.
        s_embed_dim: Dimension of schedule embedding (for u_inject/learn_nn styles).
        num_classes: Number of class labels for conditional generation.
        n_layers: Number of residual blocks in each arm of the U-Net.
        inc_attn: If True, include self-attention in residual blocks.
        dropout: Dropout rate.
        num_heads: Number of attention heads.
        n_transformers: Number of transformer blocks in the middle.
        width: Spatial width of the input image.
        not_logistic_pars: If True, add one-hot skip connection to output.
        semb_style: Schedule embedding style ('learn_embed', 'learn_nn', 'u_inject').
        first_mult: If True, apply multiplicative time/schedule modulation to input.
        input_logits: If True, input is logits instead of integer indices.
        film: If True, use FiLM (Feature-wise Linear Modulation) in ResNet blocks.
    """

    def __init__(
        self,
        n_channel: int = 3,
        N: int = 256,
        s_lengthscale: float = 50,
        time_lengthscale: float = 1,
        schedule_conditioning: bool = False,
        s_dim: int = 16,
        ch: int = 128,
        time_embed_dim: int = 128,
        s_embed_dim: int = 128,
        num_classes: int = 1,
        n_layers: int = 32,
        inc_attn: bool = False,
        dropout: float = 0.1,
        num_heads: int = 1,
        n_transformers: int = 1,
        width: int = 32,
        not_logistic_pars: bool = True,
        semb_style: str = "learn_embed",  # "learn_nn", "u_inject"
        first_mult: bool = False,
        input_logits: bool = False,
        film: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__()

        self.first_mult = first_mult
        self.schedule_conditioning = schedule_conditioning
        if schedule_conditioning:
            in_channels = ch * n_channel + n_channel * s_dim

            emb_dim = s_dim // 2
            semb_sin = MAX_EMBED_SIZE ** (-torch.arange(emb_dim) / (emb_dim - 1))
            self.register_buffer("semb_sin", semb_sin)
            if semb_style != "learn_embed":
                self.S_embed_sinusoid: nn.Module = lambda s: torch.cat(  # type: ignore[assignment]
                    [
                        torch.sin(s.reshape(*s.shape, 1) * 1000 * self.semb_sin / s_lengthscale),
                        torch.cos(s.reshape(*s.shape, 1) * 1000 * self.semb_sin / s_lengthscale),
                    ],
                    dim=-1,
                )
                in_channels = ch * n_channel + s_embed_dim
                self.S_embed_nn: nn.Module = nn.Sequential(
                    nn.Linear(n_channel * s_dim, s_embed_dim),
                    nn.SiLU(),
                    nn.Linear(s_embed_dim, s_embed_dim),
                )
                if self.first_mult:
                    self.S_mult_nn = nn.Sequential(
                        nn.Linear(n_channel * s_dim, s_embed_dim),
                        nn.SiLU(),
                        nn.Linear(s_embed_dim, ch * n_channel),
                    )

                # Tau (holding-time) embedding: mirrors S but is additive.
                # Zero-initialized last layer so tau=0 or tau=None recovers
                # standard SCUD exactly.
                self.tau_embed_sinusoid: nn.Module = lambda s: torch.cat(  # type: ignore[assignment]
                    [
                        torch.sin(s.reshape(*s.shape, 1) * 1000 * self.semb_sin / s_lengthscale),
                        torch.cos(s.reshape(*s.shape, 1) * 1000 * self.semb_sin / s_lengthscale),
                    ],
                    dim=-1,
                )
                tau_nn = nn.Sequential(
                    nn.Linear(n_channel * s_dim, s_embed_dim),
                    nn.SiLU(),
                    nn.Linear(s_embed_dim, s_embed_dim),
                )
                nn.init.zeros_(tau_nn[-1].weight)
                nn.init.zeros_(tau_nn[-1].bias)
                self.tau_embed_nn: nn.Module | None = tau_nn
                if self.first_mult:
                    tau_mult = nn.Sequential(
                        nn.Linear(n_channel * s_dim, s_embed_dim),
                        nn.SiLU(),
                        nn.Linear(s_embed_dim, ch * n_channel),
                    )
                    nn.init.zeros_(tau_mult[-1].weight)
                    nn.init.zeros_(tau_mult[-1].bias)
                    self.tau_mult_nn: nn.Module | None = tau_mult
                else:
                    self.tau_mult_nn = None
            else:
                s = torch.arange(MAX_EMBED_SIZE).reshape(-1, 1) * 1000 / s_lengthscale
                semb = torch.cat([torch.sin(s * semb_sin), torch.cos(s * semb_sin)], dim=1)
                s_embed_module = nn.Embedding(MAX_EMBED_SIZE, s_dim)
                s_embed_module.weight.data = semb
                self.S_embed_sinusoid = s_embed_module
                s_embed_dim = 0
                self.S_embed_nn: nn.Module = nn.Identity()  # type: ignore[no-redef]
                self.tau_embed_nn = None
                self.tau_mult_nn = None
            if semb_style != "u_inject":
                s_embed_dim = 0
        else:
            s_embed_dim = 0
            in_channels = ch * n_channel
            self.tau_embed_nn = None
            self.tau_mult_nn = None
        self.N = N
        self.n_channel = n_channel
        out_channels = n_channel * N
        self.ch = ch
        self.n_layers = n_layers
        self.inc_attn = inc_attn
        self.num_classes = num_classes
        self.time_lengthscale = time_lengthscale
        self.width = width
        self.not_logistic_pars = not_logistic_pars

        self.input_logits = input_logits
        self.x_embed: nn.Module
        if not self.input_logits:
            self.x_embed = nn.Embedding(N, ch)
        else:
            self.x_embed = nn.Sequential(
                nn.Linear(n_channel * N, ch * n_channel),
                nn.SiLU(),
                nn.Linear(ch * n_channel, ch * n_channel),
            )
        # Time embedding
        self.time_embed_dim = time_embed_dim
        if self.time_embed_dim > 0:
            self.time_embed = nn.Sequential(
                nn.Linear(ch, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            if self.first_mult:
                self.time_embed_mult_nn = nn.Sequential(
                    nn.Linear(ch, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, ch * n_channel),
                )

        # Class embedding
        self.cond = num_classes > 1
        self.class_embed: nn.Embedding | None
        if self.cond:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.class_embed = None

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        for _i_level in range(self.n_layers):
            block = nn.ModuleList()
            block.append(
                ResnetBlock(ch, ch, time_embed_dim, dropout, s_embed_dim, cond=self.cond, film=film)
            )
            if self.inc_attn:
                block.append(AttnBlock(ch, width, num_heads))
            else:
                block.append(nn.Identity())
            self.down_blocks.append(block)

        # Middle
        self.mid_block1 = ResnetBlock(
            ch, ch, time_embed_dim, dropout, s_embed_dim, cond=self.cond, film=film
        )
        self.mid_attn = nn.Sequential(
            *[AttnBlock(ch, width, num_heads) for i in range(n_transformers)]
        )
        self.mid_block2 = ResnetBlock(
            ch, ch, time_embed_dim, dropout, s_embed_dim, cond=self.cond, film=film
        )

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for _i_level in range(self.n_layers + 1):
            block = nn.ModuleList()
            block.append(
                ResnetBlock(
                    2 * ch, ch, time_embed_dim, dropout, s_embed_dim, cond=self.cond, film=film
                )
            )
            if self.inc_attn:
                block.append(AttnBlock(ch, width, num_heads))
            else:
                block.append(nn.Identity())
            self.up_blocks.append(block)

        self.norm_out = NormalizationLayer(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    @torch.compile()
    def flat_unet(self, x, temb, yemb, semb):
        # Downsampling
        h = self.conv_in(x)
        hs = [h]
        for blocks in self.down_blocks:
            h = blocks[0](h, temb, yemb, semb)
            h = blocks[1](h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, temb, yemb, semb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb, yemb, semb)

        # Upsampling
        for i, blocks in enumerate(self.up_blocks):
            h = blocks[0](torch.cat([h, hs[self.n_layers - (i + 1)]], dim=1), temb, yemb, semb)
            h = blocks[1](h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        S: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: embed inputs, run U-Net, reshape to per-pixel logits.

        Args:
            x: Input tensor, shape (B, C, H, W) of integer indices or logits.
            t: Diffusion time, shape (B,).
            y: Optional class labels, shape (B,).
            S: Optional schedule tensor, shape (B, C, H, W).
            tau: Optional holding-time tensor, shape (B, C, H, W).
                 When provided (and schedule_conditioning is enabled),
                 its embedding is added to the S embedding so the model
                 knows how "stale" each token prediction is.  The last
                 layer is zero-initialized, so tau=0 or tau=None recovers
                 standard SCUD exactly.

        Returns:
            Per-pixel class logits, shape (B, C, H, W, N).
        """
        B, C, H, W, *_ = x.shape
        x_onehot: torch.Tensor | int
        if not self.input_logits:
            x_onehot = F.one_hot(x.long(), num_classes=self.N).float()
            x = self.x_embed(x.permute(0, 2, 3, 1))
            x = x.reshape(*x.shape[:-2], -1).permute(0, 3, 1, 2)
        else:
            x_onehot = 0
            x = (x - x.mean(-1)[..., None]).permute(0, 2, 3, 1, 4)
            x = self.x_embed(x.reshape(*x.shape[:-2], -1)).permute(0, 3, 1, 2)

        # Time embedding
        if self.time_embed_dim > 0:
            t = t.float().reshape(-1, 1) * 1000 / self.time_lengthscale
            emb_dim = self.ch // 2
            temb_sin = MAX_EMBED_SIZE ** (-torch.arange(emb_dim, device=t.device) / (emb_dim - 1))
            temb_sin = torch.cat([torch.sin(t * temb_sin), torch.cos(t * temb_sin)], dim=1)
            temb = self.time_embed(temb_sin)
            if self.first_mult:
                t_mult = self.time_embed_mult_nn(temb_sin)
                x = x * t_mult[:, :, None, None]
        else:
            temb = None

        # S embedding
        if S is not None:
            semb_sin = self.S_embed_sinusoid(S.permute(0, 2, 3, 1))
            semb = self.S_embed_nn(semb_sin.reshape(*semb_sin.shape[:-2], -1)).permute(0, 3, 1, 2)

            # Add tau (holding-time) embedding additively to semb.
            # Zero-init last layer ensures tau=0 or tau=None recovers standard SCUD.
            if tau is not None and (not self.schedule_conditioning or self.tau_embed_nn is None):
                warnings.warn(
                    "tau passed but tau embedding is disabled "
                    "(schedule_conditioning=False or semb_style='learn_embed'). "
                    "tau will be ignored.",
                    stacklevel=2,
                )
            use_tau = (
                tau is not None and self.schedule_conditioning and self.tau_embed_nn is not None
            )
            if use_tau:
                tau_sin = self.tau_embed_sinusoid(tau.permute(0, 2, 3, 1))
                tau_flat = tau_sin.reshape(*tau_sin.shape[:-2], -1)
                semb = semb + self.tau_embed_nn(tau_flat).permute(0, 3, 1, 2)

            if self.first_mult:
                s_mult = self.S_mult_nn(semb_sin.reshape(*semb_sin.shape[:-2], -1)).permute(
                    0, 3, 1, 2
                )
                if use_tau and self.tau_mult_nn is not None:
                    s_mult = s_mult + self.tau_mult_nn(tau_flat).permute(0, 3, 1, 2)
                x = x * s_mult
            x = torch.cat([x, semb], dim=1)
        else:
            semb = None

        # Class embedding
        yemb = (
            self.class_embed(y)
            if y is not None and self.num_classes > 1 and self.class_embed is not None
            else None
        )

        # Reshape output
        h = self.flat_unet(x, temb, yemb, semb)
        h = h[:, :, :H, :W].reshape(B, C, self.N, H, W).permute((0, 1, 3, 4, 2))
        out: torch.Tensor = h + self.not_logistic_pars * x_onehot
        return out
