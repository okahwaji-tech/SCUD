"""ByteNet-based convolutional architecture for protein sequence diffusion.

Implements ByteNetLMTimeNew, a dilated causal convolution network with
time/schedule conditioning via FiLM modulation, for protein sequence
generation with discrete diffusion.

Reference: "Why Masking Diffusion Works" (NeurIPS 2025).
Code adapted from https://github.com/microsoft/evodiff
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sequence_models.convolutional import MaskedConv1d
from sequence_models.layers import PositionFeedForward
from torch import nn


# function overload
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.compile
def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


class ByteNetLMTimeNew(nn.Module):
    """ByteNet language model with time/schedule conditioning for protein diffusion.

    Stacked dilated residual convolution blocks with FiLM modulation from
    either time embeddings (classical diffusion) or schedule embeddings
    (SCUD). Adapted from the EvoDiff codebase.

    Shape:
       Input: (N, L,)
       input_mask: (N, L, 1), optional
       Output: (N, L, d)

    Args:
        simple_embed: If True, use a single embedding layer; else embed then project.
        n_tokens: Number of tokens in the vocabulary.
        d_aa_emb: Dimension of amino acid embedding (used when simple_embed=False).
        d_embedding: Dimension of conditioning embedding.
        d_model: Hidden dimension of the ByteNet blocks.
        n_layer: Number of dilated residual blocks.
        kernel_size: Convolution kernel width.
        r: Base for dilation factor calculation.
        rank: Rank for compressed weight matrices (None for full rank).
        n_frozen_embs: Number of frozen embedding rows.
        padding_idx: Padding token index in vocabulary.
        causal: If True, use causal convolutions.
        dropout: Dropout rate.
        slim: If True, use half dimensions in feed-forward layers.
        activation: Activation function ('gelu' or 'relu').
        schedule_conditioning: If True, condition on jump schedule S (SCUD mode).
    """

    def __init__(
        self,
        simple_embed: bool = True,
        n_tokens: int = 31,
        d_aa_emb: int = 8,
        d_embedding: int = 128,
        d_model: int = 1024,
        n_layer: int = 16,
        kernel_size: int = 5,
        r: int = 128,
        rank: int | None = None,
        n_frozen_embs: int | None = None,
        padding_idx: int | None = None,
        causal: bool = False,
        dropout: float = 0.1,
        slim: bool = True,
        activation: str = "gelu",
        schedule_conditioning: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.simple_embed = simple_embed
        self.schedule_conditioning = schedule_conditioning
        if not schedule_conditioning:
            self.time_embed_input = TimestepEmbedderNew(2 * d_model)
            self.time_embed_block = TimestepEmbedderNew(d_embedding)
            self.time_embed_input.mlp[2].weight.data.zero_()
            self.time_embed_input.mlp[2].bias.data.zero_()
        if schedule_conditioning:
            self.s_embed_input = TimestepEmbedderNew(2 * d_model)
            self.s_embed_block = TimestepEmbedderNew(d_embedding)
            self.s_embed_input.mlp[2].weight.data.zero_()
            self.s_embed_input.mlp[2].bias.data.zero_()
        if not simple_embed:
            self.embedder = nn.Embedding(n_tokens, d_aa_emb, padding_idx=padding_idx)
            self.up_embedder = nn.Linear(d_aa_emb, d_model)
        else:
            self.embedder = nn.Embedding(n_tokens, d_model, padding_idx=padding_idx)
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layer)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        self.layers = nn.ModuleList(
            [
                ByteNetBlock_wmod(
                    d_model,
                    d_h,
                    d_model,
                    kernel_size,
                    dilation=d,
                    causal=causal,
                    rank=rank,
                    activation=activation,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )
        c_mod_linear_layers = [nn.Linear(d_embedding, 2 * d_h) for d in dilations]
        for lin_layer in c_mod_linear_layers:
            lin_layer.weight.data.zero_()
            lin_layer.bias.data.zero_()
        self.c_mod_layers = nn.ModuleList(c_mod_linear_layers)
        self.dropout = dropout
        self.decoder = PositionFeedForward(d_model, n_tokens)
        self.last_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        input_mask: torch.Tensor | None = None,
        S: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the ByteNet with time/schedule conditioning.

        Args:
            x: Token indices, shape (B, L).
            t: Diffusion time, shape (B,).
            input_mask: Attention mask, shape (B, L).
            S: Schedule tensor for SCUD conditioning, shape (B, L).

        Returns:
            Logits over vocabulary, shape (B, L, n_tokens).
        """
        x = self.embedder(x)
        if not self.simple_embed:
            x = self.up_embedder(x)  # type: ignore[has-type]

        if self.schedule_conditioning:
            assert S is not None
            S_out = F.silu(self.s_embed_input(S.reshape(-1))).reshape(S.shape + (-1,))
            x = modulate_fused(x, *S_out.chunk(2, dim=-1))
            c = F.silu(self.s_embed_block(S.reshape(-1))).reshape(S.shape + (-1,))
        else:
            t_out = F.silu(self.time_embed_input(t))[:, None, :]
            x = modulate_fused(x, *t_out.chunk(2, dim=-1))
            c = F.silu(self.time_embed_block(t))[:, None, :]

        assert input_mask is not None
        for layer, c_layer in zip(self.layers, self.c_mod_layers, strict=True):
            c_mod = c_layer(c)
            x = layer(x, c_mod, input_mask=input_mask.unsqueeze(-1))
        result: torch.Tensor = self.decoder(self.last_norm(x))
        return result


class ByteNetBlock_wmod(nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099).

    Shape:
       Input: (N, L, d_in)
       input_mask: (N, L, 1), optional
       Output: (N, L, d_out)

    """

    def __init__(
        self,
        d_in,
        d_h,
        d_out,
        kernel_size,
        dilation=1,
        groups=1,
        causal=False,
        activation="gelu",
        rank=None,
        dropout=0.0,
    ):
        super().__init__()
        self.conv = MaskedConv1d(
            d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups
        )
        act = nn.GELU
        layers1 = [
            nn.LayerNorm(d_in),
            act(),
            PositionFeedForward(d_in, d_h, rank=rank),
        ]
        layers_mod = [nn.LayerNorm(d_h), act()]
        layers2 = [
            nn.LayerNorm(d_h),
            act(),
            PositionFeedForward(d_h, d_out, rank=rank),
        ]
        self.dropout = dropout
        self.sequence1 = nn.Sequential(*layers1)
        self.sequence_mod = nn.Sequential(*layers_mod)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, c_mod, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        skip_x = x
        x = self.sequence1(x)
        x = modulate_fused(x, *c_mod.chunk(2, dim=-1))
        x = self.sequence_mod(x)
        x = F.dropout(x, self.dropout)
        x = self.conv(x, input_mask=input_mask)
        x = self.sequence2(x)
        return skip_x + x


class TimestepEmbedderNew(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=1280):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = (
            2
            * 3.14159
            * torch.exp(
                -math.log(max_period)
                * (torch.arange(start=0, end=half, dtype=torch.float32) - half / 3)
                / half
            ).to(device=t.device)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
