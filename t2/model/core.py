# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# DiT-derived portions of this file are licensed under the
# Attribution-NonCommercial 4.0 International license found at
# https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from typing import Optional

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    b, _, d = x.shape
    assert shift.shape == scale.shape == (b, d)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
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
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.linear1.weight.dtype
        )
        t_emb = self.linear2(torch.nn.functional.silu(self.linear1(t_freq)))
        return t_emb


class AdaLNMod(nn.Module):
    def __init__(self, hidden_dim: int, n_norm_params: int = 6):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, n_norm_params * hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(x)
        return self.linear(x)


class DiTBlock(nn.Module):
    """
    TODO:
     - try adding qk normalization? [https://arxiv.org/abs/2302.05442]
     - try sandwich normalization

    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        add_zero_attn: bool = False,
        mlp_ratio: float = 4.0,
        mlp_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            add_zero_attn=add_zero_attn,
            add_bias_kv=True,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        # TODO expose mlp to config system
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(mlp_dropout_prob),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(mlp_dropout_prob),
        )
        self.adaln_mod = AdaLNMod(hidden_dim, n_norm_params=6)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim == 3 and key_padding_mask is not None and key_padding_mask.ndim == 3:
            assert key_padding_mask.shape[:2] == x.shape[:2], (
                "key_padding_mask.shape[:2] != x.shape[:2]"
            )
            assert key_padding_mask.shape[2] == 1
            key_padding_mask = key_padding_mask.squeeze(2)
        if cond is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaln_mod(cond).chunk(6, dim=1)
            )
            x = modulate(self.norm1(x), shift_msa, scale_msa)
            x = (
                x
                + gate_msa.unsqueeze(1)
                * self.attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    need_weights=False,
                )[0]
            )
            x = x + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
        else:
            x = self.norm1(x)
            x = (
                x
                + self.attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.mlp(self.norm2(x))
        return x


class AdaLnLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(in_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.adaLN_modulation = AdaLNMod(in_dim, n_norm_params=2)

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if cond is not None:
            shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        attn_dropout: float = 0.0,
        add_zero_attn: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    add_zero_attn=add_zero_attn,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, S, D) sequence of inputs
        cond: (N, D) conditioning
        key_padding_mask: (N, S) mask for padding
        """
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask, cond=cond)  # (N, T, D)
        return x


