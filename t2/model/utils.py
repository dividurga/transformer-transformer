import logging
import typing

import numpy as np
import torch
import torch.nn as nn

from t2.model.core import AdaLnLinear, DiT, DiTBlock, TimestepEmbedder
from t2.model.modality import ModalityAdapter

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Derived from Meta's MAE:
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# licensed under the Attribution-NonCommercial 4.0 International license found
# at https://github.com/facebookresearch/mae/blob/main/LICENSE


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def initialize_weights(model: nn.Module):
    # Initialize transformer layers:
    def _basic_init(module):
        # TODO init layernorm?
        # elif isinstance(module, nn.LayerNorm):
        #     torch.nn.init.zeros_(module.bias)
        #     torch.nn.init.ones_(module.weight)
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, TimestepEmbedder):
            nn.init.normal_(module.linear1.weight, std=0.02)
            nn.init.normal_(module.linear2.weight, std=0.02)
        elif isinstance(module, AdaLnLinear):
            # this is going to be a output projection layer
            adaln_linear = typing.cast(AdaLnLinear, module)
            nn.init.constant_(adaln_linear.adaLN_modulation.linear.weight, 0)
            nn.init.constant_(adaln_linear.adaLN_modulation.linear.bias, 0)
            nn.init.constant_(adaln_linear.linear.weight, 0)
            nn.init.constant_(adaln_linear.linear.bias, 0)
        elif isinstance(module, DiT):
            dit = typing.cast(DiT, module)
            for block in dit.blocks:
                dit_block = typing.cast(DiTBlock, block)
                nn.init.constant_(dit_block.adaln_mod.linear.weight, 0)
                nn.init.constant_(dit_block.adaln_mod.linear.bias, 0)
        elif isinstance(module, ModalityAdapter):
            adapter = typing.cast(ModalityAdapter, module)
            if isinstance(adapter.out_proj, nn.Linear):
                nn.init.constant_(adapter.out_proj.weight, 0)
                nn.init.constant_(adapter.out_proj.bias, 0)
        elif any(
            isinstance(module, batch_norm_cls)
            for batch_norm_cls in {
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            }
        ):
            raise ValueError(f"BatchNorm layer {module} will cause issues with EMA")
        # pos embs of attributes

    model.apply(_basic_init)


def enforce_triangular_inequality(
    x: torch.Tensor,
    ratio: float = 1.0,
    iters: int = 3,
    preprocess_fn: typing.Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    if preprocess_fn is not None:
        x = preprocess_fn(x)
    logging.warning(
        "Deprecating enforce_triangular_inequality, using balanceinertia=True in mjcf model"
    )
    return x
    assert x.shape[-1] == 3
    # python and mujoco float is 64-bit, so we need to use double
    # otherwise, inertias that pass float 32 tests could fail
    # float 64 tests after casting
    x = x.double()
    shape = x.shape
    x = x.reshape(-1, 3)
    for _ in range(iters):
        violate_1 = x[..., 0] + x[..., 1] < x[..., 2]
        x[..., 2] = torch.where(violate_1, (x[..., 0] + x[..., 1]) * ratio, x[..., 2])

        violate_2 = x[..., 0] + x[..., 2] < x[..., 1]
        x[..., 1] = torch.where(violate_2, (x[..., 0] + x[..., 2]) * ratio, x[..., 1])

        violate_3 = x[..., 1] + x[..., 2] < x[..., 0]
        x[..., 0] = torch.where(violate_3, (x[..., 1] + x[..., 2]) * ratio, x[..., 0])
    return x.reshape(shape)


def process_rotmat(x, tol: float = 1e-2):
    shape = x.shape
    if x.shape[-1] == 9:
        x = x.reshape(list(x.shape)[:-1] + [3, 3])
    close_to_identity = (
        torch.isclose(x, torch.eye(3, device=x.device), atol=tol)
        .all(dim=-1)
        .all(dim=-1)
    )
    x[close_to_identity] = torch.eye(3, device=x.device)
    return x.reshape(shape)


def clamp_zero(x, tol: float = 5e-3):
    shape = x.shape
    x = x.reshape(-1, 3)
    mask = torch.linalg.norm(x, dim=-1) < tol
    x[mask] = 0.0
    return x.reshape(shape)
