from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn

from probiogen.config import ModelConfig
from probiogen.model.embeddings import GPT2Embeddings
from probiogen.model.hyena import HyenaOperator


def create_mixer_cls(cfg: ModelConfig, layer_idx: int, device=None, dtype=None):
    use_attn = bool(cfg.attn_layers and layer_idx in set(cfg.attn_layers) and cfg.window_size <= cfg.max_attn_len)
    if use_attn:
        return partial(
            nn.MultiheadAttention,
            embed_dim=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
    return partial(
        HyenaOperator,
        d_model=cfg.d_model,
        l_max=cfg.window_size,
        order=cfg.order,
        filter_order=cfg.filter_order,
        dropout=cfg.dropout,
        drop_filter=cfg.drop_filter,
    )


class Block(nn.Module):
    def __init__(self, d_model: int, mixer_cls, device=None, dtype=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.mixer = mixer_cls()
        self.drop1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, device=device, dtype=dtype),
        )
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, res: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.mixer, nn.MultiheadAttention):
            normed = self.norm1(x)
            h, _ = self.mixer(normed, normed, normed)
        else:
            h = self.mixer(self.norm1(x))
        x = (x if res is None else res) + self.drop1(h)
        h2 = self.mlp(self.norm2(x))
        return x + self.drop2(h2), x


class LMBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig, device=None, dtype=None):
        super().__init__()
        self.cfg = cfg
        self.emb = GPT2Embeddings(cfg.d_model, cfg.vocab_size, cfg.window_size, dropout=cfg.dropout, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, create_mixer_cls(cfg, i, device=device, dtype=dtype), device=device, dtype=dtype)
            for i in range(cfg.n_layer)
        ])
        self.drop = nn.Dropout(cfg.output_dropout)
        self.ln = nn.LayerNorm(cfg.d_model, device=device, dtype=dtype)
        self.apply(partial(_init_weights, n_layer=cfg.n_layer))

    def forward(self, input_ids: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, res = self.emb(input_ids, positions), None
        for block in self.blocks:
            x, res = block(x, res)
        if res is None:
            return self.ln(self.drop(x))
        return self.ln(self.drop(x) + res)


def _init_weights(module: nn.Module, n_layer: int, initializer_range: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
