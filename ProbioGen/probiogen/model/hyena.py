from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class Sin(nn.Module):
    def __init__(self, dim: int, w: float = 10, train: bool = True):
        super().__init__()
        if train:
            self.freq = nn.Parameter(w * torch.ones(1, dim))
        else:
            self.register_buffer("freq", w * torch.ones(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int):
        super().__init__()
        bands = max((emb_dim - 1) // 2, 1)
        t = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t / seq_len
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        real_imag = torch.cat([z.real, z.imag], -1)
        combined = torch.cat([t / seq_len, real_imag], -1)
        if combined.size(-1) >= emb_dim:
            combined = combined[..., :emb_dim]
        else:
            pad = torch.zeros(1, seq_len, emb_dim - combined.size(-1))
            combined = torch.cat([combined, pad], -1)
        self.register_buffer("z", combined)
        self.register_buffer("t", t / seq_len)

    def forward(self, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    def __init__(self, d_model: int, fast: float = 0.3, slow: float = 1.5, target: float = 1e-2,
                 modulate: bool = True, shift: float = 0.05):
        super().__init__()
        md, mi = math.log(target) / fast, math.log(target) / slow
        self.register_buffer("deltas", torch.linspace(mi, md, d_model)[None, None])
        self.modulate = modulate
        self.shift = shift

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not self.modulate:
            return x
        decay = torch.exp(-t * self.deltas.abs())
        return x * (decay + self.shift)


class HyenaFilter(nn.Module):
    def __init__(self, d_model: int, emb_dim: int = 3, order: int = 64, seq_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(d_model))
        self.pos = PositionalEmbedding(emb_dim, seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, order),
            Sin(order),
            nn.Linear(order, order),
            Sin(order),
            nn.Linear(order, d_model, bias=False),
        )
        self.mod = ExponentialModulation(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor, L: Optional[int] = None) -> torch.Tensor:
        bsz, L_in, _ = u.shape
        u_fd = u.permute(0, 2, 1)
        z, t = self.pos(L_in)
        z = z.to(device=u.device, dtype=u.dtype)
        t = t.to(device=u.device, dtype=u.dtype)
        k = self.mlp(z)
        fft_size = 2 * L_in
        k_fd = k.permute(0, 2, 1).expand(bsz, -1, -1)
        kf = torch.fft.rfft(k_fd, n=fft_size) / fft_size
        uf = torch.fft.rfft(u_fd, n=fft_size)
        y_fd = torch.fft.irfft(uf * kf, n=fft_size)[..., :L_in]
        x_fd = y_fd + u_fd * self.bias.to(device=u.device, dtype=u.dtype).unsqueeze(-1)
        x_seq = x_fd.permute(0, 2, 1)
        return self.mod(t, self.drop(x_seq))


class HyenaOperator(nn.Module):
    def __init__(self, d_model: int, l_max: int, order: int = 2, filter_order: int = 64,
                 dropout: float = 0.0, drop_filter: float = 0.0):
        super().__init__()
        self.order = order
        self.inner = d_model * (order + 1)
        self.inp = nn.Linear(d_model, self.inner)
        self.short = nn.Conv1d(self.inner, self.inner, 3, padding=2, groups=self.inner)
        self.filter = HyenaFilter(d_model, order=filter_order, seq_len=l_max, dropout=drop_filter)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        _, L, _ = u.shape
        x = self.inp(u).permute(0, 2, 1)
        uc = self.short(x)[..., :L]
        chunk_size = self.inner // (self.order + 1)
        parts = uc.split(chunk_size, dim=1)
        *xs_fd, v_fd = parts
        v_seq = v_fd.permute(0, 2, 1)
        v_seq = self.filter(v_seq, L)
        xs_seq = [xi.permute(0, 2, 1) for xi in xs_fd]
        for xi in xs_seq:
            v_seq = self.drop(v_seq * xi)
        return self.out(v_seq)
