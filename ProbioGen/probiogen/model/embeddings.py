from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class GPT2Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, max_pos: int, dropout: float = 0.0, device=None, dtype=None):
        super().__init__()
        factory = {k: v for k, v in (("device", device), ("dtype", dtype)) if v is not None}
        self.wte = nn.Embedding(vocab_size, d_model, **factory)
        self.wpe = nn.Embedding(max_pos, d_model, **factory)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        return self.dropout(self.wte(input_ids) + self.wpe(position_ids))
