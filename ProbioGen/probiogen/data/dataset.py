from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenMemmapDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, indices: Sequence[int], mmap_mode: str = "r"):
        self.x_path = x_path
        self.y_path = y_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mmap_mode = mmap_mode
        self.x = None
        self.y = None

    def _ensure_loaded(self) -> None:
        if self.x is None:
            self.x = np.load(self.x_path, mmap_mode=self.mmap_mode)
        if self.y is None:
            self.y = np.load(self.y_path, mmap_mode=self.mmap_mode)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_loaded()
        idx = int(self.indices[i])
        x = np.asarray(self.x[idx], dtype=np.int64)
        y = int(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def make_loader(x_path: str, y_path: str, indices: Sequence[int], batch_size: int, shuffle: bool,
                num_workers: int = 0) -> DataLoader:
    ds = TokenMemmapDataset(x_path, y_path, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())
