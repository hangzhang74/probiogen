from __future__ import annotations

import json
import os
import random
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_cpu_threads(num_cpu: int = 10) -> None:
    os.environ["OMP_NUM_THREADS"] = str(num_cpu)
    os.environ["MKL_NUM_THREADS"] = str(num_cpu)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpu)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cpu)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpu)
    torch.set_num_threads(num_cpu)
    try:
        torch.set_num_interop_threads(max(1, num_cpu // 2))
    except RuntimeError:
        # set_num_interop_threads can only be called once per process.
        pass


def safe_mkdir(path: Optional[str]) -> str:
    if path is None or str(path).strip() == "":
        raise ValueError("Output directory is None/empty. Provide a valid path.")
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def choose_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanitize_dna(seq: str) -> str:
    seq = seq.upper()
    return "".join(c if c in {"A", "C", "G", "T", "N"} else "N" for c in seq)
