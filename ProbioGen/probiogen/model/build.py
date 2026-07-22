from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from probiogen.config import ModelConfig
from probiogen.model.backbone import LMBackbone
from probiogen.model.heads import ClassificationHead
from probiogen.model.tokenizer import CharacterTokenizer
from probiogen.utils import choose_device


def build_model_and_tokenizer(
    model_cfg: Optional[ModelConfig] = None,
    num_classes: int = 2,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[CharacterTokenizer, LMBackbone, ClassificationHead]:
    cfg = model_cfg or ModelConfig()
    dev = torch.device(device) if device is not None else choose_device()
    tokenizer = CharacterTokenizer(cfg.dna_chars, cfg.window_size)
    model = LMBackbone(cfg, device=dev, dtype=dtype).to(dev)
    clf = ClassificationHead(cfg.d_model, num_classes=num_classes, dropout_prob=cfg.dropout).to(dev)
    return tokenizer, model, clf
