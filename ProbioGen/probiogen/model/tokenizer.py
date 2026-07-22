from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from probiogen.utils import sanitize_dna


class CharacterTokenizer:
    """
    Minimal dependency-free tokenizer compatible with the original token IDs.

    Original special-token IDs:
    [CLS]=0, [SEP]=1, [BOS]=2, [MASK]=3, [PAD]=4, [UNK]=5,
    A/C/G/T/N start from 6.
    """

    def __init__(self, characters: Sequence[str] = ("A", "C", "G", "T", "N"), model_max_length: int = 4096):
        self.characters = tuple(characters)
        self.model_max_length = int(model_max_length)
        self._vocab_str_to_int: Dict[str, int] = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[UNK]": 5,
            **{ch: i + 6 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        self.cls_id = self._vocab_str_to_int["[CLS]"]
        self.sep_id = self._vocab_str_to_int["[SEP]"]
        self.mask_id = self._vocab_str_to_int["[MASK]"]
        self.pad_id = self._vocab_str_to_int["[PAD]"]
        self.unk_id = self._vocab_str_to_int["[UNK]"]
        self.n_id = self._vocab_str_to_int.get("N", self.unk_id)

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
        pad_to_max_length: bool = True,
        dtype: np.dtype = np.int64,
    ) -> np.ndarray:
        max_length = int(max_length or self.model_max_length)
        text = sanitize_dna(text)
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.cls_id)
        ids.extend(self._vocab_str_to_int.get(ch, self.unk_id) for ch in text)
        if add_special_tokens:
            ids.append(self.sep_id)
        if len(ids) > max_length:
            ids = ids[:max_length]
        if pad_to_max_length and len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))
        return np.asarray(ids, dtype=dtype)

    def batch_encode(
        self,
        texts: Sequence[str],
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
        dtype: np.dtype = np.int64,
    ) -> np.ndarray:
        return np.stack([
            self.encode(t, max_length=max_length, add_special_tokens=add_special_tokens, dtype=dtype)
            for t in texts
        ], axis=0)

    def decode(self, ids: Union[np.ndarray, torch.Tensor, Sequence[int]], remove_special: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()
        arr = np.asarray(ids).reshape(-1)
        tokens = [self._vocab_int_to_str.get(int(i), "[UNK]") for i in arr]
        if remove_special:
            tokens = [t for t in tokens if t not in {"[CLS]", "[SEP]", "[PAD]", "[BOS]", "[MASK]", "[UNK]"}]
        return "".join(tokens)
