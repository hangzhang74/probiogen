from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from probiogen.data.fasta import build_window_starts, collect_fasta_files, iter_windows, read_fasta_file
from probiogen.model.tokenizer import CharacterTokenizer
from probiogen.utils import safe_mkdir


def discover_class_fasta_records(base_path: str) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    """Return records with path/label/label_name from class subdirectories."""
    if not os.path.isdir(base_path):
        raise RuntimeError(f"base_path does not exist or is not a directory: {base_path}")
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and not d.startswith(".")]
    if not subdirs:
        raise RuntimeError(f"No class subdirectories found under: {base_path}")
    digit_dirs = [d for d in subdirs if d.isdigit()]
    class_dirs = sorted(digit_dirs, key=lambda x: int(x)) if digit_dirs else sorted(subdirs)
    label_map = {d: i for i, d in enumerate(class_dirs)}
    records: List[Dict[str, Any]] = []
    for d in class_dirs:
        paths = collect_fasta_files(os.path.join(base_path, d), recursive=False)
        for fp in paths:
            records.append({"path": fp, "label": int(label_map[d]), "label_name": d})
    if not records:
        raise RuntimeError(f"No FASTA files found inside class folders under: {base_path}")
    return records, class_dirs, label_map


def cache_paths(out_dir: str, window_size: int, step_size: int) -> Tuple[str, str, str]:
    x_path = os.path.join(out_dir, f"tokenids_ws{window_size}_ss{step_size}_X.npy")
    y_path = os.path.join(out_dir, f"tokenids_ws{window_size}_ss{step_size}_Y.npy")
    meta_path = os.path.join(out_dir, f"tokenids_ws{window_size}_ss{step_size}.meta.npz")
    return x_path, y_path, meta_path


def build_token_cache(
    records: Sequence[Dict[str, Any]],
    tokenizer: CharacterTokenizer,
    out_dir: str,
    window_size: int,
    step_size: int,
    add_special_tokens: bool = False,
    force: bool = False,
) -> Tuple[str, str, str, List[Dict[str, Any]]]:
    """Build disk-backed .npy token arrays and a file_map metadata list."""
    safe_mkdir(out_dir)
    x_path, y_path, meta_path = cache_paths(out_dir, window_size, step_size)
    if (not force) and os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(meta_path):
        try:
            _ = np.load(x_path, mmap_mode="r")
            _ = np.load(y_path, mmap_mode="r")
            meta = np.load(meta_path, allow_pickle=True)
            return x_path, y_path, meta_path, meta["file_map"].tolist()
        except Exception:
            pass

    for p in (x_path, y_path, meta_path):
        if os.path.exists(p):
            os.remove(p)

    counts: List[int] = []
    seq_lens: List[int] = []
    for r in tqdm(records, desc="Counting windows"):
        seq_len = len(read_fasta_file(r["path"]))
        seq_lens.append(seq_len)
        counts.append(len(build_window_starts(seq_len, window_size, step_size)))
    total = int(sum(counts))
    if total <= 0:
        raise RuntimeError("No windows generated from training files.")

    x_dtype = np.uint16 if tokenizer._vocab_str_to_int.get("N", 10) <= 65535 else np.int32
    y_dtype = np.uint16
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=x_dtype, shape=(total, window_size))
    Y = np.lib.format.open_memmap(y_path, mode="w+", dtype=y_dtype, shape=(total,))
    X[:] = np.array(tokenizer.n_id, dtype=x_dtype)
    Y[:] = 0

    file_map: List[Dict[str, Any]] = []
    write_idx = 0
    for r, seq_len in tqdm(list(zip(records, seq_lens)), desc="Building token cache"):
        seq = read_fasta_file(r["path"])
        start_idx = write_idx
        positions: List[Tuple[int, int]] = []
        for start, end, padded, _raw in iter_windows(seq, window_size, step_size):
            X[write_idx] = tokenizer.encode(
                padded,
                max_length=window_size,
                add_special_tokens=add_special_tokens,
                dtype=x_dtype,
            )
            Y[write_idx] = int(r["label"])
            positions.append((int(start), int(end)))
            write_idx += 1
        count = write_idx - start_idx
        if count > 0:
            file_map.append({
                "path": r["path"],
                "label": int(r["label"]),
                "label_name": r.get("label_name", str(r["label"])),
                "start": int(start_idx),
                "count": int(count),
                "positions": positions,
                "seq_len": int(seq_len),
            })

    X.flush(); Y.flush()
    np.savez_compressed(
        meta_path,
        file_map=np.asarray(file_map, dtype=object),
        actual_samples=int(write_idx),
        add_special_tokens=bool(add_special_tokens),
        window_size=int(window_size),
        step_size=int(step_size),
    )
    return x_path, y_path, meta_path, file_map


def split_file_map(
    file_map: Sequence[Dict[str, Any]],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Return train/val/test file indices with stratification when possible."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    file_indices = np.arange(len(file_map))
    labels = np.asarray([int(f["label"]) for f in file_map])

    def can_stratify(y: np.ndarray) -> bool:
        c = Counter(y.tolist())
        return len(c) > 1 and min(c.values()) >= 2

    stratify1 = labels if can_stratify(labels) else None
    idx_train, idx_tmp, _y_train, y_tmp = train_test_split(
        file_indices,
        labels,
        train_size=train_frac,
        random_state=seed,
        stratify=stratify1,
    )
    val_ratio_in_tmp = val_frac / (val_frac + test_frac)
    stratify2 = y_tmp if can_stratify(y_tmp) else None
    idx_val, idx_test, _, _ = train_test_split(
        idx_tmp,
        y_tmp,
        train_size=val_ratio_in_tmp,
        random_state=seed,
        stratify=stratify2,
    )
    return idx_train.tolist(), idx_val.tolist(), idx_test.tolist()


def window_indices_from_files(file_map: Sequence[Dict[str, Any]], file_indices: Sequence[int]) -> np.ndarray:
    parts: List[np.ndarray] = []
    for fi in file_indices:
        f = file_map[int(fi)]
        start, count = int(f["start"]), int(f["count"])
        parts.append(np.arange(start, start + count, dtype=np.int64))
    if not parts:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(parts)
