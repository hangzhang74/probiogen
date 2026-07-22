from __future__ import annotations

import glob
import gzip
import os
from typing import Iterable, List, Tuple

from probiogen.utils import sanitize_dna


def read_fasta_file(fp: str) -> str:
    opener = gzip.open if fp.endswith(".gz") else open
    mode = "rt" if fp.endswith(".gz") else "r"
    seqs: List[str] = []
    with opener(fp, mode) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seqs.append(line.upper())
    return sanitize_dna("".join(seqs))


def collect_fasta_files(root_or_file: str, recursive: bool = True) -> List[str]:
    if os.path.isfile(root_or_file):
        return [root_or_file]
    patterns = ["*.fna", "*.fa", "*.fasta", "*.fsa", "*.fna.gz", "*.fa.gz", "*.fasta.gz", "*.fsa.gz"]
    files: List[str] = []
    for p in patterns:
        if recursive:
            files.extend(glob.glob(os.path.join(root_or_file, "**", p), recursive=True))
        else:
            files.extend(glob.glob(os.path.join(root_or_file, p)))
    return sorted(set(files))


def build_window_starts(seq_len: int, window_size: int, step_size: int) -> List[int]:
    """
    Unified sliding-window function.

    - If seq_len <= window_size: one padded window starting at 0.
    - Otherwise: regular starts by step_size plus a final tail start when needed,
      so the actual genome end is always covered.
    """
    if seq_len <= 0:
        return []
    if seq_len <= window_size:
        return [0]
    max_start = seq_len - window_size
    starts = list(range(0, max_start + 1, step_size))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def iter_windows(seq: str, window_size: int, step_size: int) -> Iterable[Tuple[int, int, str, str]]:
    seq = sanitize_dna(seq)
    L = len(seq)
    for start in build_window_starts(L, window_size, step_size):
        raw = seq[start:min(start + window_size, L)]
        end = start + len(raw) - 1 if raw else start
        padded = raw + "N" * max(0, window_size - len(raw))
        yield int(start), int(end), padded, raw
