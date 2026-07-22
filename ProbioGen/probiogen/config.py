from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """Backbone and tokenizer/model-shape configuration."""
    window_size: int = 4096
    d_model: int = 128
    n_layer: int = 16
    d_inner: int = 512
    order: int = 2
    filter_order: int = 64
    dropout: float = 0.4
    drop_filter: float = 0.1
    output_dropout: float = 0.1
    num_heads: int = 8
    attn_layers: Tuple[int, ...] = field(default_factory=tuple)
    dna_chars: Tuple[str, ...] = ("A", "C", "G", "T", "N")
    # Keep 12 for compatibility with the original script, which used 7 + len(dna).
    # The actual tokenizer IDs use 0..10. ID 11 is unused but harmless.
    vocab_size: int = 12
    max_attn_len: int = 4096


@dataclass
class TrainConfig:
    """Supervised training/caching/splitting configuration."""
    base_path: str
    out_dir: Optional[str] = None
    step_size: int = 2048
    batch_size: int = 72
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    patience: int = 20
    seed: int = 42
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    force_recache: bool = False
    num_workers: int = 0
    add_special_tokens: bool = False
    file_agg: str = "mean"
    resume: Optional[str] = None
    save_test_windows: bool = True
    num_cpu_threads: int = 10


@dataclass
class InferenceConfig:
    """External FASTA prediction configuration."""
    checkpoint_path: str
    external_path: str
    external_out: Optional[str] = None
    step_size: int = 2048
    batch_size: int = 32
    add_special_tokens: bool = False
    file_agg: str = "mean"
    device: Optional[str] = None
    num_cpu_threads: int = 10


@dataclass
class PretrainConfig:
    """Masked-token pretraining configuration for unlabeled FASTA files."""
    input_path: str
    out_dir: str
    step_size: int = 2048
    batch_size: int = 72
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01
    mask_prob: float = 0.15
    seed: int = 42
    force_recache: bool = False
    add_special_tokens: bool = False
    num_workers: int = 0
    num_cpu_threads: int = 10
