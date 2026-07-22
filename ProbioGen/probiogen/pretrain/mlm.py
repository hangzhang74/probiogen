from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from probiogen.config import ModelConfig, PretrainConfig
from probiogen.data.fasta import collect_fasta_files, iter_windows, read_fasta_file
from probiogen.model.backbone import LMBackbone
from probiogen.model.tokenizer import CharacterTokenizer
from probiogen.utils import choose_device, limit_cpu_threads, safe_mkdir, save_json, set_seed


class MaskedTokenDataset(Dataset):
    """Memmap token dataset for masked-token pretraining."""

    def __init__(self, x_path: str, mask_token_id: int, pad_token_id: int, mask_prob: float = 0.15, mmap_mode: str = "r"):
        self.x_path = x_path
        self.x = None
        self.mask_token_id = int(mask_token_id)
        self.pad_token_id = int(pad_token_id)
        self.mask_prob = float(mask_prob)
        self.mmap_mode = mmap_mode

    def _ensure_loaded(self):
        if self.x is None:
            self.x = np.load(self.x_path, mmap_mode=self.mmap_mode)

    def __len__(self) -> int:
        self._ensure_loaded()
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        self._ensure_loaded()
        x = torch.from_numpy(np.asarray(self.x[int(idx)], dtype=np.int64)).long()
        labels = x.clone()
        valid = x.ne(self.pad_token_id)
        mask = torch.rand(x.shape).lt(self.mask_prob) & valid
        # Ensure at least one token is masked to avoid empty CE target.
        if not bool(mask.any()) and bool(valid.any()):
            candidates = torch.where(valid)[0]
            mask[candidates[torch.randint(0, len(candidates), (1,))]] = True
        labels[~mask] = -100
        x[mask] = self.mask_token_id
        return x, labels


def pretrain_cache_paths(out_dir: str, window_size: int, step_size: int) -> Tuple[str, str]:
    x_path = os.path.join(out_dir, f"pretrain_tokenids_ws{window_size}_ss{step_size}_X.npy")
    meta_path = os.path.join(out_dir, f"pretrain_tokenids_ws{window_size}_ss{step_size}.meta.npz")
    return x_path, meta_path


def build_pretrain_cache(
    files: Sequence[str],
    tokenizer: CharacterTokenizer,
    out_dir: str,
    window_size: int,
    step_size: int,
    add_special_tokens: bool = False,
    force: bool = False,
) -> Tuple[str, str]:
    safe_mkdir(out_dir)
    x_path, meta_path = pretrain_cache_paths(out_dir, window_size, step_size)
    if (not force) and os.path.exists(x_path) and os.path.exists(meta_path):
        try:
            _ = np.load(x_path, mmap_mode="r")
            return x_path, meta_path
        except Exception:
            pass
    for p in (x_path, meta_path):
        if os.path.exists(p):
            os.remove(p)

    total = 0
    file_counts: List[Dict[str, Any]] = []
    for fp in tqdm(files, desc="Counting pretrain windows"):
        seq = read_fasta_file(fp)
        n = sum(1 for _ in iter_windows(seq, window_size, step_size))
        file_counts.append({"path": fp, "count": int(n)})
        total += n
    if total <= 0:
        raise RuntimeError("No windows generated for pretraining.")

    x_dtype = np.uint16 if tokenizer.n_id <= 65535 else np.int32
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=x_dtype, shape=(total, window_size))
    X[:] = np.array(tokenizer.n_id, dtype=x_dtype)
    write_idx = 0
    for fp in tqdm(files, desc="Building pretrain token cache"):
        seq = read_fasta_file(fp)
        for _start, _end, padded, _raw in iter_windows(seq, window_size, step_size):
            X[write_idx] = tokenizer.encode(
                padded,
                max_length=window_size,
                add_special_tokens=add_special_tokens,
                dtype=x_dtype,
            )
            write_idx += 1
    X.flush()
    np.savez_compressed(
        meta_path,
        files=np.asarray(file_counts, dtype=object),
        actual_samples=int(write_idx),
        add_special_tokens=bool(add_special_tokens),
        window_size=int(window_size),
        step_size=int(step_size),
    )
    return x_path, meta_path


def pretrain_mlm(model_cfg: ModelConfig, pretrain_cfg: PretrainConfig) -> Dict[str, Any]:
    """
    Masked-token pretraining for unlabeled genome FASTA files.

    This is optional. It saves a backbone checkpoint that can later be loaded into supervised training.
    """
    limit_cpu_threads(pretrain_cfg.num_cpu_threads)
    set_seed(pretrain_cfg.seed)
    out_dir = safe_mkdir(pretrain_cfg.out_dir)
    device = choose_device()

    files = collect_fasta_files(pretrain_cfg.input_path, recursive=True)
    if not files:
        raise RuntimeError(f"No FASTA files found under: {pretrain_cfg.input_path}")

    tokenizer = CharacterTokenizer(model_cfg.dna_chars, model_cfg.window_size)
    x_path, meta_path = build_pretrain_cache(
        files=files,
        tokenizer=tokenizer,
        out_dir=out_dir,
        window_size=model_cfg.window_size,
        step_size=pretrain_cfg.step_size,
        add_special_tokens=pretrain_cfg.add_special_tokens,
        force=pretrain_cfg.force_recache,
    )

    ds = MaskedTokenDataset(x_path, tokenizer.mask_id, tokenizer.pad_id, pretrain_cfg.mask_prob)
    loader = DataLoader(ds, batch_size=pretrain_cfg.batch_size, shuffle=True, num_workers=pretrain_cfg.num_workers,
                        pin_memory=torch.cuda.is_available())

    model = LMBackbone(model_cfg, device=device, dtype=torch.float32).to(device)
    lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(lm_head.parameters()), lr=pretrain_cfg.lr, weight_decay=pretrain_cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    history: List[Dict[str, Any]] = []
    ckpt_path = os.path.join(out_dir, "pretrain_checkpoint.pt")
    for epoch in range(1, pretrain_cfg.epochs + 1):
        model.train(); lm_head.train()
        total_loss, total_tokens = 0.0, 0
        for xb, labels in loader:
            xb = xb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reps = model(xb)
            logits = lm_head(reps)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            n_masked = int(labels.ne(-100).sum().item())
            total_loss += float(loss.item()) * max(1, n_masked)
            total_tokens += max(1, n_masked)
        mean_loss = total_loss / max(1, total_tokens)
        history.append({"epoch": epoch, "mlm_loss": mean_loss})
        print(f"Pretrain epoch {epoch:03d} | mlm_loss={mean_loss:.5f}")
        torch.save({
            "model_state": model.state_dict(),
            "mlm_head_state": lm_head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "model_config": asdict(model_cfg),
            "pretrain_config": asdict(pretrain_cfg),
        }, ckpt_path)

    save_json(os.path.join(out_dir, "pretrain_history.json"), history)
    return {"out_dir": out_dir, "checkpoint": ckpt_path, "cache": {"x_path": x_path, "meta_path": meta_path}, "history": history}
