from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from probiogen.config import ModelConfig, TrainConfig
from probiogen.data.fasta import read_fasta_file
from probiogen.model.backbone import LMBackbone
from probiogen.model.heads import ClassificationHead
from probiogen.train.metrics import metrics_from_predictions


def forward_logits(model: LMBackbone, clf: ClassificationHead, x: torch.Tensor) -> torch.Tensor:
    reps = model(x.long())
    pooled = reps.mean(dim=1)
    return clf(pooled)


def train_one_epoch(model: LMBackbone, clf: ClassificationHead, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device) -> float:
    model.train(); clf.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_logits(model, clf, xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * int(xb.size(0))
        n += int(xb.size(0))
    return total / max(1, n)


def evaluate_windows(model: LMBackbone, clf: ClassificationHead, loader: DataLoader,
                     device: torch.device) -> Dict[str, Any]:
    model.eval(); clf.eval()
    ys: List[int] = []
    ps: List[List[float]] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = forward_logits(model, clf, xb)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            ys.extend(yb.numpy().astype(int).tolist())
            ps.extend(probs.tolist())
    y = np.asarray(ys, dtype=int)
    p = np.asarray(ps, dtype=float)
    pred = p.argmax(axis=1) if len(p) else np.asarray([], dtype=int)
    return metrics_from_predictions(y, pred, p)


def predict_probs_for_indices(model: LMBackbone, clf: ClassificationHead, x_path: str, indices: Sequence[int],
                              device: torch.device, batch_size: int = 64) -> np.ndarray:
    X = np.load(x_path, mmap_mode="r")
    idx = np.asarray(indices, dtype=np.int64)
    probs_list: List[np.ndarray] = []
    model.eval(); clf.eval()
    with torch.no_grad():
        for st in tqdm(range(0, len(idx), batch_size), desc="Predict windows"):
            batch_idx = idx[st:st + batch_size]
            xb = torch.from_numpy(np.asarray(X[batch_idx], dtype=np.int64)).long().to(device)
            logits = forward_logits(model, clf, xb)
            probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
    if not probs_list:
        return np.zeros((0, clf.fc.out_features), dtype=float)
    return np.concatenate(probs_list, axis=0)


def aggregate_probs(probs: np.ndarray, method: str = "mean") -> Tuple[int, np.ndarray, Dict[int, int]]:
    """
    Aggregate window-level probabilities into one file-level prediction.

    method:
      - mean: average probabilities, then argmax
      - max: class-wise max probabilities, then argmax
      - vote: window argmax majority vote; tie is broken by mean probability
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return -1, np.asarray([], dtype=float), {}
    method = method.lower()
    if method == "mean":
        p = probs.mean(axis=0)
        return int(np.argmax(p)), p, dict(Counter(probs.argmax(axis=1).astype(int).tolist()))
    if method == "max":
        p = probs.max(axis=0)
        return int(np.argmax(p)), p, dict(Counter(probs.argmax(axis=1).astype(int).tolist()))
    if method == "vote":
        win_pred = probs.argmax(axis=1).astype(int)
        vote_counter = Counter(win_pred.tolist())
        max_votes = max(vote_counter.values())
        tied = sorted([c for c, v in vote_counter.items() if v == max_votes])
        if len(tied) == 1:
            pred = tied[0]
        else:
            means = {c: float(probs[:, c].mean()) for c in tied}
            pred = max(means.items(), key=lambda kv: kv[1])[0]
        return int(pred), probs.mean(axis=0), dict(vote_counter)
    raise ValueError("method must be one of: mean, max, vote")


def aggregate_file_map_predictions(
    probs_for_indices: np.ndarray,
    indices: Sequence[int],
    file_map: Sequence[Dict[str, Any]],
    file_indices: Optional[Sequence[int]] = None,
    method: str = "mean",
) -> List[Dict[str, Any]]:
    idx = np.asarray(indices, dtype=np.int64)
    prob_by_idx = {int(i): probs_for_indices[k] for k, i in enumerate(idx.tolist())}
    selected_files = list(range(len(file_map))) if file_indices is None else [int(i) for i in file_indices]
    results: List[Dict[str, Any]] = []
    idx_set = set(idx.tolist())
    for fi in selected_files:
        f = file_map[fi]
        start, count = int(f["start"]), int(f["count"])
        f_indices = [i for i in range(start, start + count) if i in idx_set]
        if not f_indices:
            continue
        win_probs = np.stack([prob_by_idx[i] for i in f_indices], axis=0)
        pred, p_file, votes = aggregate_probs(win_probs, method=method)
        results.append({
            "file_index": int(fi),
            "path": f["path"],
            "label": int(f.get("label", -1)),
            "label_name": f.get("label_name", str(f.get("label", ""))),
            "pred_label": int(pred),
            "pred_prob": p_file.tolist(),
            "n_windows": int(count),
            "n_seen_windows": int(len(f_indices)),
            "votes": votes,
        })
    return results


def compute_file_metrics(file_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    y = np.asarray([int(r["label"]) for r in file_results if int(r.get("label", -1)) >= 0], dtype=int)
    p = np.asarray([int(r["pred_label"]) for r in file_results if int(r.get("label", -1)) >= 0], dtype=int)
    return metrics_from_predictions(y, p)


def save_file_results_csv(path: str, rows: Sequence[Dict[str, Any]], include_label: bool = True) -> None:
    fields = ["file_index", "path"]
    if include_label:
        fields += ["label", "label_name"]
    fields += ["pred_label", "pred_prob", "n_windows", "n_seen_windows", "votes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fields}
            if isinstance(out.get("pred_prob"), (list, tuple, np.ndarray)):
                out["pred_prob"] = ",".join(f"{float(x):.6f}" for x in out["pred_prob"])
            if isinstance(out.get("votes"), dict):
                out["votes"] = json.dumps(out["votes"], ensure_ascii=False)
            w.writerow(out)


def save_checkpoint(
    path: str,
    model: LMBackbone,
    clf: ClassificationHead,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    class_dirs: Sequence[str],
    label_map: Dict[str, int],
    metrics: Dict[str, Any],
) -> None:
    ck = {
        "model_state": model.state_dict(),
        "clf_state": clf.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "class_dirs": list(class_dirs),
        "label_map": dict(label_map),
        "metrics": metrics,
    }
    torch.save(ck, path)


def export_window_predictions_csv(
    path: str,
    probs: np.ndarray,
    indices: Sequence[int],
    file_map: Sequence[Dict[str, Any]],
    include_sequences: bool = True,
) -> None:
    idx = np.asarray(indices, dtype=np.int64)
    file_lookup: List[Tuple[int, Dict[str, Any]]] = [(fi, f) for fi, f in enumerate(file_map)]
    fields = [
        "global_index", "file_index", "file_path", "label", "window_start", "window_end",
        "pred_label", "pred_prob_max", "per_class_probs",
    ]
    if include_sequences:
        fields.append("window_sequence")
    with open(path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()
        seq_cache: Dict[str, str] = {}
        for gi, pv in zip(idx.tolist(), probs):
            file_info = None
            file_index = -1
            local = -1
            for fi, fm in file_lookup:
                s, c = int(fm["start"]), int(fm["count"])
                if s <= gi < s + c:
                    file_info = fm
                    file_index = fi
                    local = gi - s
                    break
            if file_info is None:
                continue
            w_start, w_end = file_info.get("positions", [(-1, -1)])[local]
            row = {
                "global_index": int(gi),
                "file_index": int(file_index),
                "file_path": file_info["path"],
                "label": int(file_info.get("label", -1)),
                "window_start": int(w_start),
                "window_end": int(w_end),
                "pred_label": int(np.argmax(pv)),
                "pred_prob_max": f"{float(np.max(pv)):.6f}",
                "per_class_probs": ",".join(f"{float(x):.6f}" for x in pv.tolist()),
            }
            if include_sequences:
                fp = file_info["path"]
                if fp not in seq_cache:
                    seq_cache[fp] = read_fasta_file(fp)
                seq = seq_cache[fp]
                row["window_sequence"] = seq[int(w_start):int(w_end) + 1] if int(w_start) >= 0 else ""
            w.writerow(row)
