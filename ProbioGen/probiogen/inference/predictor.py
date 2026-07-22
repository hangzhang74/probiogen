from __future__ import annotations

import csv
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from probiogen.config import InferenceConfig, ModelConfig
from probiogen.data.fasta import collect_fasta_files, iter_windows, read_fasta_file
from probiogen.model.backbone import LMBackbone
from probiogen.model.build import build_model_and_tokenizer
from probiogen.model.heads import ClassificationHead
from probiogen.model.tokenizer import CharacterTokenizer
from probiogen.train.engine import aggregate_probs, forward_logits, save_file_results_csv
from probiogen.utils import choose_device, limit_cpu_threads, safe_mkdir, sanitize_dna, save_json


def _model_config_from_checkpoint(ck: Dict[str, Any], override: Optional[ModelConfig] = None) -> ModelConfig:
    if override is not None:
        return override
    cfg_raw = ck.get("model_config")
    if isinstance(cfg_raw, dict):
        cfg_raw = dict(cfg_raw)
        if isinstance(cfg_raw.get("attn_layers"), list):
            cfg_raw["attn_layers"] = tuple(cfg_raw["attn_layers"])
        if isinstance(cfg_raw.get("dna_chars"), list):
            cfg_raw["dna_chars"] = tuple(cfg_raw["dna_chars"])
        return ModelConfig(**{k: v for k, v in cfg_raw.items() if k in ModelConfig.__dataclass_fields__})
    return ModelConfig()


def load_checkpoint_for_inference(
    checkpoint_path: str,
    device: Optional[Union[str, torch.device]] = None,
    model_cfg: Optional[ModelConfig] = None,
) -> Tuple[CharacterTokenizer, LMBackbone, ClassificationHead, ModelConfig, Dict[str, Any]]:
    dev = torch.device(device) if device is not None else choose_device()
    ck = torch.load(checkpoint_path, map_location=dev)
    cfg = _model_config_from_checkpoint(ck, override=model_cfg)
    clf_state = ck.get("clf_state")
    if clf_state is None or "fc.weight" not in clf_state:
        raise RuntimeError("Checkpoint must contain clf_state['fc.weight'] to infer num_classes.")
    num_classes = int(clf_state["fc.weight"].shape[0])
    tokenizer, model, clf = build_model_and_tokenizer(cfg, num_classes=num_classes, device=dev, dtype=torch.float32)
    if ck.get("model_state") is None:
        raise RuntimeError("Checkpoint has no model_state.")
    model.load_state_dict(ck["model_state"], strict=False)
    clf.load_state_dict(clf_state, strict=True)
    model.eval(); clf.eval()
    return tokenizer, model, clf, cfg, ck


def predict_sequence_windows(
    seq: str,
    tokenizer: CharacterTokenizer,
    model: LMBackbone,
    clf: ClassificationHead,
    device: Union[str, torch.device],
    window_size: int,
    step_size: int,
    batch_size: int = 32,
    add_special_tokens: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    dev = torch.device(device)
    seq = sanitize_dna(seq)
    positions: List[Tuple[int, int]] = []
    windows: List[str] = []
    for start, end, padded, _raw in iter_windows(seq, window_size, step_size):
        positions.append((start, end))
        windows.append(padded)

    if not windows:
        return np.zeros((0, clf.fc.out_features), dtype=np.float32), np.zeros((0, clf.fc.out_features), dtype=np.float32), []

    logits_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    model.eval(); clf.eval()
    with torch.no_grad():
        for st in range(0, len(windows), batch_size):
            batch_windows = windows[st:st + batch_size]
            x_np = tokenizer.batch_encode(
                batch_windows,
                max_length=window_size,
                add_special_tokens=add_special_tokens,
                dtype=np.int64,
            )
            xb = torch.from_numpy(x_np).long().to(dev)
            logits = forward_logits(model, clf, xb)
            probs = torch.softmax(logits, dim=-1)
            logits_all.append(logits.cpu().numpy())
            probs_all.append(probs.cpu().numpy())

    return np.concatenate(logits_all, axis=0), np.concatenate(probs_all, axis=0), positions


def predict_fasta_files(
    infer_cfg: InferenceConfig,
    model_cfg: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """External FASTA/folder prediction. Writes window-level and file-level CSVs."""
    limit_cpu_threads(infer_cfg.num_cpu_threads)
    device = choose_device(infer_cfg.device)

    tokenizer, model, clf, cfg, ck = load_checkpoint_for_inference(infer_cfg.checkpoint_path, device=device, model_cfg=model_cfg)

    files = collect_fasta_files(infer_cfg.external_path, recursive=True)
    if not files:
        raise RuntimeError(f"No FASTA files found under: {infer_cfg.external_path}")

    external_out = infer_cfg.external_out
    if external_out is None:
        base = infer_cfg.external_path if os.path.isdir(infer_cfg.external_path) else os.path.dirname(infer_cfg.external_path)
        external_out = os.path.join(base, "external_eval_out")
    safe_mkdir(external_out)

    win_csv = os.path.join(external_out, "external_window_level_results.csv")
    file_csv = os.path.join(external_out, "external_file_level_results.csv")

    file_rows: List[Dict[str, Any]] = []
    with open(win_csv, "w", newline="", encoding="utf-8") as fwin:
        wwin = csv.writer(fwin)
        wwin.writerow([
            "file_index", "file_path", "window_start", "window_end",
            "pred_label", "pred_prob_max", "per_class_probs", "window_sequence",
        ])
        for file_index, fp in enumerate(tqdm(files, desc="External inference")):
            try:
                seq = read_fasta_file(fp)
            except Exception as e:
                print(f"Warning: failed to read {fp}: {e}")
                file_rows.append({
                    "file_index": file_index,
                    "path": fp,
                    "pred_label": -1,
                    "pred_prob": [],
                    "n_windows": 0,
                    "n_seen_windows": 0,
                    "votes": {},
                })
                continue
            logits, probs, positions = predict_sequence_windows(
                seq=seq,
                tokenizer=tokenizer,
                model=model,
                clf=clf,
                device=device,
                window_size=cfg.window_size,
                step_size=infer_cfg.step_size,
                batch_size=infer_cfg.batch_size,
                add_special_tokens=infer_cfg.add_special_tokens,
            )
            for (start, end), pv in zip(positions, probs):
                wwin.writerow([
                    int(file_index),
                    fp,
                    int(start),
                    int(end),
                    int(np.argmax(pv)),
                    f"{float(np.max(pv)):.6f}",
                    ",".join(f"{float(x):.6f}" for x in pv.tolist()),
                    seq[int(start):int(end) + 1],
                ])
            pred, p_file, votes = aggregate_probs(probs, method=infer_cfg.file_agg)
            file_rows.append({
                "file_index": file_index,
                "path": fp,
                "pred_label": int(pred),
                "pred_prob": p_file.tolist() if p_file.size else [],
                "n_windows": int(len(probs)),
                "n_seen_windows": int(len(probs)),
                "votes": votes,
            })

    save_file_results_csv(file_csv, file_rows, include_label=False)
    save_json(os.path.join(external_out, "inference_config.json"), {
        "inference_config": asdict(infer_cfg),
        "model_config": asdict(cfg),
        "class_dirs": ck.get("class_dirs"),
        "label_map": ck.get("label_map"),
    })
    return {"external_out": external_out, "window_csv": win_csv, "file_csv": file_csv, "rows": file_rows}
