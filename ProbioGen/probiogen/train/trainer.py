from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from torch import nn

from probiogen.config import ModelConfig, TrainConfig
from probiogen.data.cache import build_token_cache, discover_class_fasta_records, split_file_map, window_indices_from_files
from probiogen.data.dataset import make_loader
from probiogen.model.build import build_model_and_tokenizer
from probiogen.model.tokenizer import CharacterTokenizer
from probiogen.train.engine import (
    aggregate_file_map_predictions,
    compute_file_metrics,
    evaluate_windows,
    export_window_predictions_csv,
    predict_probs_for_indices,
    save_checkpoint,
    save_file_results_csv,
    train_one_epoch,
)
from probiogen.utils import choose_device, limit_cpu_threads, safe_mkdir, save_json, set_seed


def train_finetune(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Dict[str, Any]:
    """
    End-to-end importable supervised finetuning function.

    Returns checkpoint paths, cache paths, class mapping and window/file metrics.
    """
    limit_cpu_threads(train_cfg.num_cpu_threads)
    set_seed(train_cfg.seed)
    out_dir = safe_mkdir(train_cfg.out_dir or train_cfg.base_path)
    device = choose_device()

    records, class_dirs, label_map = discover_class_fasta_records(train_cfg.base_path)
    num_classes = len(class_dirs)
    tokenizer = CharacterTokenizer(model_cfg.dna_chars, model_cfg.window_size)

    x_path, y_path, meta_path, file_map = build_token_cache(
        records=records,
        tokenizer=tokenizer,
        out_dir=out_dir,
        window_size=model_cfg.window_size,
        step_size=train_cfg.step_size,
        add_special_tokens=train_cfg.add_special_tokens,
        force=train_cfg.force_recache,
    )

    train_files, val_files, test_files = split_file_map(
        file_map,
        train_frac=train_cfg.train_frac,
        val_frac=train_cfg.val_frac,
        test_frac=train_cfg.test_frac,
        seed=train_cfg.seed,
    )
    train_idx = window_indices_from_files(file_map, train_files)
    val_idx = window_indices_from_files(file_map, val_files)
    test_idx = window_indices_from_files(file_map, test_files)

    train_loader = make_loader(x_path, y_path, train_idx, train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    val_loader = make_loader(x_path, y_path, val_idx, train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)
    test_loader = make_loader(x_path, y_path, test_idx, train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)

    _, model, clf = build_model_and_tokenizer(model_cfg, num_classes=num_classes, device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(clf.parameters()), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 1
    if train_cfg.resume and os.path.isfile(train_cfg.resume):
        ck = torch.load(train_cfg.resume, map_location=device)
        if ck.get("model_state") is not None:
            model.load_state_dict(ck["model_state"], strict=False)
        if ck.get("clf_state") is not None:
            clf.load_state_dict(ck["clf_state"], strict=False)
        if ck.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(ck["optimizer_state"])
            except Exception:
                pass
        start_epoch = int(ck.get("epoch", 0)) + 1

    best_val_f1 = -1.0
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    best_ckpt = os.path.join(out_dir, "best_checkpoint.pt")
    last_ckpt = os.path.join(out_dir, "last_checkpoint.pt")

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        train_loss = train_one_epoch(model, clf, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate_windows(model, clf, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(train_loss), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"Epoch {epoch:03d} | loss={train_loss:.5f} | val_f1={val_metrics['f1_macro']:.5f} | val_mcc={val_metrics['mcc']:.5f}")

        save_checkpoint(last_ckpt, model, clf, optimizer, epoch, model_cfg, train_cfg, class_dirs, label_map, val_metrics)
        cur_f1 = float(val_metrics.get("f1_macro") or 0.0)
        if cur_f1 > best_val_f1:
            best_val_f1 = cur_f1
            bad_epochs = 0
            save_checkpoint(best_ckpt, model, clf, optimizer, epoch, model_cfg, train_cfg, class_dirs, label_map, val_metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= train_cfg.patience:
                print(f"Early stopping at epoch {epoch}: no val F1 improvement for {train_cfg.patience} epochs.")
                break

    ck_best = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ck_best["model_state"])
    clf.load_state_dict(ck_best["clf_state"])
    test_window_metrics = evaluate_windows(model, clf, test_loader, device)

    test_probs = predict_probs_for_indices(model, clf, x_path, test_idx, device, batch_size=train_cfg.batch_size)
    test_file_results = aggregate_file_map_predictions(test_probs, test_idx, file_map, file_indices=test_files, method=train_cfg.file_agg)
    test_file_metrics = compute_file_metrics(test_file_results)

    save_json(os.path.join(out_dir, "history.json"), history)
    save_json(os.path.join(out_dir, "split_summary.json"), {
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "class_dirs": class_dirs,
        "label_map": label_map,
        "n_train_windows": int(len(train_idx)),
        "n_val_windows": int(len(val_idx)),
        "n_test_windows": int(len(test_idx)),
    })
    save_json(os.path.join(out_dir, "test_metrics.json"), {
        "window_level": test_window_metrics,
        "file_level": test_file_metrics,
    })
    save_file_results_csv(os.path.join(out_dir, "test_file_level_results.csv"), test_file_results, include_label=True)

    if train_cfg.save_test_windows:
        export_window_predictions_csv(
            os.path.join(out_dir, "test_window_level_results_with_positions.csv"),
            test_probs,
            test_idx,
            file_map,
            include_sequences=True,
        )

    return {
        "out_dir": out_dir,
        "best_checkpoint": best_ckpt,
        "last_checkpoint": last_ckpt,
        "cache": {"x_path": x_path, "y_path": y_path, "meta_path": meta_path},
        "window_level_metrics": test_window_metrics,
        "file_level_metrics": test_file_metrics,
        "class_dirs": class_dirs,
        "label_map": label_map,
    }
