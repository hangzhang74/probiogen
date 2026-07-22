from __future__ import annotations

import argparse
import json
from typing import Any, List, Optional, Sequence, Tuple

from probiogen.config import InferenceConfig, ModelConfig, PretrainConfig, TrainConfig
from probiogen.inference import predict_fasta_files
from probiogen.pretrain import pretrain_mlm
from probiogen.train import train_finetune


def _parse_attn_layers(values: Optional[List[int]]) -> Tuple[int, ...]:
    return tuple(values or [])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ProbioGen: genome-level probiotic prediction with window-level interpretation")
    sub = p.add_subparsers(dest="command", required=True)

    common_model = argparse.ArgumentParser(add_help=False)
    common_model.add_argument("--window_size", type=int, default=4096)
    common_model.add_argument("--d_model", type=int, default=128)
    common_model.add_argument("--n_layer", type=int, default=16)
    common_model.add_argument("--d_inner", type=int, default=512)
    common_model.add_argument("--dropout", type=float, default=0.4)
    common_model.add_argument("--drop_filter", type=float, default=0.1)
    common_model.add_argument("--order", type=int, default=2)
    common_model.add_argument("--filter_order", type=int, default=64)
    common_model.add_argument("--num_heads", type=int, default=8)
    common_model.add_argument("--attn_layers", nargs="*", type=int, default=[])

    pre = sub.add_parser("pretrain", parents=[common_model], help="Masked-token pretraining from unlabeled FASTA files")
    pre.add_argument("--input_path", required=True)
    pre.add_argument("--out_dir", required=True)
    pre.add_argument("--step_size", type=int, default=2048)
    pre.add_argument("--epochs", type=int, default=10)
    pre.add_argument("--batch_size", type=int, default=72)
    pre.add_argument("--lr", type=float, default=1e-4)
    pre.add_argument("--mask_prob", type=float, default=0.15)
    pre.add_argument("--force_recache", action="store_true")
    pre.add_argument("--add_special_tokens", action="store_true")
    pre.add_argument("--num_workers", type=int, default=0)
    pre.add_argument("--num_cpu_threads", type=int, default=10)

    tr = sub.add_parser("train", parents=[common_model], help="Supervised finetuning from class folders")
    tr.add_argument("--base_path", required=True)
    tr.add_argument("--out_dir", default=None)
    tr.add_argument("--step_size", type=int, default=2048)
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch_size", type=int, default=72)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--patience", type=int, default=20)
    tr.add_argument("--force_recache", action="store_true")
    tr.add_argument("--add_special_tokens", action="store_true", help="Use old tokenizer behavior with [CLS]/[SEP].")
    tr.add_argument("--file_agg", choices=["mean", "max", "vote"], default="mean")
    tr.add_argument("--resume", default=None, help="Checkpoint to resume from. Can be a pretrain checkpoint for backbone init.")
    tr.add_argument("--num_workers", type=int, default=0)
    tr.add_argument("--num_cpu_threads", type=int, default=10)

    pr = sub.add_parser("predict", help="Predict external FASTA/folder")
    pr.add_argument("--external_path", required=True)
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--external_out", default=None)
    pr.add_argument("--step_size", type=int, default=2048)
    pr.add_argument("--batch_size", type=int, default=32)
    pr.add_argument("--file_agg", choices=["mean", "max", "vote"], default="mean")
    pr.add_argument("--add_special_tokens", action="store_true", help="Use this for old checkpoints trained with [CLS]/[SEP].")
    pr.add_argument("--device", default=None)
    pr.add_argument("--num_cpu_threads", type=int, default=10)
    return p


def _model_cfg_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        window_size=args.window_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        d_inner=args.d_inner,
        order=args.order,
        filter_order=args.filter_order,
        dropout=args.dropout,
        drop_filter=args.drop_filter,
        num_heads=args.num_heads,
        attn_layers=_parse_attn_layers(args.attn_layers),
    )


def main(argv: Optional[Sequence[str]] = None) -> Any:
    args = build_argparser().parse_args(argv)
    if args.command == "pretrain":
        model_cfg = _model_cfg_from_args(args)
        pretrain_cfg = PretrainConfig(
            input_path=args.input_path,
            out_dir=args.out_dir,
            step_size=args.step_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            mask_prob=args.mask_prob,
            force_recache=args.force_recache,
            add_special_tokens=args.add_special_tokens,
            num_workers=args.num_workers,
            num_cpu_threads=args.num_cpu_threads,
        )
        result = pretrain_mlm(model_cfg, pretrain_cfg)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return result

    if args.command == "train":
        model_cfg = _model_cfg_from_args(args)
        train_cfg = TrainConfig(
            base_path=args.base_path,
            out_dir=args.out_dir,
            step_size=args.step_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            force_recache=args.force_recache,
            add_special_tokens=args.add_special_tokens,
            file_agg=args.file_agg,
            resume=args.resume,
            num_workers=args.num_workers,
            num_cpu_threads=args.num_cpu_threads,
        )
        result = train_finetune(model_cfg, train_cfg)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return result

    if args.command == "predict":
        infer_cfg = InferenceConfig(
            checkpoint_path=args.checkpoint,
            external_path=args.external_path,
            external_out=args.external_out,
            step_size=args.step_size,
            batch_size=args.batch_size,
            add_special_tokens=args.add_special_tokens,
            file_agg=args.file_agg,
            device=args.device,
            num_cpu_threads=args.num_cpu_threads,
        )
        result = predict_fasta_files(infer_cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, ensure_ascii=False, default=str))
        return result

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
