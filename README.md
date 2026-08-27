# ProbioGen

ProbioGen is a HyenaDNA-style genome classification framework for probiotic candidate prediction and window-level functional interpretation.

This repository is organized as importable modules instead of a single script:

```text
ProbioGen_GitHub/
├── probiogen/
│   ├── model/        # tokenizer, embedding, Hyena blocks, backbone, heads
│   ├── data/         # FASTA reading, sliding windows, token cache, datasets
│   ├── pretrain/     # optional masked-token pretraining
│   ├── train/        # supervised training, metrics, aggregation, CSV export
│   ├── inference/    # external FASTA prediction
│   ├── utils/        # seed, device, json, CPU-thread helpers
│   ├── config.py     # dataclass configs
│   └── cli.py        # command line interface
├── scripts/          # thin command wrappers
├── configs/          # example parameter files
├── requirements.txt
└── pyproject.toml
```

## Data layout for supervised training

Use one subdirectory per class:

```text
./
├── 0/
│   ├── genome_a.fna
│   └── genome_b.fna
└── 1/
    ├── genome_c.fna
    └── genome_d.fna
```

Numeric folder names are sorted numerically. Non-numeric folder names are sorted alphabetically and mapped to integer labels.

## Install

```bash
cd ProbioGen_GitHub
pip install -e .
```

## Optional pretraining

```bash
python -m probiogen.cli pretrain \
  --input_path ./unlabeled_genomes \
  --out_dir ./probiogen_pretrain \
  --window_size 4096 \
  --step_size 2048 \
  --epochs 10 \
  --batch_size 72 \
  --lr 1e-4
```

The pretraining module uses masked-token prediction and saves:

```text
pretrain_checkpoint.pt
pretrain_history.json
pretrain_tokenids_ws4096_ss2048_X.npy
```

## Supervised training

```bash
python -m probiogen.cli train \
  --base_path ./abla \
  --out_dir ./probiogen_run \
  --window_size 4096 \
  --step_size 2048 \
  --epochs 50 \
  --batch_size 72 \
  --lr 1e-4 \
  --file_agg mean
```

If you want to initialize the backbone from pretraining:

```bash
python -m probiogen.cli train \
  --base_path . \
  --out_dir ./probiogen_run \
  --resume ./pretrain_checkpoint.pt
```

## External prediction

```bash
python -m probiogen.cli predict \
  --external_path ./external_genomes \
  --checkpoint ./probiogen.pt \
  --external_out ./external_out \
  --step_size 2048 \
  --batch_size 32 \
  --file_agg mean
```

For old checkpoints trained by the original script that used `[CLS]` and `[SEP]` while keeping `max_length=window_size`, add:

```bash
--add_special_tokens
```

## Python import examples

```python
from probiogen import ModelConfig, TrainConfig, train_finetune

model_cfg = ModelConfig(window_size=4096, n_layer=16, d_model=128)
train_cfg = TrainConfig(
    base_path=".",
    out_dir="./probiogen_run",
    step_size=2048,
    batch_size=72,
    epochs=50,
    lr=1e-4,
    file_agg="mean",
)

result = train_finetune(model_cfg, train_cfg)
print(result["best_checkpoint"])
```

```python
from probiogen import InferenceConfig, predict_fasta_files

infer_cfg = InferenceConfig(
    checkpoint_path="./probiogen.pt",
    external_path="./external_genomes",
    external_out="./external_out",
    step_size=2048,
    batch_size=32,
    file_agg="mean",
)

result = predict_fasta_files(infer_cfg)
print(result["file_csv"])
```

## Main fixes compared with the previous single-file script

1. Training, pretraining, inference, model and data code are separated.
2. The sliding-window logic is unified across training and prediction.
3. The default tokenizer no longer adds `[CLS]`/`[SEP]`, so a 4096 bp window maps to 4096 tokens.
4. `--add_special_tokens` is kept for old checkpoint compatibility.
5. File-level aggregation is explicit: `mean`, `vote`, or `max`.
6. External output directory is safely auto-created when not provided.
7. Window-level CSV exports raw genomic window sequences for downstream BLASTX/KEGG/GO analysis.
