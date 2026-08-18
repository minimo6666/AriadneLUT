#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd "$repo_root"

/home/minimo/miniconda3/envs/DavinciLUT/bin/python -u \
  "$script_dir/train.py" \
  --config "$script_dir/config.yaml" \
  --output-dir "$script_dir/runs/sanity_gpu0" \
  --epochs 1 \
  --global-batch-size 2 \
  --batch-size-per-gpu 2 \
  --num-workers-per-gpu 2 \
  --max-train-batches 2 \
  --max-val-batches 2 \
  "$@"
