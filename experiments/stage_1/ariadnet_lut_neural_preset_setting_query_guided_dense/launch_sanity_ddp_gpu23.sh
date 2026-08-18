#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
run_dir="$script_dir/runs/sanity_ddp_gpu23"

mkdir -p "$run_dir/logs"
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONUNBUFFERED=1

cd "$repo_root"

/home/minimo/miniconda3/envs/DavinciLUT/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  "$script_dir/train.py" \
  --config "$script_dir/config.yaml" \
  --output-dir "experiments/stage_1/ariadnet_lut_neural_preset_setting_query_guided_dense/runs/sanity_ddp_gpu23" \
  --epochs 1 \
  --max-train-batches 2 \
  --max-val-batches 1 \
  "$@" 2>&1 | tee -a "$run_dir/logs/train.log"
