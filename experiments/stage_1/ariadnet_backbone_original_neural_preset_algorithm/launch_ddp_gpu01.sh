#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
run_dir="$script_dir/runs/main_ddp_gpu01_b272_final"

mkdir -p "$run_dir/logs"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

cd "$repo_root"

/home/minimo/miniconda3/envs/DavinciLUT/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  "$script_dir/train.py" \
  --config "$script_dir/config.yaml" \
  "$@" 2>&1 | tee -a "$run_dir/logs/train.log"
