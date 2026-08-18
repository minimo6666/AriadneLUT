#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
run_dir="$script_dir/runs/main_gpu2_b24"

mkdir -p "$run_dir/logs"

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

cd "$repo_root"

/home/minimo/miniconda3/envs/DavinciLUT/bin/python -u \
  "$script_dir/train.py" \
  --config "$script_dir/config.yaml" \
  "$@" 2>&1 | tee -a "$run_dir/logs/train.log"

