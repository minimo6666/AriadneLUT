#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export QA_CONSOLE_LOG_DIR="${QA_CONSOLE_LOG_DIR:-$script_dir/runs/sanity_gpu2}"
bash "$script_dir/launch_gpu2.sh" \
  --output-dir experiments/ariadnet_lut_neural_preset_setting_query_adaptive_stage1/runs/sanity_gpu2 \
  --epochs 1 \
  --max-train-batches 100 \
  --max-val-batches 20 \
  "$@"
