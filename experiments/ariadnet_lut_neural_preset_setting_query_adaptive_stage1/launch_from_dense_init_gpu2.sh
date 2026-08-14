#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
ckpt="${DENSE_CKPT:-$repo_root/experiments/ariadnet_lut_neural_preset_setting/runs/main_gpu2_b24/checkpoints/best.pth}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export QA_CONSOLE_LOG_DIR="${QA_CONSOLE_LOG_DIR:-$script_dir/runs/from_dense_init_gpu2}"
bash "$script_dir/launch_gpu2.sh" \
  --output-dir experiments/ariadnet_lut_neural_preset_setting_query_adaptive_stage1/runs/from_dense_init_gpu2 \
  --init-dense-checkpoint "$ckpt" \
  "$@"
