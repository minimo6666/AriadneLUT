#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$script_dir/../../.." && pwd)"
EXP="$script_dir"
BASE="$REPO/experiments/stage_1/ariadnet_lut_neural_preset_setting/config.yaml"

cd "$REPO"
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

/home/minimo/miniconda3/envs/DavinciLUT/bin/python -u \
  "$EXP/train.py" \
  --config "$BASE" \
  --output-dir "$EXP/runs/main_gpu2_b24" \
  --lambda-tonal 0.5 \
  --lambda-chromatic 1.0 \
  --lambda-leakage 0.5 \
  --lambda-slider 0.5 \
  --lambda-perceptual 0.03 \
  --perceptual-size 128 \
  --lambda-lut-smooth 0.02 \
  --lambda-gamut 0.02 \
  --chromatic-scale-multiplier 0.5 \
  "$@"
