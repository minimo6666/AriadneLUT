#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "$script_dir/.." && pwd)"
experiment_root="$(cd "$script_dir/../.." && pwd)"
repo_root="$(cd "$experiment_root/../../../.." && pwd)"
python_bin="${ARIADNE_PYTHON:-$(command -v python)}"
gpu="${ARIADNE_TRAIN_GPU:-4}"

if [[ -z "$python_bin" ]]; then
  echo "Python was not found. Activate the DavinciLUT environment or set ARIADNE_PYTHON." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$repo_root"
exec "$python_bin" -u "$experiment_root/_shared/train/source/train_ddp.py" \
  --output-dir "$experiment_dir/outputs/train" \
  --epochs 32 \
  --batch-size-per-gpu 28 \
  --global-batch-size 168 \
  --num-workers-per-gpu 4 \
  --learning-rate 0.0003 \
  --ablation vivid_tail_loss \
  --lambda-vivid-tail 1.0 \
  --vivid-tail-quantile 0.90 \
  "$@"
