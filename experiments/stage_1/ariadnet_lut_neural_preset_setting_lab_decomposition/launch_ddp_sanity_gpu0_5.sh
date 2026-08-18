#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$script_dir/../../.." && pwd)"
EXP="$script_dir"
PYTHON="/home/minimo/miniconda3/envs/DavinciLUT/bin/python"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "$ROOT"
"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=6 \
  "$EXP/train_ddp.py" \
  --output-dir "$EXP/runs/sanity_ddp_gpu0_5_b174_lr3e4" \
  --epochs 1 \
  --max-train-batches 10 \
  --max-val-batches 2 \
  --batch-size-per-gpu 29 \
  --global-batch-size 174 \
  --num-workers-per-gpu 4 \
  --learning-rate 0.0003 \
  "$@"
