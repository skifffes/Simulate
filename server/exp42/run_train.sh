#!/usr/bin/env bash
# 挑一张空闲 A40 跑实验 4.2 训练（probe 或 train）
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
MODE=${1:-probe}
shift || true
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

echo "=== GPU state ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' '$2 < 2000 {print $1; exit}')
FREE=${FREE:-4}
echo "using GPU $FREE"
export CUDA_VISIBLE_DEVICES=$FREE

echo "=== git HEAD ==="
git rev-parse HEAD

echo "=== run: $MODE $* ==="
"$OAPI/.venv/bin/python" scripts/exp42/train_lora.py --mode "$MODE" "$@" 2>&1
