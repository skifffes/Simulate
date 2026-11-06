#!/usr/bin/env bash
# 实验 4.5 正式训练：Action Expert-only 全参数微调，311 条数据，12,000 updates
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export CUDA_VISIBLE_DEVICES=${GPUS:-0}
mkdir -p /scratch/lijh/exp45/run
exec "$OAPI/.venv/bin/python" scripts/exp45/train_ae.py \
  --steps 12000 --batch 4 --seed 0 \
  --save-at 1200,2400,4000,6000,8000,10000,12000 \
  --save-opt-at 6000,12000 \
  --max-hours 5.0 \
  --out /scratch/lijh/exp45/run
