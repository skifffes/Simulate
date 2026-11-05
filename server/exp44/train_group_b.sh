#!/usr/bin/env bash
# 实验 4.4 Group B（扩充 311 条）正式训练
#   12,000 optimizer updates；batch 4；seed 0；官方 warmup=1000/peak=2.5e-5/decay=30000
#   LoRA 保存点：1200,1800,2400,3600,4800,6000,8000,10000,12000
#   optimizer state：6000,12000（真正续训用）
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export EXP42_DATA=/scratch/lijh/exp43/export_v3
export EXP42_OUT=/scratch/lijh/exp44/B
export CUDA_VISIBLE_DEVICES=${GPUS:-0}
mkdir -p "$EXP42_OUT"
exec "$OAPI/.venv/bin/python" scripts/exp42/train_lora.py \
  --mode train --steps 12000 --batch 4 --seed 0 --label-mode anchor \
  --save-every 100000 \
  --save-at 1200,1800,2400,3600,4800,6000,8000,10000,12000 \
  --save-opt-at 6000,12000 \
  --max-hours 9.5
