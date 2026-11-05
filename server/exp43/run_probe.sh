#!/usr/bin/env bash
# 在合并导出（314 条 train）上跑 probe：验证数据能被 OpenPI 接口正确读取、可前向/反向
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export EXP42_DATA=/scratch/lijh/exp43/export
export CUDA_VISIBLE_DEVICES=0
exec "$OAPI/.venv/bin/python" scripts/exp42/train_lora.py \
  --mode probe --steps 2 --batch 4 --label-mode anchor \
   2>&1 | tail -12
