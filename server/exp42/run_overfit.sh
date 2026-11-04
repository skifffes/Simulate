#!/usr/bin/env bash
# P1 小样本过拟合测试（8 个固定训练窗口，anchor-16）
set -eu
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
cd "$REPO"
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export CUDA_VISIBLE_DEVICES=${GPU:-0}
exec "$OAPI/.venv/bin/python" scripts/exp42/overfit_probe.py \
  --steps "${STEPS:-150}" --lr-scale "${LRSCALE:-4.0}" --n-windows "${NWIN:-8}" \
  --out "${OUTDIR:-/scratch/lijh/exp42/overfit/lr4}" 2>&1 | tail -20
