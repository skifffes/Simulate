#!/usr/bin/env bash
# 导出 A（32D loss）/ B（8D masked loss）的完整 params，并跑离线对比
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
PY=/scratch/lijh/exp40/openpi/.venv/bin/python
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

for TAG in A B; do
  NPZ=$(ls /scratch/lijh/exp42/ab/$TAG/ckpts/lora_step*.npz 2>/dev/null | sort | tail -1)
  echo "=== export $TAG from $NPZ ==="
  [ -z "$NPZ" ] && { echo "no npz for $TAG"; continue; }
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/exp42/export_params_from_lora.py \
    --lora "$NPZ" --out "/scratch/lijh/exp42/ab/$TAG/ckpts/params_nested/params" 2>&1 | tail -3
  ln -sfn /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets \
    "/scratch/lijh/exp42/ab/$TAG/ckpts/params_nested/assets"
  ls "/scratch/lijh/exp42/ab/$TAG/ckpts/params_nested/"
done

echo "=== 离线对比：A(32D) vs B(8D masked) 在初始帧与运动帧上的动作 ==="
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/exp42/p0c_initial_frames.py \
  --ckpt501 /scratch/lijh/exp42/ab/A/ckpts/params_nested \
  --ckpt901 /scratch/lijh/exp42/ab/B/ckpts/params_nested \
  --out /scratch/lijh/exp42/ab/diag 2>&1 | tail -30
