#!/usr/bin/env bash
# 实验 4.4 训练链路冒烟：12 步 + 从 opt_step006 续训到 12（验证真正的 optimizer-state 恢复）
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export EXP42_DATA=/scratch/lijh/exp43/export_v3
export EXP42_OUT=/scratch/lijh/exp44/smoke
export CUDA_VISIBLE_DEVICES=0
rm -rf "$EXP42_OUT"
mkdir -p "$EXP42_OUT"

echo '--- phase 1: 12 steps, save ckpt+opt at 6 ---'
"$OAPI/.venv/bin/python" scripts/exp42/train_lora.py --mode train --steps 12 --batch 4 \
  --save-at 6 --save-opt-at 6 --seed 0 --label-mode anchor 2>&1 | tail -12
ls -l "$EXP42_OUT/ckpts" | head -6

echo '--- phase 2: resume from opt_step006 to 12 ---'
"$OAPI/.venv/bin/python" scripts/exp42/train_lora.py --mode train --steps 12 --batch 4 \
  --init-lora "$EXP42_OUT/ckpts/lora_step00006.npz" \
  --resume-opt "$EXP42_OUT/ckpts/opt_step00006.npz" \
  --seed 0 --label-mode anchor 2>&1 | tail -12
echo '--- done ---'
