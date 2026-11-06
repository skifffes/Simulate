#!/usr/bin/env bash
# Gate 4：真实 resume 一致性测试（GPU1，与主训练并行）
#   run1：连续 12 步（每步记录），第 6 步保存 ckpt + opt
#   run2：从第 6 步恢复，继续到 12 步（每步记录）
# 期望：step 7..12 的 loss 与 run1 完全一致
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export CUDA_VISIBLE_DEVICES=1
BASE=/scratch/lijh/exp45/resume_test
rm -rf "$BASE"; mkdir -p "$BASE"

echo "=== run1: 12 steps (log every step), save at 6 ==="
"$OAPI/.venv/bin/python" scripts/exp45/train_ae.py --steps 12 --batch 4 --seed 0 \
  --save-at 6 --save-opt-at 6 --log-every 1 --out "$BASE/run1" 2>&1 | grep -E 'step|resumed' | tail -14

echo "=== run2: resume from step6 -> 12 (log every step) ==="
"$OAPI/.venv/bin/python" scripts/exp45/train_ae.py --steps 12 --batch 4 --seed 0 \
  --resume-params "$BASE/run1/ckpts/lora_step00006.npz" \
  --resume-opt "$BASE/run1/ckpts/opt_step00006.npz" \
  --log-every 1 --out "$BASE/run2" 2>&1 | grep -E 'step|resumed' | tail -14
echo "=== resume test done ==="
