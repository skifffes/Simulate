#!/usr/bin/env bash
# 把 A/B 的 LoRA npz 合并成 nested 完整 params（官方 {"params": tree} 布局）
set -eu
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
cd "$REPO"
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
for G in A B; do
  echo "=== $G ==="
  "$OAPI/.venv/bin/python" scripts/exp42/export_params_from_lora.py \
    --lora "/scratch/lijh/exp42/ab_repr/$G/ckpts/lora_step00300.npz" \
    --out "/scratch/lijh/exp42/ab_repr/$G/ckpt_merged/params" 2>&1 | tail -3
done
ls /scratch/lijh/exp42/ab_repr/A/ckpt_merged/params | head -3
