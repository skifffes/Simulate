#!/usr/bin/env bash
# 最终验证 Test C：A90/B90 短时真实物理闭环（v3 相机 + 15 Hz + 显式 episode seed）
set -u
REPO=/scratch/lijh/exp40/repo
SIM=/scratch/lijh/simulate_vla/venv/bin/python
OUT=$REPO/results/experiment_4_0/final_validation
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT/A90_closed_loop" "$OUT/B90_closed_loop"
for sc in A90 B90; do
  echo "=========== $sc closed loop ==========="
  "$SIM" scripts/exp40/level4_closed_loop.py --scene "$sc" --ticks 12 --port 8000 \
    --out "$OUT/${sc}_closed_loop" 2>&1 | tail -12
done
