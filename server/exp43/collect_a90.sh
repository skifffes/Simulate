#!/usr/bin/env bash
# 实验 4.3 数据扩充：A90 训练目标 6 个 × 新种子；严格成功才入库
set -u
REPO=/scratch/lijh/exp40/repo
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
GOALS="A_x-3_y-3,A_x00_y-3,A_x+3_y-3,A_x-3_y00,A_x00_y00,A_x+3_y00"
exec /scratch/lijh/simulate_vla/venv/bin/python scripts/exp41/collect41.py \
  --mode collect --out /scratch/lijh/exp43/a90 --data /scratch/lijh/exp43/data \
  --only-goals "$GOALS" --collect-seeds 14 --seed-offset 40 \
  --skills A90 --target-a 70 2>&1 | tail -40
