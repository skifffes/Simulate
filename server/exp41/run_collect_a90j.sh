#!/usr/bin/env bash
# 重采 A90（加入目标点确定性抖动 ±8 mm，使 A 终态不再只有 9 种）
set -u
REPO=/scratch/lijh/exp40/repo
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
rm -rf data/exp41/A90
LOG=/scratch/lijh/exp41/collect_a90j.log
setsid nohup /scratch/lijh/simulate_vla/venv/bin/python scripts/exp41/collect41.py \
  --mode collect --skills A90 --target-a 45 --collect-seeds 12 \
  --out results/experiment_4_1/dataset --data data/exp41 > "$LOG" 2>&1 < /dev/null &
echo "started pid $!"
sleep 20
tail -2 "$LOG"
