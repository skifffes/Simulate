#!/usr/bin/env bash
# 实验 4.1 有界采集（重跑：干净 records，按轮次交错覆盖所有条件）
set -u
REPO=/scratch/lijh/exp40/repo
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
rm -rf data/exp41 results/experiment_4_1/dataset
LOG=/scratch/lijh/exp41/collect2.log
setsid nohup /scratch/lijh/simulate_vla/venv/bin/python scripts/exp41/collect41.py \
  --mode collect --target-a 45 --target-b 45 --collect-seeds 12 --fresh-records \
  --out results/experiment_4_1/dataset --data data/exp41 > "$LOG" 2>&1 < /dev/null &
echo "started pid $!"
sleep 15
tail -2 "$LOG"
