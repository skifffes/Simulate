#!/usr/bin/env bash
# 导出实验 4.2 训练数组（simulate venv，只读，不写 4.1 数据）
set -u
REPO=/scratch/lijh/exp40/repo
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
mkdir -p /scratch/lijh/exp42
LOG=/scratch/lijh/exp42/export.log
setsid nohup /scratch/lijh/simulate_vla/venv/bin/python scripts/exp42/export_data.py \
  --data data/exp41 --out /scratch/lijh/exp42/data > "$LOG" 2>&1 < /dev/null &
echo "started pid $!"
sleep 25
tail -5 "$LOG"
