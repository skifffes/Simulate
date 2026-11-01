#!/usr/bin/env bash
# 重跑 Level 4（指尖互碰已加入预期接触白名单）
set -u
REPO=/scratch/lijh/exp40/repo
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
for sc in B90 A90; do
  echo "=========== Level 4 $sc ==========="
  "$SIM" scripts/exp40/level4_closed_loop.py --scene "$sc" --ticks 20 --port 8000 2>&1 | tail -14
done
