#!/usr/bin/env bash
# 重新生成 15 Hz 可行性 与 B 目标条件可区分性 两个工件（dataset 目录曾被重采清空）
set -u
REPO=/scratch/lijh/exp40/repo
PY=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
echo "=== fps_feasibility ==="
"$PY" scripts/exp41/fps_feasibility.py --seeds 3 --out results/experiment_4_1/dataset 2>&1 | tail -3
echo "=== goal_conditioning ==="
"$PY" scripts/exp41/goal_conditioning_test.py --out results/experiment_4_1/dataset 2>&1 | tail -9
ls results/experiment_4_1/dataset/
