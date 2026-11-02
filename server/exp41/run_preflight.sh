#!/usr/bin/env bash
# Data Preflight：对已有 90 条数据做复核（不重新采集）
set -u
REPO=/scratch/lijh/exp40/repo
PY=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8

echo "=== preflight (groups 1,2,3,4,5,6) ==="
"$PY" scripts/exp41/preflight.py --data data/exp41 --out results/experiment_4_1/preflight 2>&1 | tail -8

echo
echo "=== reeval sample (v1.1 重放 6 条) ==="
"$PY" scripts/exp41/reeval_sample.py --n 6 --out results/experiment_4_1/preflight 2>&1 | tail -9

echo
echo "=== files ==="
ls results/experiment_4_1/preflight/
