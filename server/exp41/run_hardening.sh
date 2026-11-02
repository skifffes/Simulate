#!/usr/bin/env bash
# 全量视频解码 + 回环校验 + 审计（review 后的加固版）
set -u
REPO=/scratch/lijh/exp40/repo
PY=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8

echo "=== verify_dataset (decode ALL videos) ==="
"$PY" scripts/exp41/verify_dataset.py --data data/exp41 --out results/experiment_4_1/dataset 2>&1 | tail -4

echo
echo "=== audit_dataset (keeps frozen split) ==="
"$PY" scripts/exp41/audit_dataset.py --data data/exp41 --out results/experiment_4_1/dataset 2>&1 | grep -E "audit|gripper|split" | head -6

echo
echo "=== reeval sample with v1.2 (two-finger overlap) ==="
"$PY" scripts/exp41/reeval_sample.py --n 6 --out results/experiment_4_1/preflight 2>&1 | tail -8

echo
echo "=== preflight (refreshed) ==="
"$PY" scripts/exp41/preflight.py --data data/exp41 --out results/experiment_4_1/preflight 2>&1 | tail -6
