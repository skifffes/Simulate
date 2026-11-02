#!/usr/bin/env bash
# 采集完成后的全套验证：回环读取、多样性/统计/split、旧判据对照
set -u
REPO=/scratch/lijh/exp40/repo
PY=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8

echo "=== verify_dataset ==="
"$PY" scripts/exp41/verify_dataset.py --data data/exp41 --out results/experiment_4_1/dataset 2>&1 | tail -4

echo
echo "=== audit_dataset (diversity / split / stats) ==="
"$PY" scripts/exp41/audit_dataset.py --data data/exp41 --out results/experiment_4_1/dataset 2>&1 | head -40

echo
echo "=== compare_legacy_eval ==="
"$PY" scripts/exp41/compare_legacy_eval.py --records results/experiment_4_1/dataset/records \
  --out results/experiment_4_1/evaluation/legacy_vs_strict.json 2>&1 | tail -12

echo
echo "=== final summary ==="
/scratch/lijh/.conda/envs/gxc/bin/python3.10 /scratch/lijh/exp41/final_summary.py 2>&1 | head -80
