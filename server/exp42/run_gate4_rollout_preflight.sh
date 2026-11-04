#!/usr/bin/env bash
# 用审计方脚本跑 contact timeline（避免命令行转义问题）
set -u
REPO=/scratch/lijh/exp40/repo
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
OUT=results/experiment_4_2/contract_replay_p0c
for E in 00 01; do
  echo "=== B90 ep$E ==="
  "$SIM" scripts/exp42/gate4_preflight.py rollout \
    --csv "results/experiment_4_2/rollouts_p0_logged/B90_ep${E}_substeps.csv" \
    --out "results/experiment_4_2/contract_replay_p0c/gate4_contact_timeline_B90_ep${E}.json" 2>&1 | tail -4
done
