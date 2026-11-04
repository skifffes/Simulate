#!/usr/bin/env bash
set -eu
cd /scratch/lijh/exp40/repo
export PYTHONPATH=/scratch/lijh/exp40/repo/src:/scratch/lijh/exp40/repo
OUT=results/experiment_4_2/contract_replay_p0c
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp42/ab_repr_compare.py \
  --a "$OUT/expert_state_analysis_A.json" \
  --b "$OUT/expert_state_analysis_B.json" \
  --out "$OUT/ab_repr_comparison.json" 2>&1 | tail -16
