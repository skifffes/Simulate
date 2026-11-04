#!/usr/bin/env bash
set -eu
cd /scratch/lijh/exp40/repo
export PYTHONPATH=/scratch/lijh/exp40/repo/src:/scratch/lijh/exp40/repo
OUT=results/experiment_4_2/contract_replay_p0c
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp42/full_chunk_eval.py \
  --npz "$OUT/expert_state_diagnostic.npz" "$OUT/expert_state_A.npz" "$OUT/expert_state_B.npz" \
  --labels anchor anchor target_increment \
  --out "$OUT/full_chunk_eval.json" 2>&1 | tail -6
