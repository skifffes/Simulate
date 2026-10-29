#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
rm -rf results/experiment_2_7/crossed_action_replay
venv/bin/python -u scripts/exp27_cross_replay.py \
  --snapshot results/experiment_2_6/C3_direct_B90/states/G1/C3_snapshot/seed_40000000/c3_terminal.json \
  --p14-actions results/experiment_2_6/C3_seed40000000_frozen_14mm_x10/run00/frozen/seed_40000000_g014/policy_trace.jsonl \
  --p40-actions results/experiment_2_6/C3_direct_B90/frozen/seed_40000000/policy_trace.jsonl \
  --out results/experiment_2_7/crossed_action_replay > logs/e27_cross.log 2>&1
tail -10 logs/e27_cross.log
