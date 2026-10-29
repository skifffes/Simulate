#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
M=results/experiment_2_8/data/state_manifest.csv
O=results/experiment_2_8/policy

launch() {
  gpu=$1; shard=$2
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup venv/bin/python -u scripts/exp28_policy_sensitivity.py \
    --manifest "$M" --policy "$P" --out "$O" --seeds 10 --shard "$shard" --nshards 4 \
    > "logs/e28_policy_${shard}.log" 2>&1 &
}
launch 0 0
launch 1 1
launch 3 2
launch 4 3
echo launched
