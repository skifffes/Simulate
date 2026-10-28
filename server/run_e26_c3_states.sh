#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_2_6/C3_direct_B90/states
for i in 0 1 2 3 4; do
  a=$((40000000 + i*2)); b=$((a+1))
  CUDA_VISIBLE_DEVICES=$((i % 4)) setsid nohup venv/bin/python -u scripts/run_experiment_2_5.py \
    --condition C3_snapshot --episodes 2 --seed-base "$a" --policy "$P" --out "$O" \
    > "logs/e26_c3_states_$i.log" 2>&1 &
done
echo launched
