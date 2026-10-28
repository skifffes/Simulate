#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
SNAP=results/experiment_2_6/B90_gripper_extension/states/ep066_f039.json
OUT=results/experiment_2_6/ep66_f39_frozen_14mm_x10
for i in 8 9; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) venv/bin/python -u scripts/run_b90_snapshot.py \
    --snap "$SNAP" --condition frozen --gripper-width 0.014 --policy "$P" \
    --out "$OUT/run$(printf %02d $i)" > "logs/e26_ep66_g14_$i.log" 2>&1
done
echo done
