#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
SNAP=results/experiment_2_6/C3_direct_B90/states/G1/C3_snapshot/seed_40000000/c3_terminal.json
OUT=results/experiment_2_6/C3_seed40000000_frozen_14mm_x10
for i in 8 9; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) venv/bin/python -u scripts/run_b90_snapshot.py \
    --snap "$SNAP" --condition frozen --gripper-width 0.014 --policy "$P" \
    --out "$OUT/run$(printf %02d $i)" > "logs/e26_c3_g14_$i.log" 2>&1
done
echo done
