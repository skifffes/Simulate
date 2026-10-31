#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
S=results/experiment_2_6/C3_direct_B90/states/G1/C3_snapshot/seed_40000000/c3_terminal.json
O=results/experiment_2_11/expert_c3
mkdir -p "$O"
for w in 0.014 0.035 0.040; do
  tag="c3_g$(printf '%03d' "$(python -c "print(int($w*1000))")")"
  CUDA_VISIBLE_DEVICES=0 setsid nohup venv/bin/python -u scripts/run_b90_snapshot.py \
    --snap "$S" --condition expert --gripper-width "$w" --policy "$P" \
    --out "$O/$tag" > "logs/e211_${tag}.log" 2>&1 &
done
echo launched
