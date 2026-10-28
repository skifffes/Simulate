#!/usr/bin/env bash
# 实验 2.6 A：10 个重建中间状态，专家后缀 vs 冻结 D2 独立启动。
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_2_6/A_midstate_launch
S=(ep066_f039 ep066_f045 ep022_f030 ep022_f039 ep012_f039 ep038_f039 ep074_f039 ep098_f039 ep046_f039 ep080_f039)

# expert: one process, no GPU needed
CUDA_VISIBLE_DEVICES="" setsid nohup venv/bin/python -u scripts/run_experiment_2_6.py \
  --condition expert --states "${S[*]}" --policy "$P" --out "$O" \
  > logs/e26_A_expert.log 2>&1 &

# frozen: 5 processes, 2 states each
for i in 0 1 2 3 4; do
  a=${S[$((i*2))]}; b=${S[$((i*2+1))]}
  gpu=$((i % 4))
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup venv/bin/python -u scripts/run_experiment_2_6.py \
    --condition frozen --states "$a,$b" --policy "$P" --out "$O" \
    > "logs/e26_A_frozen_$i.log" 2>&1 &
done
echo launched
