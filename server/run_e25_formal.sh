#!/usr/bin/env bash
# 实验 2.5 正式批：C1/C2/C3/C4 与 R1，各 3 个配对 seed（40000000-40000002）。
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_2_5

launch() {
  gpu=$1; cond=$2
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup venv/bin/python -u scripts/run_experiment_2_5.py \
    --condition "$cond" --episodes 3 --seed-base 40000000 --policy "$P" --out "$O" \
    > "logs/e25_${cond}.log" 2>&1 &
  echo "launched $cond"
}

launch 0 C1
launch 1 C2
launch 3 C3
launch 4 C4
launch 0 R1
