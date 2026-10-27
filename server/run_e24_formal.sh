#!/usr/bin/env bash
# 实验 2.4 正式批：6 组 × 10 episodes，三视角视频 + 全量日志。
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_2_4
SHA=34a6e3080a88250302a9f0abbc01608b58097324af2c321b00b84c395fbdd58b
C=fcf7665

launch() {
  gpu=$1; group=$2; eps=$3
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup venv/bin/python -u scripts/run_experiment_2_4.py \
    --group "$group" --episodes "$eps" --seed-base 40000000 \
    --policy "$P" --checkpoint-sha "$SHA" --commit "$C" --out "$O" \
    > "logs/e24_${group}.log" 2>&1 &
  echo "launched $group on GPU$gpu"
}

launch 0 G0_a0b_direct 10
launch 1 G1_a90b_direct 10
launch 3 G2_b_home_yaw0 10
launch 4 G3_b_home_yaw90 10
launch 0 G4_a0b_retreat_home 10
launch 1 G5_a0b_local_retreat 10
