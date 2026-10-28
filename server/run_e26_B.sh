#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_2_6/B_gripper_variation
R=results/experiment_2_6/A_midstate_launch/reconstructed/reconstruction_report.json

run() {
  gpu=$1; cond=$2; state=$3; w=$4
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup venv/bin/python -u scripts/run_experiment_2_6.py \
    --condition "$cond" --states "$state" --gripper-width "$w" \
    --report "$R" --policy "$P" --out "$O" \
    > "logs/e26_B_${cond}_${state}_g${w}.log" 2>&1 &
}

# ep66: four widths, two conditions
run 0 expert ep066_f039 0.014
run 0 frozen ep066_f039 0.014
run 1 expert ep066_f039 0.025
run 1 frozen ep066_f039 0.025
run 3 expert ep066_f039 0.035
run 3 frozen ep066_f039 0.035
run 4 expert ep066_f039 0.040
run 4 frozen ep066_f039 0.040
# ep22: two widths
run 0 expert ep022_f039 0.025
run 0 frozen ep022_f039 0.025
run 1 expert ep022_f039 0.035
run 1 frozen ep022_f039 0.035
echo launched
