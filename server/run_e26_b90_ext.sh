#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_2_6/B90_gripper_extension
S=results/experiment_2_6/B90_gripper_extension/states
W=(0.014 0.025 0.035 0.040)

run_group() {
  gpu=$1; cond=$2; shift 2
  for st in "$@"; do
    for w in "${W[@]}"; do
      CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 PYTHONPATH=src \
        venv/bin/python -u scripts/run_b90_snapshot.py \
        --snap "$S/${st}.json" --condition "$cond" --gripper-width "$w" \
        --policy "$P" --out "$O" >> "logs/e26_b90_${cond}_${st}.log" 2>&1
    done
  done
}

run_group 0 frozen ep066_f039 ep074_f039 &
run_group 1 frozen ep098_f039 ep080_f039 &
run_group 3 frozen ep088_f039 ep104_f039 &
run_group 4 frozen ep064_f039 ep082_f039 &
run_group 0 expert ep066_f039 ep074_f039 &
run_group 1 expert ep098_f039 ep080_f039 &
run_group 3 expert ep088_f039 ep104_f039 &
run_group 4 expert ep064_f039 ep082_f039 &

echo launched
wait
echo done
