#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
S=results/experiment_3_controlled/initial_states/B_place.json
O=results/experiment_3_controlled/rollout_v4
mkdir -p "$O"
for s in 0 1 2 3 4; do
 for g in B C; do
  CUDA_VISIBLE_DEVICES=0 venv/bin/python -u scripts/run_b90_controlled_v3.py --group "$g" --snap "$S" --policy "$P" --out "$O" --seed "$s" > "logs/e3cv4_${g}_r${s}.log" 2>&1 &
 done
 wait
done
echo DONE
