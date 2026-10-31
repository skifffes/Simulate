#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
S=results/experiment_3_controlled/initial_states/B_place.json
for g in B C; do
 venv/bin/python -u scripts/run_b90_controlled_v4.py --group "$g" --snap "$S" --policy "$P" --out results/experiment_3_controlled/act_audit_v4 --seed 0 > "logs/e3cact_$g.log" 2>&1 &
done
wait
echo DONE
