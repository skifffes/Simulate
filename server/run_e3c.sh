#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src
O=results/experiment_3_controlled/rollout
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
mkdir -p "$O"
runone(){ g=$1; s=$2; case $g in A) S=results/experiment_3/snapshots/b_reference/ep064_f000.json;; B) S=results/experiment_3_controlled/initial_states/B_place.json;; C) S=results/experiment_3/snapshots/transition_endpoints/a90_40000000__ep064_f000.json;; esac
 venv/bin/python -u scripts/run_b90_seeded.py --snap "$S" --policy "$P" --out "$O/${g}_r${s}" --seed "$s" > "logs/e3c_${g}_r${s}.log" 2>&1
}
for s in 0 1 2 3 4; do for g in A B C; do runone $g $s & done; wait; done
echo DONE
