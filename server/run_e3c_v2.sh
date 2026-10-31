#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_3_controlled/rollout_v2
mkdir -p "$O"
gpu=0
run(){ g=$1;mode=$2;s=$3
 case $g in A) S=results/experiment_3/snapshots/b_reference/ep064_f000.json;; B) S=results/experiment_3_controlled/initial_states/B_place.json;; C) S=results/experiment_3/snapshots/transition_endpoints/a90_40000000__ep064_f000.json;; esac
 CUDA_VISIBLE_DEVICES=$gpu venv/bin/python -u scripts/run_b90_controlled.py --group $g --mode $mode --snap "$S" --policy "$P" --out "$O" --seed $s > "logs/e3cv2_${g}_${mode}_r${s}.log" 2>&1
 gpu=$(( (gpu+1)%4 ))
}
for s in 0 1 2 3 4; do
 for g in A B C; do run $g snapshot $s & sleep 1; done
 wait
done
for s in 0 1 2 3 4; do
 for g in B C; do run $g continuous $s & sleep 1; done
 wait
done
echo DONE
