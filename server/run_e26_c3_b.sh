#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_2_6/C3_direct_B90
S=results/experiment_2_6/C3_direct_B90/states/G1/C3_snapshot

run_pair() {
  gpu=$1; cond=$2; a=$3; b=$4
  for s in "$a" "$b"; do
    CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 PYTHONPATH=src \
      venv/bin/python -u scripts/run_b90_snapshot.py \
      --snap "$S/seed_${s}/c3_terminal.json" --condition "$cond" \
      --policy "$P" --out "$O" >> "logs/e26_c3_B_${cond}_${s}.log" 2>&1
  done
}

run_pair 0 frozen 40000000 40000001 &
run_pair 1 frozen 40000002 40000003 &
run_pair 3 frozen 40000004 40000005 &
run_pair 4 frozen 40000006 40000007 &
run_pair 0 frozen 40000008 40000009 &
run_pair 0 expert 40000000 40000001 &
run_pair 1 expert 40000002 40000003 &
run_pair 3 expert 40000004 40000005 &
run_pair 4 expert 40000006 40000007 &
run_pair 0 expert 40000008 40000009 &
echo launched
wait
echo done
