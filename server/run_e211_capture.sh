#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
O=results/experiment_2_11/observations
mkdir -p "$O"
EP=results/experiment_2_6/B90_gripper_extension/states/ep066_f039.json
C3=results/experiment_2_6/C3_direct_B90/states/G1/C3_snapshot/seed_40000000/c3_terminal.json
run(){ snap=$1; w=$2; trace=$3; out=$4; venv/bin/python -u scripts/exp211_capture.py --snap "$snap" --width "$w" --trace "$trace" --out "$O/$out" > "logs/e211_cap_$out.log" 2>&1; }
run "$EP" 0.014 results/experiment_2_6/B_gripper_variation/expert/ep066_f039_g014/policy_trace.jsonl ep66_g014.npz &
run "$EP" 0.035 results/experiment_2_6/B_gripper_variation/expert/ep066_f039_g035/policy_trace.jsonl ep66_g035.npz &
run "$C3" 0.014 results/experiment_2_11/expert_c3/c3_g014/expert/seed_40000000_g014/policy_trace.jsonl c3_g014.npz &
run "$C3" 0.035 results/experiment_2_11/expert_c3/c3_g035/expert/seed_40000000_g035/policy_trace.jsonl c3_g035.npz &
run "$C3" 0.040 results/experiment_2_11/expert_c3/c3_g040/expert/seed_40000000_g040/policy_trace.jsonl c3_g040.npz &
wait
echo DONE
