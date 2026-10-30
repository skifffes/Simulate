#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
S=results/experiment_2_9/c3_home_width_10/c3_home.json
O=results/experiment_2_9/c3_home_width_10/runs
mkdir -p "$O"
runone() {
  w=$1; rep=$2; gpu=$3
  tag="c3home_g$(printf '%03d' "$(python -c "print(int($w*1000))")")_r$(printf '%02d' "$rep")"
  CUDA_VISIBLE_DEVICES=$gpu venv/bin/python -u scripts/run_b90_snapshot.py \
    --snap "$S" --condition frozen --gripper-width "$w" --policy "$P" \
    --out "$O/$tag" > "logs/e29_c3home_${tag}.log" 2>&1
}
i=0; n=0; gpu=(0 1 2 3)
for w in 0.014 0.040; do
 for rep in $(seq 0 9); do
  runone "$w" "$rep" ${gpu[$((i%4))]} & i=$((i+1)); n=$((n+1))
  if [ $n -ge 8 ]; then wait; n=0; fi
 done
done
wait
echo ALLDONE
