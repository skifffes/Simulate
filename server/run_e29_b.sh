#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_2_9/ab/runs
STATES="low_00 low_03 high_00 high_03"
mkdir -p "$O"

run_task() {
  kind=$1; sid=$2; w=$3; rep=$4; slot=$5
  tag=$(printf '%s_g%03d_r%d' "$sid" "$(python -c "print(int($w*1000))")" "$rep")
  out="$O/$kind/$tag"
  snap="results/experiment_2_9/ab/states/$sid.json"
  case "$kind" in
    E0) venv/bin/python -u scripts/run_b90_snapshot.py --snap "$snap" --condition expert \
          --gripper-width "$w" --out "$out" --policy "$P" > "logs/e29_b_${kind}_${tag}.log" 2>&1 ;;
    E1) venv/bin/python -u scripts/run_b90_e1.py --snap "$snap" --gripper-width "$w" \
          --out "$out" > "logs/e29_b_${kind}_${tag}.log" 2>&1 ;;
    D2) CUDA_VISIBLE_DEVICES=$slot venv/bin/python -u scripts/run_b90_snapshot.py --snap "$snap" \
          --condition frozen --gripper-width "$w" --out "$out" --policy "$P" \
          > "logs/e29_b_${kind}_${tag}.log" 2>&1 ;;
  esac
}

gpu_slots=(0 1 3 4)
g=0
for sid in $STATES; do
  for w in 0.014 0.035 0.040; do
    for kind in E0 E1; do
      run_task "$kind" "$sid" "$w" 0 0 &
    done
    for rep in 0 1; do
      slot=${gpu_slots[$((g % 4))]}; g=$((g+1))
      run_task D2 "$sid" "$w" "$rep" "$slot" &
    done
    wait
    echo "batch $sid $w done"
  done
done
echo ALLDONE
