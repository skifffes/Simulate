#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_2_9/b2_low_near/runs
mkdir -p "$O"
TASKS=()
for sid in low_near_00 low_near_01 low_near_02; do
  for w in 0.014 0.035 0.040; do
    for rep in 0 1; do TASKS+=("E1 $sid $w $rep"); TASKS+=("D2 $sid $w $rep"); done
    TASKS+=("E0 $sid $w 0")
  done
done
for sid in low_far_00 mid_near_00; do
  for w in 0.014 0.035 0.040; do
    for rep in 0 1; do TASKS+=("E1 $sid $w $rep"); TASKS+=("D2 $sid $w $rep"); done
  done
done
runone() {
  kind=$1; sid=$2; w=$3; rep=$4; slot=$5
  tag=$(printf '%s_g%03d_r%d' "$sid" "$(python -c "print(int($w*1000))")" "$rep")
  out="$O/$kind/$tag"; snap="results/experiment_2_9/b2_low_near/states/$sid.json"
  case "$kind" in
    E0) venv/bin/python -u scripts/run_b90_snapshot.py --snap "$snap" --condition expert --gripper-width "$w" --out "$out" --policy "$P" > "logs/e29b2_${kind}_${tag}.log" 2>&1 ;;
    E1) venv/bin/python -u scripts/run_b90_e1.py --snap "$snap" --gripper-width "$w" --out "$out" > "logs/e29b2_${kind}_${tag}.log" 2>&1 ;;
    D2) CUDA_VISIBLE_DEVICES=$slot venv/bin/python -u scripts/run_b90_snapshot.py --snap "$snap" --condition frozen --gripper-width "$w" --out "$out" --policy "$P" > "logs/e29b2_${kind}_${tag}.log" 2>&1 ;;
  esac
}
i=0; n=0; gpu=(0 1 3 4)
for t in "${TASKS[@]}"; do
  set -- $t; slot=${gpu[$((i%4))]}; i=$((i+1))
  runone "$1" "$2" "$3" "$4" "$slot" &
  n=$((n+1))
  if [ $n -ge 4 ]; then wait; n=0; fi
done
wait
echo ALLDONE
