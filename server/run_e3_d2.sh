#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_3/d2_rollouts
mkdir -p "$O"
runone(){ kind=$1; bid=$2; rep=$3; gpu=$4
  if [ "$kind" = "AtoB" ]; then S=results/experiment_3/snapshots/transition_endpoints/a90_40000000__${bid}.json; else S=results/experiment_3/snapshots/b_reference/${bid}.json; fi
  tag=${kind}_${bid}_r${rep}
  CUDA_VISIBLE_DEVICES=$gpu venv/bin/python -u scripts/run_b90_snapshot.py --snap "$S" --condition frozen --policy "$P" --out "$O/$tag" > "logs/e3_${tag}.log" 2>&1
}
i=0;n=0;gpu=(0 1 2 3)
for bid in ep074_f000 ep064_f000 ep070_f000; do
 for kind in AtoB Bref; do
  for rep in 0 1 2; do
   runone $kind $bid $rep ${gpu[$((i%4))]} & i=$((i+1));n=$((n+1))
   if [ $n -ge 8 ]; then wait;n=0;fi
  done
 done
done
wait
echo DONE
