#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_3_2/rollout
mkdir -p "$O"
targets=(ep068_f000 ep074_f000 ep079_f000 ep080_f000 ep082_f000 ep104_f000 ep064_f000)
i=0;n=0
for t in "${targets[@]}"; do
 for g in A B C; do
  for s in 0 1 2 3 4; do
   CUDA_VISIBLE_DEVICES=$((i%4)) venv/bin/python -u scripts/exp32_episode.py --target "$t" --group "$g" --policy "$P" --out "$O" --seed "$s" > "logs/e32_${t}_${g}_r${s}.log" 2>&1 &
   i=$((i+1));n=$((n+1))
   if [ $n -ge 4 ]; then wait;n=0;fi
  done
 done
done
wait
echo DONE
