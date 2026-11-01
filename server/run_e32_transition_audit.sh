#!/usr/bin/env bash
set -u
cd /scratch/lijh/simulate_vla
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
P=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
O=results/experiment_3_2/transition_audit
mkdir -p "$O"
targets=(ep068_f000 ep074_f000 ep079_f000 ep080_f000 ep082_f000 ep104_f000 ep064_f000)
i=0
for t in "${targets[@]}"; do
 CUDA_VISIBLE_DEVICES=$((i%4)) venv/bin/python -u scripts/exp32_episode.py --target "$t" --group C --policy "$P" --out "$O" --seed 0 --state-only > "logs/e32trans_${t}.log" 2>&1 &
 i=$((i+1));if [ $((i%4)) -eq 0 ]; then wait;fi
done
wait
echo DONE
