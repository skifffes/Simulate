#!/usr/bin/env bash
# 实验 2.3：等待标准原子评测结束，再跑 D1/D2 的未知合法起点评测。
set -u
cd /scratch/lijh/simulate_vla
VENV=venv/bin/python
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
E=results/experiment_2_3/eval
P1=outputs/smolvla_e23_D1/checkpoints/020000/pretrained_model
P2=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
S=$E/holdout_starts.json

while [ ! -f "$E/atomic_std/D1/eval_C.json" ] || [ ! -f "$E/atomic_std/D2/eval_C.json" ]; do
  sleep 20
done

CUDA_VISIBLE_DEVICES=0 setsid nohup "$VENV" -u scripts/eval_atomic_starts_e23.py \
  --policy "$P1" --starts "$S" --tag D1 --out "$E/atomic_starts/D1" \
  > logs/e23_eval_D1_starts.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 setsid nohup "$VENV" -u scripts/eval_atomic_starts_e23.py \
  --policy "$P2" --starts "$S" --tag D2 --out "$E/atomic_starts/D2" \
  > logs/e23_eval_D2_starts.log 2>&1 &
echo "[eval] starts started"
