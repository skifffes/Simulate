#!/usr/bin/env bash
# 训练量/学习率曲线（anchor-16，官方 schedule）：600 步，300/600 各存一次并离线评测
set -eu
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO"
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
OUT=results/experiment_4_2/contract_replay_p0c
CURVE=/scratch/lijh/exp42/curve600

mkdir -p "$CURVE"
CUDA_VISIBLE_DEVICES=0 EXP42_OUT="$CURVE" setsid nohup "$OAPI/.venv/bin/python" \
  scripts/exp42/train_lora.py --mode train --steps 600 --batch 4 --seed 0 \
  --label-mode anchor --save-every 300 --export-params \
  > /scratch/lijh/exp42/train_curve600.log 2>&1 < /dev/null &
echo "launched curve600 pid $!"
