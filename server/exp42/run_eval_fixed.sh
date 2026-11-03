#!/usr/bin/env bash
# 修复执行契约后重跑闭环：固定锚点 + 连续夹爪（米），其他一切不变
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
SIM=/scratch/lijh/simulate_vla/venv/bin/python
CKPT=${1:-/scratch/lijh/exp42/run2/ckpts/params_nested}
PORT=${2:-8141}
GPU=${3:-1}
NEPS=${4:-2}
TICKS=${5:-45}
OUT=results/experiment_4_2/rollouts_fixed_contract
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --mode checkpoint --config exp42_pi05_droid_lora --checkpoint "$CKPT" --port "$PORT" \
  > /scratch/lijh/exp42/serve_fixed.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 60); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died'; tail -20 /scratch/lijh/exp42/serve_fixed.log; exit 1; }
  grep -qi listening /scratch/lijh/exp42/serve_fixed.log && break
done
echo "server up"

for S in A90 B90; do
  "$SIM" scripts/exp42/eval_rollouts.py --skill $S --episodes "$NEPS" --ticks "$TICKS" \
    --port "$PORT" --tag fixed_contract --out "$OUT" 2>&1 | tail -6
done

kill "$PID" 2>/dev/null || true
echo done
