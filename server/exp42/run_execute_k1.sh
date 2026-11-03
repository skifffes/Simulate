#!/usr/bin/env bash
# 单变量诊断：execute_k=1（每步重规划）vs execute_k=8（已有结果）
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
SIM=/scratch/lijh/simulate_vla/venv/bin/python
CKPT=/scratch/lijh/exp42/run2/ckpts/params_nested
PORT=8143
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
CUDA_VISIBLE_DEVICES=1 setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --mode checkpoint --config exp42_pi05_droid_lora --checkpoint "$CKPT" --port "$PORT" \
  > /scratch/lijh/exp42/serve_k1.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 60); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died'; tail -15 /scratch/lijh/exp42/serve_k1.log; exit 1; }
  grep -qi listening /scratch/lijh/exp42/serve_k1.log && break
done
echo "server up"
OUT=results/experiment_4_2/rollouts_execute_k1
for S in A90 B90; do
  "$SIM" scripts/exp42/eval_rollouts.py --skill $S --episodes 2 --ticks 45 --port "$PORT" \
    --execute-k 1 --tag k1 --out "$OUT" 2>&1 | tail -4
done
kill "$PID" 2>/dev/null || true
echo done
