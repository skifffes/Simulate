#!/usr/bin/env bash
# 1200 步模型 + execute_k=1（每步重规划）：检验"整段累积误差"是否是闭环失败主因
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
SIM=/scratch/lijh/simulate_vla/venv/bin/python
CKPT=/scratch/lijh/exp42/curve1200/ckpt_1200
PORT=8162
OUT=results/experiment_4_2/rollouts_curve1200_k1
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

CUDA_VISIBLE_DEVICES=1 setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --mode checkpoint --config exp42_pi05_droid_lora --checkpoint "$CKPT" --port "$PORT" \
  > /scratch/lijh/exp42/serve_curve1200_k1.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 60); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died'; tail -10 /scratch/lijh/exp42/serve_curve1200_k1.log; exit 1; }
  grep -qi listening /scratch/lijh/exp42/serve_curve1200_k1.log && break
done
echo "server up"
for S in B90 A90; do
  "$SIM" scripts/exp42/eval_rollouts.py --skill $S --episodes 2 --ticks 45 --port "$PORT" \
    --execute-k 1 --label-mode anchor --tag curve1200_k1 --out "$OUT" 2>&1 | tail -3
done
kill "$PID" 2>/dev/null || true
echo done
