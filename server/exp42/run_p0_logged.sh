#!/usr/bin/env bash
# P0：带逐子步日志与拒绝检查的 B90/A90 rollout（同 checkpoint / 场景 / seed / 评价器）
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
SIM=/scratch/lijh/simulate_vla/venv/bin/python
CKPT=/scratch/lijh/exp42/run2/ckpts/params_nested
PORT=8151
OUT=results/experiment_4_2/rollouts_p0_logged
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

CUDA_VISIBLE_DEVICES=1 setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --mode checkpoint --config exp42_pi05_droid_lora --checkpoint "$CKPT" --port "$PORT" \
  > /scratch/lijh/exp42/serve_p0.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 60); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died'; tail -15 /scratch/lijh/exp42/serve_p0.log; exit 1; }
  grep -qi listening /scratch/lijh/exp42/serve_p0.log && break
done
echo "server up"

echo "=== B90 (2 ep, substep logging) ==="
"$SIM" scripts/exp42/eval_rollouts.py --skill B90 --episodes 2 --ticks 45 --port "$PORT" \
  --tag p0_logged --out "$OUT" 2>&1 | tail -6
echo "=== A90 (1 ep) ==="
"$SIM" scripts/exp42/eval_rollouts.py --skill A90 --episodes 1 --ticks 45 --port "$PORT" \
  --tag p0_logged --out "$OUT" 2>&1 | tail -6

kill "$PID" 2>/dev/null || true
echo done
ls -1 "$OUT" 2>/dev/null | head
