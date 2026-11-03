#!/usr/bin/env bash
# Gate 4/5：启动微调 checkpoint 的策略服务器 → 自主闭环 rollout（严格物理评价）→ 关闭
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
CKPT=${1:-/scratch/lijh/exp42/run/ckpts/params_nested}
PORT=${2:-8123}
NEPS=${3:-2}
TICKS=${4:-45}
GPU=${5:-1}
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export CUDA_VISIBLE_DEVICES=$GPU

LOG=/scratch/lijh/exp42/serve_ft.log
setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --mode checkpoint --config exp42_pi05_droid_lora --checkpoint "$CKPT" \
  --port "$PORT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "server pid $PID on GPU $GPU -> $LOG"
for i in $(seq 1 120); do
  sleep 3
  if ! kill -0 $PID 2>/dev/null; then echo 'server died'; tail -25 "$LOG"; exit 1; fi
  if grep -qi 'listening\|serving\|started' "$LOG" 2>/dev/null; then echo "server up after ${i}x3s"; break; fi
  if [ "$i" = 120 ]; then echo 'timeout waiting for server'; tail -25 "$LOG"; kill $PID; exit 1; fi
done
tail -3 "$LOG"

SIM=/scratch/lijh/simulate_vla/venv/bin/python
echo "=== A90 rollouts (n=$NEPS ticks=$TICKS) ==="
"$SIM" scripts/exp42/eval_rollouts.py --skill A90 --episodes "$NEPS" --ticks "$TICKS" \
  --port "$PORT" --tag ft --out results/experiment_4_2/finetuned_rollouts 2>&1 | tail -10
echo "=== B90 rollouts (n=$NEPS ticks=$TICKS) ==="
"$SIM" scripts/exp42/eval_rollouts.py --skill B90 --episodes "$NEPS" --ticks "$TICKS" \
  --port "$PORT" --tag ft --out results/experiment_4_2/finetuned_rollouts 2>&1 | tail -10

kill $PID 2>/dev/null || true
sleep 2
echo "server stopped"
