#!/usr/bin/env bash
# curve2400 的严格闭环（与 1800 相同的 2 B90 + 2 A90，配对比较）
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
SIM=/scratch/lijh/simulate_vla/venv/bin/python
STEP=${1:-2400}
CKPT=/scratch/lijh/exp42/curve2400/ckpt_${STEP}
MANIFEST=/scratch/lijh/exp42/curve2400/ckpts/manifest.json
PORT=8181
OUT=results/experiment_4_2/rollouts_curve${STEP}
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

CUDA_VISIBLE_DEVICES=0 setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --mode checkpoint --config exp42_pi05_droid_lora --checkpoint "$CKPT" --port "$PORT" \
  > /scratch/lijh/exp42/serve_curve${STEP}.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 60); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died'; tail -12 /scratch/lijh/exp42/serve_curve${STEP}.log; exit 1; }
  grep -qi listening /scratch/lijh/exp42/serve_curve${STEP}.log && break
done
echo "server up"

for S in B90 A90; do
  "$SIM" scripts/exp42/eval_rollouts.py --skill $S --episodes 2 --ticks 45 --port "$PORT" \
    --label-mode anchor --ckpt-manifest "$MANIFEST" --tag curve${STEP} --out "$OUT" 2>&1 | tail -3
done
kill "$PID" 2>/dev/null || true
echo done
