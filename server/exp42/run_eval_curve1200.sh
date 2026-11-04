#!/usr/bin/env bash
# 用 1200-step 模型做一次闭环（2 B90 + 2 A90），检查离线改善是否转化为物理行为
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
SIM=/scratch/lijh/simulate_vla/venv/bin/python
CKPT=/scratch/lijh/exp42/curve1200/ckpt_1200
MANIFEST=/scratch/lijh/exp42/curve1200/ckpts/manifest.json
PORT=8161
OUT=results/experiment_4_2/rollouts_curve1200
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

CUDA_VISIBLE_DEVICES=1 setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --mode checkpoint --config exp42_pi05_droid_lora --checkpoint "$CKPT" --port "$PORT" \
  > /scratch/lijh/exp42/serve_curve1200.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 60); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died'; tail -15 /scratch/lijh/exp42/serve_curve1200.log; exit 1; }
  grep -qi listening /scratch/lijh/exp42/serve_curve1200.log && break
done
echo "server up"

for S in B90 A90; do
  "$SIM" scripts/exp42/eval_rollouts.py --skill $S --episodes 2 --ticks 45 --port "$PORT" \
    --label-mode anchor --ckpt-manifest "$MANIFEST" --tag curve1200 --out "$OUT" 2>&1 | tail -4
done

kill "$PID" 2>/dev/null || true
echo done
ls -1 "$OUT" 2>/dev/null | head
