#!/usr/bin/env bash
# 在**训练集初始状态**上扩大样本：B90 ep0-9、A90 ep5-9（全部 train split）
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
SIM=/scratch/lijh/simulate_vla/venv/bin/python
CKPT=/scratch/lijh/exp42/curve1800/ckpt_1800
PORT=8192
OUT=results/experiment_4_2/rollouts_dataset_init_more
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

CUDA_VISIBLE_DEVICES=0 setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --mode checkpoint --config exp42_pi05_droid_lora --checkpoint "$CKPT" --port "$PORT" \
  > /scratch/lijh/exp42/serve_dataset_init2.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 60); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died'; tail -12 /scratch/lijh/exp42/serve_dataset_init2.log; exit 1; }
  grep -qi listening /scratch/lijh/exp42/serve_dataset_init2.log && break
done
echo "server up"

echo '=== B90 训练集初始状态 ep0-9 ==='
"$SIM" scripts/exp42/eval_from_dataset_init.py --skill B90 --episodes 0,1,2,3,4,5,6,7,8,9 \
  --ticks 45 --execute-k 8 --port "$PORT" --out "$OUT" 2>&1 | grep -E 'ep[0-9]+\] success|\{"skill"'

echo '=== A90 训练集初始状态 ep5-9 ==='
"$SIM" scripts/exp42/eval_from_dataset_init.py --skill A90 --episodes 5,6,7,8,9 \
  --ticks 45 --execute-k 8 --port "$PORT" --out "$OUT" 2>&1 | grep -E 'ep[0-9]+\] success|\{"skill"'

kill "$PID" 2>/dev/null || true
echo done
