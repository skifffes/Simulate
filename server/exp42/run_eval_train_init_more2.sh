#!/usr/bin/env bash
# B90 训练集初始状态 ep10-19（扩大样本估计成功率）
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
SIM=/scratch/lijh/simulate_vla/venv/bin/python
CKPT=/scratch/lijh/exp42/curve1800/ckpt_1800
PORT=8193
OUT=results/experiment_4_2/rollouts_dataset_init_more2
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

CUDA_VISIBLE_DEVICES=0 setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --mode checkpoint --config exp42_pi05_droid_lora --checkpoint "$CKPT" --port "$PORT" \
  > /scratch/lijh/exp42/serve_dataset_init3.log 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 60); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died'; tail -12 /scratch/lijh/exp42/serve_dataset_init3.log; exit 1; }
  grep -qi listening /scratch/lijh/exp42/serve_dataset_init3.log && break
done
echo "server up"

"$SIM" scripts/exp42/eval_from_dataset_init.py --skill B90 \
  --episodes 10,11,12,13,14,15,16,17,18,19 \
  --ticks 45 --execute-k 8 --port "$PORT" --out "$OUT" 2>&1 | tail -12

kill "$PID" 2>/dev/null || true
echo done
