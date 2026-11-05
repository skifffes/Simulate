#!/usr/bin/env bash
# 实验 4.4：对选中的 checkpoint 做 PyBullet 严格闭环（4.3 的原始初始状态协议）
# 用法：bash run_rollout.sh <GROUP> <STEP> <TAG> [GPU] [EPISODES_B90] [EPISODES_A90]
set -u
GROUP=${1:?group}
STEP=${2:?step}
TAG=${3:-main}
GPU=${4:-2}
B90_EPS=${5:-0,1}
A90_EPS=${6:-5,6}
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
LORA=/scratch/lijh/exp44/$GROUP/ckpts/lora_step$(printf '%05d' "$STEP").npz
MERGED=/scratch/lijh/exp44/merged/${GROUP}_step${STEP}_roll
PORT=$((8200 + RANDOM % 100))
OUT=results/experiment_4_4/rollouts/${GROUP}_step${STEP}_${TAG}
LOG=/scratch/lijh/exp44/rollout_${GROUP}_${STEP}_${TAG}.log
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

[ -f "$LORA" ] || { echo "missing lora $LORA"; exit 2; }
echo "=== merge $GROUP $STEP ===" | tee -a "$LOG"
rm -rf "$MERGED"
CUDA_VISIBLE_DEVICES=$GPU "$OAPI/.venv/bin/python" scripts/exp42/export_params_from_lora.py \
  --lora "$LORA" --out "$MERGED/params" 2>&1 | tail -2 | tee -a "$LOG"
ln -sfn /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets "$MERGED/assets"

echo "=== serve on GPU $GPU port $PORT ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --mode checkpoint --config exp42_pi05_droid_lora --checkpoint "$MERGED" --port "$PORT" \
  >> "$LOG" 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 80); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died' | tee -a "$LOG"; tail -8 "$LOG"; exit 1; }
  grep -qi listening "$LOG" && break
done
echo "server up" | tee -a "$LOG"

echo "=== B90 rollouts: $B90_EPS ===" | tee -a "$LOG"
"$SIM" scripts/exp42/eval_from_dataset_init.py --skill B90 --episodes "$B90_EPS" \
  --ticks 45 --execute-k 8 --port "$PORT" --out "$OUT" 2>&1 | grep -E 'success=|skill' | tee -a "$LOG"

echo "=== A90 rollouts: $A90_EPS ===" | tee -a "$LOG"
"$SIM" scripts/exp42/eval_from_dataset_init.py --skill A90 --episodes "$A90_EPS" \
  --ticks 45 --execute-k 8 --port "$PORT" --out "$OUT" 2>&1 | grep -E 'success=|skill' | tee -a "$LOG"

kill "$PID" 2>/dev/null || true
sleep 2
rm -rf "$MERGED"
echo "=== done $GROUP $STEP ===" | tee -a "$LOG"
