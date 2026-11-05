#!/usr/bin/env bash
# 单个 (group, step, skill) 的严格闭环：合并 → 起服务 → 跑指定 episode → 清理
# 用法：bash run_rollout_skill.sh <GROUP> <STEP> <SKILL> <EPS> [GPU]
set -u
GROUP=${1:?group}
STEP=${2:?step}
SKILL=${3:?A90|B90}
EPS=${4:?episodes}
GPU=${5:-2}
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
LORA=/scratch/lijh/exp44/$GROUP/ckpts/lora_step$(printf '%05d' "$STEP").npz
MERGED=/scratch/lijh/exp44/merged/${GROUP}_${STEP}_${SKILL}
PORT=$((8300 + RANDOM % 120))
OUT=results/experiment_4_4/rollouts/${GROUP}_step${STEP}_${SKILL}
LOG=/scratch/lijh/exp44/rollout_${GROUP}_${STEP}_${SKILL}.log
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
: > "$LOG"
[ -f "$LORA" ] || { echo "missing lora $LORA" | tee -a "$LOG"; exit 2; }
echo "=== merge $GROUP $STEP ===" | tee -a "$LOG"
rm -rf "$MERGED"
CUDA_VISIBLE_DEVICES=$GPU "$OAPI/.venv/bin/python" scripts/exp42/export_params_from_lora.py \
  --lora "$LORA" --out "$MERGED/params" 2>&1 | tail -2 | tee -a "$LOG"
ln -sfn /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets "$MERGED/assets"

echo "=== serve GPU $GPU port $PORT ===" | tee -a "$LOG"
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

"$SIM" scripts/exp42/eval_from_dataset_init.py --skill "$SKILL" --episodes "$EPS" \
  --ticks 45 --execute-k 8 --port "$PORT" --out "$OUT" 2>&1 | grep -E 'success=|skill' | tee -a "$LOG"

kill "$PID" 2>/dev/null || true
sleep 2
rm -rf "$MERGED"
echo "=== done $GROUP $STEP $SKILL ===" | tee -a "$LOG"
