#!/usr/bin/env bash
# 实验 4.6：从 4 条过拟合 episode 的原始初始状态做闭环（3 个推理种子）
set -u
STEP=${1:?step}
GPU=${2:-3}
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export CUDA_VISIBLE_DEVICES=$GPU
LORA=/scratch/lijh/exp46/run/ckpts/lora_step$(printf '%05d' "$STEP").npz
MERGED=/scratch/lijh/exp46/merged/roll_${STEP}
OUT=results/experiment_4_6/rollouts/step${STEP}
LOG=/scratch/lijh/exp46/rollout_${STEP}.log
PORT=$((8900 + RANDOM % 80))
mkdir -p /scratch/lijh/exp46/merged "$OUT"
: > "$LOG"
[ -f "$LORA" ] || { echo "missing $LORA" | tee -a "$LOG"; exit 2; }
rm -rf "$MERGED"
"$OAPI/.venv/bin/python" scripts/exp42/export_params_from_lora.py --lora "$LORA" \
  --out "$MERGED/params" >> "$LOG" 2>&1
ln -sfn /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets "$MERGED/assets"
setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
  --config exp42_pi05_droid_lora --checkpoint "$MERGED" --port "$PORT" >> "$LOG" 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 80); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died' | tee -a "$LOG"; exit 1; }
  grep -qi listening "$LOG" && break
done
for OFF in 0 1 2; do
  "$SIM" scripts/exp42/eval_from_dataset_init.py --skill A90 --episodes 0,5 \
    --ticks 45 --execute-k 8 --port "$PORT" --seed-offset "$OFF" \
    --out "$OUT/off${OFF}" >> "$LOG" 2>&1
  "$SIM" scripts/exp42/eval_from_dataset_init.py --skill B90 --episodes 0,2 \
    --ticks 45 --execute-k 8 --port "$PORT" --seed-offset "$OFF" \
    --out "$OUT/off${OFF}" >> "$LOG" 2>&1
done
kill "$PID" 2>/dev/null || true
sleep 2
rm -rf "$MERGED"
echo "done $STEP" | tee -a "$LOG"
