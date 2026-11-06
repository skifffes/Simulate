#!/usr/bin/env bash
# 实验 4.5 配对闭环：对指定模型（ae | lora）在 16 条冻结 val episode 上跑 3 个推理种子
# 用法：bash run_paired.sh <ae|lora> <STEP> [GPU]
set -u
MODEL=${1:?ae|lora}
STEP=${2:?step}
GPU=${3:-3}
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export CUDA_VISIBLE_DEVICES=$GPU

A90_EPS=6,7,15,16,24,25,33,34,42,43
B90_EPS=3,10,16,30,36,42
BASE=/scratch/lijh/exp45/paired
OUT=results/experiment_4_5/rollouts/${MODEL}_step${STEP}
LOG=/scratch/lijh/exp45/paired_${MODEL}_${STEP}.log
PORT=$((8500 + RANDOM % 90))
mkdir -p "$BASE" "$OUT"
: > "$LOG"

if [ "$MODEL" = "ae" ]; then
  PARAMS=/scratch/lijh/exp45/run/ckpts/lora_step$(printf '%05d' "$STEP").npz
  MERGED=$BASE/ae_${STEP}
  CFG=exp45_pi05_ae
  SERVE=scripts/exp45/serve_ae.py
  EXPORT=scripts/exp45/export_ae_params.py
else
  PARAMS=/scratch/lijh/exp44/B/ckpts/lora_step$(printf '%05d' "$STEP").npz
  MERGED=$BASE/lora_${STEP}
  CFG=exp42_pi05_droid_lora
  SERVE=scripts/exp42/serve_finetuned.py
  EXPORT=scripts/exp42/export_params_from_lora.py
fi
[ -f "$PARAMS" ] || { echo "missing params $PARAMS" | tee -a "$LOG"; exit 2; }

echo "=== merge $MODEL step $STEP ===" | tee -a "$LOG"
rm -rf "$MERGED"
if [ "$MODEL" = "ae" ]; then
  "$OAPI/.venv/bin/python" "$EXPORT" --params "$PARAMS" --out "$MERGED/params" >> "$LOG" 2>&1
else
  "$OAPI/.venv/bin/python" "$EXPORT" --lora "$PARAMS" --out "$MERGED/params" >> "$LOG" 2>&1
fi
ln -sfn /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets "$MERGED/assets"

echo "=== serve $MODEL on GPU $GPU port $PORT ===" | tee -a "$LOG"
setsid nohup "$OAPI/.venv/bin/python" "$SERVE" --config "$CFG" \
  --checkpoint "$MERGED" --port "$PORT" >> "$LOG" 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 80); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died' | tee -a "$LOG"; tail -6 "$LOG"; exit 1; }
  grep -qi listening "$LOG" && break
done
echo "server up" | tee -a "$LOG"

for OFF in 0 1 2; do
  echo "=== seed-offset $OFF ===" | tee -a "$LOG"
  "$SIM" scripts/exp42/eval_from_dataset_init.py --skill A90 --episodes "$A90_EPS" \
    --ticks 45 --execute-k 8 --port "$PORT" --seed-offset "$OFF" \
    --out "$OUT/off${OFF}" 2>&1 | grep -E 'success=' | tee -a "$LOG"
  "$SIM" scripts/exp42/eval_from_dataset_init.py --skill B90 --episodes "$B90_EPS" \
    --ticks 45 --execute-k 8 --port "$PORT" --seed-offset "$OFF" \
    --out "$OUT/off${OFF}" 2>&1 | grep -E 'success=' | tee -a "$LOG"
done

kill "$PID" 2>/dev/null || true
sleep 2
rm -rf "$MERGED"
echo "=== done $MODEL $STEP ===" | tee -a "$LOG"
