#!/usr/bin/env bash
# 实验 4.5 补充：**训练集初始状态**上的配对闭环（4.3 协议：B90 ep0-3、A90 ep5,6 × 3 seeds）
# 用途：val 是"留出条件"，两边都 0 成功；训练初态才能区分"是否改善分布内能力"。
# 用法：bash run_paired_traininit.sh <ae|lora> <STEP> [GPU]
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

A90_EPS=5,6
B90_EPS=0,1,2,3
BASE=/scratch/lijh/exp45/paired_traininit
OUT=results/experiment_4_5/rollouts_traininit/${MODEL}_step${STEP}
LOG=/scratch/lijh/exp45/traininit_${MODEL}_${STEP}.log
PORT=$((8700 + RANDOM % 90))
mkdir -p "$BASE" "$OUT"
: > "$LOG"

if [ "$MODEL" = "ae" ]; then
  PARAMS=/scratch/lijh/exp45/run/ckpts/lora_step$(printf '%05d' "$STEP").npz
  MERGED=$BASE/ae_${STEP}; CFG=exp45_pi05_ae
  SERVE=scripts/exp45/serve_ae.py; EXPORT=scripts/exp45/export_ae_params.py
else
  PARAMS=/scratch/lijh/exp44/B/ckpts/lora_step$(printf '%05d' "$STEP").npz
  MERGED=$BASE/lora_${STEP}; CFG=exp42_pi05_droid_lora
  SERVE=scripts/exp42/serve_finetuned.py; EXPORT=scripts/exp42/export_params_from_lora.py
fi
[ -f "$PARAMS" ] || { echo "missing $PARAMS" | tee -a "$LOG"; exit 2; }

rm -rf "$MERGED"
if [ "$MODEL" = "ae" ]; then
  "$OAPI/.venv/bin/python" "$EXPORT" --params "$PARAMS" --out "$MERGED/params" >> "$LOG" 2>&1
else
  "$OAPI/.venv/bin/python" "$EXPORT" --lora "$PARAMS" --out "$MERGED/params" >> "$LOG" 2>&1
fi
ln -sfn /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets "$MERGED/assets"
setsid nohup "$OAPI/.venv/bin/python" "$SERVE" --config "$CFG" \
  --checkpoint "$MERGED" --port "$PORT" >> "$LOG" 2>&1 < /dev/null &
PID=$!
for i in $(seq 1 80); do
  sleep 3
  kill -0 "$PID" 2>/dev/null || { echo 'server died' | tee -a "$LOG"; tail -6 "$LOG"; exit 1; }
  grep -qi listening "$LOG" && break
done
for OFF in 0 1 2; do
  "$SIM" scripts/exp42/eval_from_dataset_init.py --skill A90 --episodes "$A90_EPS" \
    --ticks 45 --execute-k 8 --port "$PORT" --seed-offset "$OFF" \
    --out "$OUT/off${OFF}" >> "$LOG" 2>&1
  "$SIM" scripts/exp42/eval_from_dataset_init.py --skill B90 --episodes "$B90_EPS" \
    --ticks 45 --execute-k 8 --port "$PORT" --seed-offset "$OFF" \
    --out "$OUT/off${OFF}" >> "$LOG" 2>&1
done
kill "$PID" 2>/dev/null || true
sleep 2
rm -rf "$MERGED"
echo "done $MODEL $STEP" | tee -a "$LOG"
