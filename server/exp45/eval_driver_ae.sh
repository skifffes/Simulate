#!/usr/bin/env bash
# 实验 4.5 增量离线评测：等每个 AE checkpoint 出现→合并→查询（4 seeds）→分析→删合并
# 评测 GPU 用 GPU2（训练在 GPU0）
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
STEPS="1200 2400 4000 6000 8000 10000 12000"
LOG=/scratch/lijh/exp45/eval_driver.log
OUT=results/experiment_4_5/offline
: > "$LOG"
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export CUDA_VISIBLE_DEVICES=2
mkdir -p "$OUT"

for S in $STEPS; do
  AN="$REPO/$OUT/ae_step${S}_val_analysis.json"
  [ -f "$AN" ] && { echo "[$(date +%H:%M:%S)] skip $S" >> "$LOG"; continue; }
  for i in $(seq 1 900); do
    F="/scratch/lijh/exp45/run/ckpts/lora_step$(printf '%05d' "$S").npz"
    if [ -f "$F" ]; then
      s1=$(stat -c %s "$F"); sleep 10; s2=$(stat -c %s "$F")
      [ "$s1" = "$s2" ] && [ "$s1" -gt 100000000 ] && break
    fi
    sleep 10
  done
  echo "[$(date +%H:%M:%S)] merge+eval $S" >> "$LOG"
  M=/scratch/lijh/exp45/merged/step${S}
  rm -rf "$M"
  "$OAPI/.venv/bin/python" scripts/exp45/export_ae_params.py \
    --params "$F" --out "$M/params" >> "$LOG" 2>&1
  ln -sfn /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets "$M/assets"
  "$OAPI/.venv/bin/python" scripts/exp45/query_ae.py --ckpt "$M" \
    --picks results/experiment_4_4/val_picks.json --seeds 0,1,2,3 \
    --out "$OUT/ae_step${S}_val.npz" >> "$LOG" 2>&1
  "$SIM" scripts/exp44/analyze_val.py --npz "$OUT/ae_step${S}_val.npz" \
    --out "$OUT/ae_step${S}_val_stats.json" >> "$LOG" 2>&1
  "$SIM" scripts/exp42/full_chunk_eval.py --npz "$OUT/ae_step${S}_val.npz" \
    --out "$OUT/ae_step${S}_chunk.json" >> "$LOG" 2>&1
  rm -rf "$M"
  echo "[$(date +%H:%M:%S)] done $S" >> "$LOG"
done
echo "[$(date +%H:%M:%S)] driver done" >> "$LOG"
