#!/usr/bin/env bash
# 实验 4.6：随 checkpoint 出现逐个做逐帧离线评测（GPU2）
set -u
REPO=/scratch/lijh/exp40/repo
SIM=/scratch/lijh/simulate_vla/venv/bin/python
STEPS="500 1000 1500"
LOG=/scratch/lijh/exp46/eval_driver.log
: > "$LOG"
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:/scratch/lijh/exp40/openpi/src:/scratch/lijh/exp40/openpi/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
for S in $STEPS; do
  AN="$REPO/results/experiment_4_6/offline/step${S}_stats.json"
  [ -f "$AN" ] && { echo "[$(date +%H:%M:%S)] skip $S" >> "$LOG"; continue; }
  for i in $(seq 1 600); do
    F="/scratch/lijh/exp46/run/ckpts/lora_step$(printf '%05d' "$S").npz"
    if [ -f "$F" ]; then
      s1=$(stat -c %s "$F"); sleep 10; s2=$(stat -c %s "$F")
      [ "$s1" = "$s2" ] && [ "$s1" -gt 100000000 ] && break
    fi
    sleep 10
  done
  echo "[$(date +%H:%M:%S)] eval $S" >> "$LOG"
  bash /scratch/lijh/exp40/repo/server/exp46/eval_overfit.sh "$S" >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] done $S" >> "$LOG"
done
echo "[$(date +%H:%M:%S)] driver done" >> "$LOG"
