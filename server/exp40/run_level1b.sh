#!/usr/bin/env bash
# Level 1 复跑（修正显存统计口径：只统计本进程；并关闭 XLA 预分配以测量真实占用）
set -u
WS=/scratch/lijh/exp40
REPO=$WS/repo
VENV=$WS/openpi/.venv
GPU="${1:-3}"
LOG="$WS/level1_rerun.log"

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$WS/lerobot_stub:$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
export OPENPI_DATA_HOME="$HOME/.cache/openpi"
export OPENPI_COMMIT=$(git -C "$WS/openpi" rev-parse HEAD 2>/dev/null || echo unknown)

{
  echo "=== $(date -Is) Level 1 复跑（PREALLOCATE=false, GPU$GPU） ==="
  "$VENV/bin/python" scripts/exp40/level1_model_inference.py --mode checkpoint \
    --out results/experiment_4_0/model_smoke 2>&1 | tail -25
  echo "=== $(date -Is) 结束 ==="
} > "$LOG" 2>&1
