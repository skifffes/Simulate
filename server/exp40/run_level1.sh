#!/usr/bin/env bash
# 在 A40 上跑 Level 1（模型独立推理）。
# 用法: bash run_level1.sh <checkpoint|random> [gpu]
set -u
MODE="${1:-random}"
GPU="${2:-2}"
REPO=/scratch/lijh/exp40/repo
VENV=/scratch/lijh/exp40/openpi/.venv
WS=/scratch/lijh/exp40
cd "$REPO" || exit 1
mkdir -p "$WS/normstats/droid"
[ -f "$WS/normstats/droid/norm_stats.json" ] || cp "$WS/normstats/droid.json" "$WS/normstats/droid/norm_stats.json"

export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
export OPENPI_COMMIT=$(git -C "$WS/openpi" rev-parse HEAD 2>/dev/null || echo unknown)
export OPENPI_DATA_HOME="$HOME/.cache/openpi"

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n "$((GPU + 1))p"

echo "=== Level 1 ($MODE) ==="
if [ "$MODE" = "checkpoint" ]; then
  "$VENV/bin/python" scripts/exp40/level1_model_inference.py --mode checkpoint \
    --out "$REPO/results/experiment_4_0/model_smoke" 2>&1 | tail -40
else
  "$VENV/bin/python" scripts/exp40/level1_model_inference.py --mode random \
    --norm-stats "$WS/normstats/droid" \
    --out "$REPO/results/experiment_4_0/model_smoke" 2>&1 | tail -40
fi
