#!/usr/bin/env bash
# 启动 π0.5 policy server（官方 websocket 协议）。
# 用法: bash run_server.sh <checkpoint|random> [gpu] [port]
set -u
MODE="${1:-random}"
GPU="${2:-2}"
PORT="${3:-8000}"
REPO=/scratch/lijh/exp40/repo
VENV=/scratch/lijh/exp40/openpi/.venv
WS=/scratch/lijh/exp40
cd "$REPO" || exit 1
mkdir -p "$WS/normstats/droid"
[ -f "$WS/normstats/droid/norm_stats.json" ] || cp "$WS/normstats/droid.json" "$WS/normstats/droid/norm_stats.json"

export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONPATH="$WS/lerobot_stub:$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
export OPENPI_COMMIT=$(git -C "$WS/openpi" rev-parse HEAD 2>/dev/null || echo unknown)
export OPENPI_DATA_HOME="$HOME/.cache/openpi"

LOG="$WS/server_${MODE}_${PORT}.log"
if [ "$MODE" = "checkpoint" ]; then
  setsid nohup "$VENV/bin/python" scripts/exp40/serve_policy.py --mode checkpoint \
    --port "$PORT" --dump-dir "$REPO/results/experiment_4_0/model_smoke/calls_$PORT" \
    > "$LOG" 2>&1 < /dev/null &
else
  setsid nohup "$VENV/bin/python" scripts/exp40/serve_policy.py --mode random \
    --norm-stats "$WS/normstats/droid" --port "$PORT" \
    --dump-dir "$REPO/results/experiment_4_0/model_smoke/calls_$PORT" \
    > "$LOG" 2>&1 < /dev/null &
fi
echo "server pid $! log=$LOG"
for i in $(seq 1 90); do
  sleep 5
  if grep -q "serving on" "$LOG" 2>/dev/null; then echo "UP after $((i*5))s"; break; fi
  if grep -qi "Traceback" "$LOG" 2>/dev/null; then echo "FAILED"; tail -20 "$LOG"; break; fi
done
tail -6 "$LOG"
ss -ltn 2>/dev/null | grep ":$PORT" || echo "no listener on $PORT"
