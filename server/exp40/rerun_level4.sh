#!/usr/bin/env bash
# 复跑 Level 4（trace 里补上 chunk 推理往返时间），需要 checkpoint server 在 :8000
set -u
WS=/scratch/lijh/exp40
REPO=$WS/repo
SIM=/scratch/lijh/simulate_vla/venv/bin/python
LOG="$WS/level4_rerun.log"

{
  echo "=== $(date -Is) 起策略服务器 ==="
  bash "$WS/run_server.sh" checkpoint 2 8000 2>&1 | tail -4

  cd "$REPO" || exit 1
  export PYTHONPATH="$REPO/src:$REPO"
  export PYTHONIOENCODING=utf-8
  for sc in B90 A90; do
    echo "=========== Level 4 $sc ==========="
    "$SIM" scripts/exp40/level4_closed_loop.py --scene "$sc" --ticks 20 --port 8000 2>&1 | tail -12
  done
  echo "=== $(date -Is) 结束 ==="
} > "$LOG" 2>&1
echo done
tail -40 "$LOG"
