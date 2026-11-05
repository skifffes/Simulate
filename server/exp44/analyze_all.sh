#!/usr/bin/env bash
# 对已有 val npz 批量跑分层分析（GPU 不需要，FK 用 CPU）
set -u
REPO=/scratch/lijh/exp40/repo
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
for f in results/experiment_4_4/offline/*_val.npz; do
  case "$f" in
    *SMOKE*) continue ;;
  esac
  out="${f%.npz}_stats.json"
  [ -f "$out" ] && continue
  echo "== $f"
  "$SIM" scripts/exp44/analyze_val.py --npz "$f" --out "$out" 2>&1 | tail -4
done
echo done
