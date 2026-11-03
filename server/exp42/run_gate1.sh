#!/usr/bin/env bash
# Gate 1：expert vs adapter 回放，两种标签锚点（recorded / measured），horizon-scaled clip 已生效
set -u
REPO=/scratch/lijh/exp40/repo
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
OUT=results/experiment_4_2/contract_replay_p0c

for ANCHOR in measured recorded; do
  for S in A90 B90; do
    if [ "$S" = "A90" ]; then EPS=5,6; else EPS=0,1; fi
    echo "=== $S episodes $EPS anchor=$ANCHOR ==="
    "$SIM" scripts/exp42/gate1_expert_replay.py --skill "$S" --episodes "$EPS" \
      --label-anchor "$ANCHOR" --out "$OUT" 2>&1 | tail -4
    mv "$OUT/expert_replay_results_${S}.json" "$OUT/expert_replay_results_${S}_${ANCHOR}.json"
    mv "$OUT/expert_vs_adapter_trace_${S}.csv" "$OUT/expert_vs_adapter_trace_${S}_${ANCHOR}.csv"
  done
done
"$SIM" /scratch/lijh/exp42/show_gate1.py 2>/dev/null || true
ls -1 "$OUT" | head -20
