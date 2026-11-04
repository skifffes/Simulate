#!/usr/bin/env bash
# Gate 1 回归（回应 gate4_review 第 2 条）：anchor/fixed 与 target_increment/recursive
# 送入同一 Adapter + 同一物理控制器，验证命令序列与严格判据一致。
set -u
REPO=/scratch/lijh/exp40/repo
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
OUT=results/experiment_4_2/contract_replay_p0c

for MODE in anchor target_increment; do
  for S in A90 B90; do
    if [ "$S" = "A90" ]; then EPS=5,6; else EPS=0,1; fi
    echo "=== $S $MODE ==="
    "$SIM" scripts/exp42/gate1_expert_replay.py --skill "$S" --episodes "$EPS" \
      --label-anchor measured --label-mode "$MODE" --out "$OUT" 2>&1 | tail -3
    mv "$OUT/expert_replay_results_${S}.json" "$OUT/expert_replay_results_${S}_${MODE}.json"
    mv "$OUT/expert_vs_adapter_trace_${S}.csv" "$OUT/expert_vs_adapter_trace_${S}_${MODE}.csv"
  done
done
echo done
