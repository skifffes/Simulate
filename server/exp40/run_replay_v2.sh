#!/usr/bin/env bash
# 重放两个已正确映射的 episode，并把关键输出打到日志
set -u
REPO=/scratch/lijh/exp40/repo
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8

echo "### B90 episode 241 (== b_training_start_states.csv episode_id 61)"
"$SIM" scripts/exp40/replay_fix_v2.py --task B90 --episode-index 241 2>&1 | tail -14

echo
echo "### A90 episode 120"
"$SIM" scripts/exp40/replay_fix_v2.py --task A90 --episode-index 120 2>&1 | tail -14
