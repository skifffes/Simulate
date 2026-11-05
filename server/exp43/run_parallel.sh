#!/usr/bin/env bash
# 实验 4.3：并行采集（每个进程独立 out/data/seed 区间），最后统一汇总
# 用法: run_parallel.sh <tag> <skill> <offset> <seeds> <target>
set -u
REPO=/scratch/lijh/exp40/repo
TAG=$1; SKILL=$2; OFFSET=$3; SEEDS=$4; TARGET=$5
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A_GOALS="A_x-3_y-3,A_x00_y-3,A_x+3_y-3,A_x-3_y00,A_x00_y00,A_x+3_y00"
B_GEOM="B_wall_y_low|B_obj_y015|B_robot_home,B_wall_y_low|B_obj_y015|B_robot_offset,B_wall_y_mid|B_obj_y021|B_robot_home,B_wall_y_mid|B_obj_y021|B_robot_offset"
if [ "$SKILL" = "A90" ]; then
  exec /scratch/lijh/simulate_vla/venv/bin/python scripts/exp41/collect41.py \
    --mode collect --out "/scratch/lijh/exp43/$TAG" --data "/scratch/lijh/exp43/data_$TAG" \
    --only-goals "$A_GOALS" --collect-seeds "$SEEDS" --seed-offset "$OFFSET" \
    --skills A90 --target-a "$TARGET" > "/scratch/lijh/exp43/$TAG.log" 2>&1
else
  exec /scratch/lijh/simulate_vla/venv/bin/python scripts/exp41/collect41.py \
    --mode collect --out "/scratch/lijh/exp43/$TAG" --data "/scratch/lijh/exp43/data_$TAG" \
    --only-goals "$B_GEOM" --collect-seeds "$SEEDS" --seed-offset "$OFFSET" \
    --skills B90 --target-b "$TARGET" > "/scratch/lijh/exp43/$TAG.log" 2>&1
fi
