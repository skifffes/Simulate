#!/usr/bin/env bash
# 实验 4.3 数据扩充：B90 训练配置 4 个 × 新种子；严格成功才入库
set -u
REPO=/scratch/lijh/exp40/repo
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
GEOM="B_wall_y_low|B_obj_y015|B_robot_home,B_wall_y_low|B_obj_y015|B_robot_offset,B_wall_y_mid|B_obj_y021|B_robot_home,B_wall_y_mid|B_obj_y021|B_robot_offset"
exec /scratch/lijh/simulate_vla/venv/bin/python scripts/exp41/collect41.py \
  --mode collect --out /scratch/lijh/exp43/b90 --data /scratch/lijh/exp43/data \
  --only-goals "$GEOM" --collect-seeds 20 --seed-offset 40 \
  --skills B90 --target-b 69 2>&1 | tail -40
