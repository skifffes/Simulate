#!/usr/bin/env bash
# 4.4 全部物理闭环：两组各用 A90-best 与 B90-best checkpoint，各跑 2 条
set -u
S=/scratch/lijh/exp44/run_rollout_skill.sh
bash "$S" A 8000  A90 5,6  2
bash "$S" A 4800  B90 0,1  2
bash "$S" B 12000 A90 5,6  2
bash "$S" B 10000 B90 0,1  2
echo ALL_DONE
