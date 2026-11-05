#!/usr/bin/env bash
# 启动 4 个并行采集进程（A90 x2、B90 x2），不同 seed 区间与独立输出目录
set -u
pkill -f 'collect41.py --mode collect' 2>/dev/null || true
sleep 2
echo "killed old collectors"
for spec in "a90a A90 40 14 70" "a90b A90 54 14 70" "b90a B90 40 20 69" "b90b B90 60 20 69"; do
  set -- $spec
  setsid nohup bash /scratch/lijh/exp43/run_parallel.sh "$1" "$2" "$3" "$4" "$5" \
    > /dev/null 2>&1 < /dev/null &
  echo "launched $1 ($2 offset=$3 seeds=$4 target=$5)"
done
sleep 30
date +%H:%M:%S
for T in a90a a90b b90a b90b; do
  n=$(ls "/scratch/lijh/exp43/$T/records" 2>/dev/null | wc -l)
  echo "$T $n"
done
