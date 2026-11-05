#!/usr/bin/env bash
# 启动 A/B 两组训练（GPU1=Group A, GPU0=Group B），日志与 PID 落盘
set -u
mkdir -p /scratch/lijh/exp44
pkill -f 'exp44/A' 2>/dev/null || true
pkill -f 'exp44/B' 2>/dev/null || true

GPUS=1 setsid nohup bash /scratch/lijh/exp44/train_group_a.sh \
  > /scratch/lijh/exp44/logs_A.txt 2>&1 < /dev/null &
echo "A pid $!"
sleep 3
GPUS=0 setsid nohup bash /scratch/lijh/exp44/train_group_b.sh \
  > /scratch/lijh/exp44/logs_B.txt 2>&1 < /dev/null &
echo "B pid $!"
sleep 60
date +%H:%M:%S
for g in A B; do
  echo "--- $g ---"
  tail -3 /scratch/lijh/exp44/logs_$g.txt
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
