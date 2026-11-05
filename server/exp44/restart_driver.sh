#!/usr/bin/env bash
# 重启增量评测驱动（step 外层、组内层），并打印当前日志尾部
set -u
pkill -f eval_driver.sh 2>/dev/null || true
sleep 2
sed -i 's/\r$//' /scratch/lijh/exp44/eval_driver.sh /scratch/lijh/exp44/run_rollout.sh
cd /scratch/lijh/exp40/repo || exit 1
setsid nohup bash /scratch/lijh/exp44/eval_driver.sh > /dev/null 2>&1 < /dev/null &
sleep 3
echo restarted
tail -4 /scratch/lijh/exp44/eval_driver.log
