#!/usr/bin/env bash
# 等待下载或验证任务，最多 sleep N 秒后汇报进度
set -u
WS=/scratch/lijh/exp40
sleep_secs="${1:-300}"
for i in $(seq 1 "$sleep_secs"); do
  sleep 1
done
date
echo "--- download ---"
tail -5 "$WS/download_ckpt6.log"
du -sh "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid" 2>/dev/null
echo "--- wait_and_run ---"
tail -25 "$WS/wait_and_run.log" 2>/dev/null
echo "--- run_all_checkpoint ---"
tail -30 "$WS/run_all_checkpoint.log" 2>/dev/null
