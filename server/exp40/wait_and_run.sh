#!/usr/bin/env bash
# 等 checkpoint 下载完成，然后自动跑完整验证序列（Level 1–5）。
set -u
WS=/scratch/lijh/exp40
LOG="$WS/wait_and_run.log"

{
echo "=== $(date -Is) 等待下载完成 ==="
for i in $(seq 1 180); do
  if ! pgrep -f "download_ckpt6" > /dev/null 2>&1; then
    echo "下载脚本已结束（第 $i 次检查）"
    break
  fi
  sleep 20
done
tail -4 "$WS/download_ckpt6.log"
du -sh "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid"

echo
echo "=== $(date -Is) 开始完整验证 ==="
bash "$WS/run_all_checkpoint.sh" 3
echo "=== $(date -Is) 完成 ==="
} > "$LOG" 2>&1
echo "done"
tail -40 "$LOG"
