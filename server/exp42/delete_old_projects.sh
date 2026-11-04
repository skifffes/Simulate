#!/usr/bin/env bash
# 删除 /scratch/lijh/OneTwoVLA 与 /scratch/lijh/DINO-WM（用户确认）；不动其它任何目录
set -u
for D in /scratch/lijh/OneTwoVLA /scratch/lijh/DINO-WM; do
  if [ -d "$D" ]; then
    echo "removing $D ($(du -sh "$D" 2>/dev/null | cut -f1))"
    rm -rf --one-file-system "$D"
  else
    echo "already gone: $D"
  fi
done
echo '--- verify ---'
for D in /scratch/lijh/OneTwoVLA /scratch/lijh/DINO-WM; do
  if [ -e "$D" ]; then echo "STILL EXISTS: $D"; else echo "gone: $D"; fi
done
df -h /scratch | tail -1
