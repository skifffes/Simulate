#!/usr/bin/env bash
# 只删除 /scratch/lijh/OneTwoVLA 与 /scratch/lijh/DINO-WM，别的一律不动
set -e
for D in /scratch/lijh/OneTwoVLA /scratch/lijh/DINO-WM; do
  if [ -d "$D" ]; then
    echo "=== $D ==="
    du -sh "$D" 2>/dev/null || true
    if [ -d "$D/.git" ]; then
      git -C "$D" remote -v 2>/dev/null | head -2 || true
      echo "--- uncommitted changes (top 5) ---"
      git -C "$D" status --short 2>/dev/null | head -5 || true
    else
      echo "not a git repo"
    fi
  else
    echo "missing: $D"
  fi
done
