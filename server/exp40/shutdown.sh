#!/usr/bin/env bash
# 收尾：停掉实验用策略服务器并确认服务器状态干净
set -u
pkill -f serve_policy.py 2>/dev/null
sleep 3
echo "=== 剩余策略服务进程数 ==="
pgrep -fc serve_policy.py 2>/dev/null || echo 0
echo "=== GPU 状态 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== 本用户残留进程（应只剩 grep） ==="
ps -u "$(whoami)" -o pid,etime,args | grep -E 'level[1-5]_|download_ckpt|serve_policy' | grep -v grep | head -5 || true
echo "=== 显存占用进程 ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || echo none
echo "=== 磁盘 ==="
du -sh "$HOME/.cache/openpi" 2>/dev/null
df -h /scratch | tail -1
