#!/usr/bin/env bash
# 检查安装进度 + 测试 GCS 下载速度（小文件、可续传）
set -u
WS=/scratch/lijh/exp40
echo "=== install_wrapper.log ==="
tail -20 "$WS/install_wrapper.log" 2>/dev/null

echo
echo "=== install_openpi.log (tail) ==="
tail -30 "$WS/install_openpi.log" 2>/dev/null

echo
echo "=== 进程 ==="
pgrep -af "uv |uv sync|install_openpi" | head -5

echo
echo "=== GCS 小文件下载测速（droid norm_stats） ==="
cd "$WS" || exit 1
mkdir -p normstats
time curl -sS -m 120 -o normstats/droid.json --retry 3 --retry-delay 2 \
  https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json 2>&1
ls -l normstats/ 2>/dev/null
head -c 400 normstats/droid.json 2>/dev/null
echo
