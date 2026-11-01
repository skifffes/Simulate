#!/usr/bin/env bash
# 精确测量当前 GCS 分段下载的实际吞吐
set -u
cd /tmp || exit 1
HOST="https://storage-download.googleapis.com/openpi-assets/checkpoints/pi05_droid/params/ocdbt.process_0/d"
S=213a5648ef56840990e3a9d6fca5eb89

echo "=== 单连接 16MB range（60s 上限） ==="
timeout 70 curl -4 -sS -m 60 -r 0-16777215 -o /tmp/s1.bin \
  -w "http=%{http_code} size=%{size_download} speed=%{speed_download} time=%{time_total}\n" "$HOST/$S" || echo fail
ls -l /tmp/s1.bin 2>/dev/null

echo "=== 当前下载进程与已下载量 ==="
ls /scratch/lijh/exp40/ckpt_parts/params/ocdbt.process_0/d/ 2>/dev/null | wc -l
du -sb /scratch/lijh/exp40/ckpt_parts 2>/dev/null
sleep 60
echo "--- 60s 后 ---"
du -sb /scratch/lijh/exp40/ckpt_parts 2>/dev/null
ps aux | grep '[c]url' | wc -l
