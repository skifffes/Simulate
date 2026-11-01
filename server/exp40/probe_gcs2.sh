#!/usr/bin/env bash
# 探测 GCS 单文件下载行为（不同大小、不同超时、range 请求）
set -u
BASE=https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid
cd /tmp || exit 1

echo "=== A. norm_stats 小文件（之前成功过） ==="
timeout 60 curl -sS -m 55 -o ns.json -w "http=%{http_code} size=%{size_download} speed=%{speed_download}\n" \
  "$BASE/assets/droid/norm_stats.json" || echo "fail A"

echo "=== B. _METADATA 再试（120s） ==="
timeout 130 curl -sS -m 120 -o met.bin -w "http=%{http_code} size=%{size_download} speed=%{speed_download}\n" \
  "$BASE/params/_METADATA" || echo "fail B"
ls -l met.bin 2>/dev/null

echo "=== C. _CHECKPOINT_METADATA（258B） ==="
timeout 130 curl -sS -m 120 -o ckpt_meta.bin -w "http=%{http_code} size=%{size_download} speed=%{speed_download}\n" \
  "$BASE/params/_CHECKPOINT_METADATA" || echo "fail C"

echo "=== D. 大文件 range 前 8MB 测速 ==="
timeout 90 curl -sS -m 85 -r 0-8388607 -o shard_head.bin -w "http=%{http_code} size=%{size_download} speed=%{speed_download}\n" \
  "$BASE/params/ocdbt.process_0/d/0832cad6c37f82d4eedd897dcbb8da9d" || echo "fail D"
ls -l shard_head.bin 2>/dev/null

echo "=== E. DNS / route ==="
getent hosts storage.googleapis.com | head -3
