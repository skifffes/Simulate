#!/usr/bin/env bash
# 验证假设：带 Range 的请求快，不带 Range 的请求慢
set -u
cd /tmp || exit 1
U="https://storage-download.googleapis.com/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json"

echo "=== A plain（无 Range），45s 上限 ==="
timeout 50 curl -4 -sS -m 45 -o /tmp/p_plain.json -w "http=%{http_code} size=%{size_download} t=%{time_total}\n" "$U" || echo "plain FAIL"

echo "=== B 显式 Range: bytes=0- ，45s 上限 ==="
timeout 50 curl -4 -sS -m 45 -r 0- -o /tmp/p_range.json -w "http=%{http_code} size=%{size_download} t=%{time_total}\n" "$U" || echo "range FAIL"

echo "=== C 大文件 Range 16MB 复测，45s ==="
timeout 50 curl -4 -sS -m 45 -r 0-16777215 -o /tmp/p_big.bin \
  -w "http=%{http_code} size=%{size_download} t=%{time_total} speed=%{speed_download}\n" \
  "https://storage-download.googleapis.com/openpi-assets/checkpoints/pi05_droid/params/ocdbt.process_0/d/3d66a3d641ef2a488f1a3dad19dcc9fa" || echo "big FAIL"
