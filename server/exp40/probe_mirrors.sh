#!/usr/bin/env bash
# GCS 太慢（~86 B/s），寻找国内可达的 checkpoint 镜像并测速
set -u
echo "=== HF 搜索 pi05 / openpi ==="
for q in "pi05" "pi0.5" "openpi" "pi05_droid"; do
  echo "--- query=$q ---"
  timeout 40 curl -sS -m 35 "https://hf-mirror.com/api/models?search=${q}&limit=30" \
    | tr '}' '\n' | grep -o '"modelId":"[^"]*"' | head -30
done

echo
echo "=== HF 搜索 physical-intelligence 组织 ==="
timeout 40 curl -sS -m 35 "https://hf-mirror.com/api/models?author=physical-intelligence&limit=50" \
  | tr '}' '\n' | grep -o '"modelId":"[^"]*"' | head -50

echo
echo "=== hf-mirror 下载测速（任意公开小文件，10MB 量级） ==="
timeout 90 curl -sS -L -m 85 -o /tmp/speedtest.bin \
  -w "http=%{http_code} size=%{size_download} speed=%{speed_download}\n" \
  "https://hf-mirror.com/lerobot/smolvla_base/resolve/main/config.json" 2>&1 | tail -2
ls -l /tmp/speedtest.bin 2>/dev/null

echo
echo "=== GitHub 测速（对照） ==="
timeout 60 curl -sS -L -m 55 -o /dev/null -w "http=%{http_code} speed=%{speed_download}\n" \
  "https://github.com/Physical-Intelligence/openpi/archive/refs/heads/main.tar.gz" 2>&1 | tail -1

echo
echo "=== pypi tuna 测速（对照） ==="
timeout 60 curl -sS -m 55 -o /dev/null -w "http=%{http_code} speed=%{speed_download}\n" \
  "https://pypi.tuna.tsinghua.edu.cn/simple/numpy/" 2>&1 | tail -1
