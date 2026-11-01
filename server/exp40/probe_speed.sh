#!/usr/bin/env bash
# 关键测速：大文件吞吐（GCS range 8MB vs hf-mirror range 8MB）
set -u
cd /tmp || exit 1

echo "=== D. GCS 大分片 range 0-16MB 测速 ==="
timeout 200 curl -sS -m 190 -r 0-16777215 -o gcs_head.bin \
  -w "http=%{http_code} size=%{size_download} speed_Bps=%{speed_download} time=%{time_total}\n" \
  "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/params/ocdbt.process_0/d/0832cad6c37f82d4eedd897dcbb8da9d" || echo "fail D"
ls -l gcs_head.bin 2>/dev/null

echo
echo "=== D2. GCS 同一分片再试（观察稳定性） ==="
timeout 200 curl -sS -m 190 -r 0-16777215 -o gcs_head2.bin \
  -w "http=%{http_code} size=%{size_download} speed_Bps=%{speed_download} time=%{time_total}\n" \
  "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/params/ocdbt.process_0/d/0832cad6c37f82d4eedd897dcbb8da9d" || echo "fail D2"

echo
echo "=== F. hf-mirror 大文件 range 测速（lerobot smolvla_base 权重） ==="
timeout 200 curl -sS -L -m 190 -r 0-16777215 -o hf_head.bin \
  -w "http=%{http_code} size=%{size_download} speed_Bps=%{speed_download} time=%{time_total}\n" \
  "https://hf-mirror.com/lerobot/smolvla_base/resolve/main/model.safetensors" || echo "fail F"
ls -l hf_head.bin 2>/dev/null
