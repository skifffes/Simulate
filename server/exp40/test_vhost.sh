#!/usr/bin/env bash
# 验证 vhost 域名（openpi-assets.storage.googleapis.com）的稳定性与是否支持并发
set -u
cd /tmp || exit 1
OBJ="checkpoints/pi05_droid/params/ocdbt.process_0/d/3d66a3d641ef2a488f1a3dad19dcc9fa"
U="https://openpi-assets.storage.googleapis.com/$OBJ"

echo "=== 单连接 16MB x2 ==="
for i in 1 2; do
  timeout 60 curl -4 -sS -m 55 -r 0-16777215 -o /tmp/v_$i.bin \
    -w "single$i http=%{http_code} size=%{size_download} speed=%{speed_download} t=%{time_total}\n" "$U" || echo "single$i FAIL"
done

echo "=== 4 并发 各 8MB ==="
t0=$(date +%s)
for i in 0 1 2 3; do
  s=$((i * 8388608)); e=$((s + 8388607))
  ( timeout 120 curl -4 -sS -m 110 -r ${s}-${e} -o /tmp/vp_$i.bin "$U" ) &
done
wait
t1=$(date +%s)
total=$(du -cb /tmp/vp_*.bin 2>/dev/null | tail -1 | cut -f1)
dt=$((t1 - t0)); [ "$dt" -lt 1 ] && dt=1
echo "并发合计 $total B / ${dt}s = $((total / dt)) B/s"
