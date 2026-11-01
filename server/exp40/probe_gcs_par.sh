#!/usr/bin/env bash
# GCS 并发 range 吞吐测试：官方 checkpoint 是否可用并行方式下载
set -u
cd /tmp || exit 1
SHARD="https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/params/ocdbt.process_0/d/213a5648ef56840990e3a9d6fca5eb89"

echo "=== 8 并发，每个 range 4MB（合计 32MB） ==="
t0=$(date +%s)
pids=""
for i in 0 1 2 3 4 5 6 7; do
  s=$((i * 4194304)); e=$((s + 4194303))
  ( timeout 240 curl -sS -m 230 -r ${s}-${e} -o /tmp/g_$i.bin "$SHARD" ) &
  pids="$pids $!"
done
wait $pids
t1=$(date +%s)
total=$(du -cb /tmp/g_*.bin 2>/dev/null | tail -1 | cut -f1)
dt=$((t1 - t0))
echo "合计 $total bytes / ${dt}s = $((total / (dt > 0 ? dt : 1))) B/s"
ls -l /tmp/g_*.bin
