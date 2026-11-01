#!/usr/bin/env bash
# 1) 并发吞吐测试：国际链路是否按连接限速
# 2) ModelScope 可达性 + 搜索 pi05 / openpi 权重
set -u
cd /tmp || exit 1
URL="https://hf-mirror.com/lerobot/smolvla_base/resolve/main/model.safetensors"

echo "=== 4 并发 range 各 8MB（合计 32MB） ==="
t0=$(date +%s)
for i in 0 1 2 3; do
  s=$((i * 8388608)); e=$((s + 8388607))
  curl -sS -L -m 300 -r ${s}-${e} -o /tmp/par_$i.bin "$URL" &
done
wait
t1=$(date +%s)
total=$(du -cb /tmp/par_*.bin 2>/dev/null | tail -1 | cut -f1)
dt=$((t1 - t0))
echo "合计 $total bytes / ${dt}s = $((total / (dt > 0 ? dt : 1))) B/s"
ls -l /tmp/par_*.bin

echo
echo "=== ModelScope 可达性 ==="
timeout 40 curl -sS -m 35 -o /dev/null -w "http=%{http_code} time=%{time_total}\n" "https://www.modelscope.cn/" || echo "modelscope unreachable"

echo "=== ModelScope 搜索 pi05 / openpi ==="
for q in pi05 openpi; do
  echo "--- $q ---"
  timeout 40 curl -sS -m 35 -X POST "https://www.modelscope.cn/api/v1/dolphin/models" \
    -H "Content-Type: application/json" \
    -d "{\"PageSize\":20,\"PageNumber\":1,\"Name\":\"${q}\",\"SortBy\":\"Default\"}" 2>/dev/null \
    | tr ',' '\n' | grep -o '"Name":"[^"]*"' | head -20
done

echo
echo "=== ModelScope 下载测速（公开小文件） ==="
timeout 90 curl -sS -L -m 85 -o /tmp/ms_speed.bin \
  -w "http=%{http_code} size=%{size_download} speed_Bps=%{speed_download}\n" \
  "https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/resolve/master/config.json" || echo "ms dl fail"
