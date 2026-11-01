#!/usr/bin/env bash
# 寻找官方 pi05_droid JAX checkpoint 的可行镜像来源
set -u
echo "=== HF datasets 搜索 openpi / pi05 ==="
for q in openpi pi05 openpi-assets; do
  echo "--- dataset $q ---"
  timeout 50 curl -sS -m 45 "https://hf-mirror.com/api/datasets?search=${q}&limit=30" \
    | tr '}' '\n' | grep -o '"id":"[^"]*"' | head -20
done

echo
echo "=== lerobot/pi05_droid 文件树与大小 ==="
timeout 50 curl -sS -m 45 "https://hf-mirror.com/api/models/lerobot/pi05_droid?blobs=true" \
  | tr '}' '\n' | grep -oE '"(rfilename|size)":("[^"]*"|[0-9]+)' | paste - - | head -30

echo
echo "=== zaper27/pi05_real_droid 是否 openpi JAX 布局 ==="
timeout 50 curl -sS -m 45 "https://hf-mirror.com/api/models/zaper27/pi05_real_droid?blobs=true" \
  | tr '}' '\n' | grep -oE '"(rfilename|size)":("[^"]*"|[0-9]+)' | paste - - | head -25

echo
echo "=== lerobot 版本里的策略实现 ==="
ls /scratch/lijh/simulate_vla/venv/lib/python3.10/site-packages/lerobot/policies/ 2>/dev/null

echo
echo "=== 安装进度 ==="
tail -12 /scratch/lijh/exp40/install_openpi.log 2>/dev/null
du -sh /scratch/lijh/exp40/openpi/.venv 2>/dev/null
