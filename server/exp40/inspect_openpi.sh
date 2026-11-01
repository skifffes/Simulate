#!/usr/bin/env bash
# 读取 openpi 的版本要求与 checkpoint 清单
set -u
cd /scratch/lijh/exp40/openpi || exit 1

echo "=== commit ==="
git rev-parse HEAD
git log -1 --format='%cI %s'

echo
echo "=== pyproject 头部 ==="
sed -n '1,45p' pyproject.toml

echo
echo "=== 关键源码清单 ==="
ls src/openpi/
echo "--- policies ---"
ls src/openpi/policies/
echo "--- training ---"
ls src/openpi/training/ | head -20

echo
echo "=== GCS: checkpoints/ 前缀（顶层目录） ==="
curl -s -m 60 "https://storage.googleapis.com/openpi-assets?prefix=checkpoints/&delimiter=/&max-keys=1000" | tr '<' '\n' | grep "^Prefix>" | head -40

echo
echo "=== GCS: pi05 相关前缀 ==="
curl -s -m 60 "https://storage.googleapis.com/openpi-assets?prefix=checkpoints/pi05&delimiter=/&max-keys=1000" | tr '<' '\n' | grep -E "^(Prefix|Key)>" | head -40
