#!/usr/bin/env bash
# 检查 openpi 在“仅推理”路径上的实际 import 依赖
set -u
cd /scratch/lijh/exp40/openpi || exit 1

echo "=== 哪些文件 import torch / lerobot ==="
grep -rn "^import torch\|^from torch\|^import lerobot\|^from lerobot" src/openpi/ --include=*.py | head -30

echo
echo "=== 哪些文件 import torch（任意位置） ==="
grep -rln "import torch" src/openpi/ --include=*.py | head -30

echo
echo "=== config.py 里 pi05_droid 相关配置 ==="
grep -n "pi05_droid\|pi05_base\|pi05_droid_jointpos" src/openpi/training/config.py | head -40

echo
echo "=== policy_config.py 内容 ==="
cat src/openpi/policies/policy_config.py

echo
echo "=== droid_policy.py ==="
cat src/openpi/policies/droid_policy.py
