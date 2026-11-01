#!/usr/bin/env bash
# 检查关键 wheel 的 manylinux 标签，判断 glibc 2.17 (CentOS7) 是否可装
set -u
echo "=== conda 安装位置 ==="
ls -d /scratch/lijh/*conda* /scratch/lijh/*/bin/conda /opt/*conda* 2>/dev/null
for p in /scratch/lijh/miniconda3/bin/conda /scratch/lijh/anaconda3/bin/conda /scratch/lijh/.conda/bin/conda; do
  [ -x "$p" ] && echo "found: $p" && "$p" --version
done
ls /scratch/lijh/.conda/ 2>/dev/null | head

echo
echo "=== jaxlib 0.5.3 wheel 标签 (cp311) ==="
curl -s -m 40 https://pypi.org/pypi/jaxlib/0.5.3/json | tr ',' '\n' | grep -o '"[^"]*cp311[^"]*manylinux[^"]*\.whl"' | sort -u

echo
echo "=== jax 0.5.3 是否需要 jaxlib 单独安装 ==="
curl -s -m 40 https://pypi.org/pypi/jax/0.5.3/json | tr ',' '\n' | grep -o '"[^"]*jax-0.5.3[^"]*"' | head -3

echo
echo "=== torch 2.7.1 wheel 标签 (cp311) ==="
curl -s -m 40 https://pypi.org/pypi/torch/2.7.1/json | tr ',' '\n' | grep -o '"[^"]*cp311[^"]*linux_x86_64\.whl"' | sort -u | head -8

echo
echo "=== torch 2.6.0 wheel 标签对照 ==="
curl -s -m 40 https://pypi.org/pypi/torch/2.6.0/json | tr ',' '\n' | grep -o '"[^"]*cp311[^"]*linux_x86_64\.whl"' | sort -u | head -8

echo
echo "=== flax 0.10.2 / orbax 0.11.13 py3-none 或 cp311 ==="
curl -s -m 40 https://pypi.org/pypi/orbax-checkpoint/0.11.13/json | tr ',' '\n' | grep -o '"[^"]*\.whl"' | sort -u | head -5
