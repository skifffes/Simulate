#!/usr/bin/env bash
# 为 A/B 的导出参数建 nested 布局（params 符号链接 + assets），供 create_trained_policy 使用
set -eu
for G in A B; do
  D=/scratch/lijh/exp42/ab_repr/$G
  mkdir -p "$D/ckpt_nested"
  ln -sfn "$D/ckpts/params_step00300" "$D/ckpt_nested/params"
  ln -sfn /scratch/lijh/exp42/run2/ckpts/params_nested/assets "$D/ckpt_nested/assets"
  ls -l "$D/ckpt_nested"
done
ls /scratch/lijh/exp42/ab_repr/A/ckpt_nested/params/_METADATA
