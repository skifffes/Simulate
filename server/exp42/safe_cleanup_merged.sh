#!/usr/bin/env bash
# 删除"合并产物"（可由保留的 LoRA npz 重新导出）；逐项二次校验，绝不使用通配符
set -u
fail=0
del() {
  local dir="$1" src="$2"
  if [ ! -e "$dir" ]; then echo "skip (missing): $dir"; return; fi
  if [ ! -f "$src" ]; then echo "REFUSE: $dir 的 LoRA 源不在: $src"; fail=1; return; fi
  local owner
  owner=$(stat -c '%U' "$dir" 2>/dev/null || echo '?')
  if [ "$owner" != "$(whoami)" ]; then echo "REFUSE: $dir 属主是 $owner"; fail=1; return; fi
  echo "rm -rf $dir  (lora_src ok: $src)"
  rm -rf -- "$dir"
  [ -e "$dir" ] && { echo "FAILED: $dir 仍存在"; fail=1; } || echo "  gone"
}

del /scratch/lijh/exp42/ab_repr/A/ckpt_merged           /scratch/lijh/exp42/ab_repr/A/ckpts/lora_step00300.npz
del /scratch/lijh/exp42/ab_repr/B/ckpt_merged           /scratch/lijh/exp42/ab_repr/B/ckpts/lora_step00300.npz
del /scratch/lijh/exp42/ab_repr/A/ckpts/params_step00300 /scratch/lijh/exp42/ab_repr/A/ckpts/lora_step00300.npz
del /scratch/lijh/exp42/ab_repr/B/ckpts/params_step00300 /scratch/lijh/exp42/ab_repr/B/ckpts/lora_step00300.npz
del /scratch/lijh/exp42/curve600/ckpt_300               /scratch/lijh/exp42/curve600/ckpts/lora_step00300.npz
del /scratch/lijh/exp42/curve600/ckpt_600               /scratch/lijh/exp42/curve600/ckpts/lora_step00600.npz
del /scratch/lijh/exp42/curve600/ckpts/params_step00600  /scratch/lijh/exp42/curve600/ckpts/lora_step00600.npz
del /scratch/lijh/exp42/curve1200/ckpt_900              /scratch/lijh/exp42/curve1200/ckpts/lora_step00900.npz

echo
echo "=== 保留项自检 ==="
for d in /scratch/lijh/exp42/curve1200/ckpt_1200/params/_METADATA \
         /scratch/lijh/exp42/curve1200/ckpts/lora_step01200.npz \
         /scratch/lijh/exp42/run2/ckpts/params_nested/params/_METADATA \
         /scratch/lijh/exp42/overfit/lr4/params_step00150/_METADATA \
         /scratch/lijh/exp42/data/meta.jsonl; do
  [ -e "$d" ] && echo "OK   $d" || { echo "MISSING $d"; fail=1; }
done
echo
df -h /scratch | tail -1
exit $fail
