#!/usr/bin/env bash
# 只列出候选（不删除）：合并出的 params 目录是否可安全删除（对应 LoRA npz 是否仍在）
set -u
check() {
  local dir="$1" src="$2"
  local size="missing" ok_src="NO" is_merge="NO"
  [ -d "$dir" ] && size=$(du -sh "$dir" 2>/dev/null | cut -f1)
  [ -f "$src" ] && ok_src="YES"
  [ -f "$dir/_METADATA" ] && is_merge="YES"
  printf '%-52s size=%-6s merge=%-3s lora_src=%-3s %s\n' "$dir" "$size" "$is_merge" "$ok_src" "$src"
}

echo '=== 可删候选（合并产物，LoRA npz 仍在）==='
check /scratch/lijh/exp42/ab_repr/A/ckpt_merged          /scratch/lijh/exp42/ab_repr/A/ckpts/lora_step00300.npz
check /scratch/lijh/exp42/ab_repr/B/ckpt_merged          /scratch/lijh/exp42/ab_repr/B/ckpts/lora_step00300.npz
check /scratch/lijh/exp42/ab_repr/A/ckpts/params_step00300 /scratch/lijh/exp42/ab_repr/A/ckpts/lora_step00300.npz
check /scratch/lijh/exp42/ab_repr/B/ckpts/params_step00300 /scratch/lijh/exp42/ab_repr/B/ckpts/lora_step00300.npz
check /scratch/lijh/exp42/curve600/ckpt_300              /scratch/lijh/exp42/curve600/ckpts/lora_step00300.npz
check /scratch/lijh/exp42/curve600/ckpt_600              /scratch/lijh/exp42/curve600/ckpts/lora_step00600.npz
check /scratch/lijh/exp42/curve600/ckpts/params_step00600 /scratch/lijh/exp42/curve600/ckpts/lora_step00600.npz
check /scratch/lijh/exp42/curve1200/ckpt_900             /scratch/lijh/exp42/curve1200/ckpts/lora_step00900.npz

echo
echo '=== 必须保留 ==='
for d in /scratch/lijh/exp42/curve1200/ckpt_1200 \
         /scratch/lijh/exp42/curve1200/ckpts \
         /scratch/lijh/exp42/run2/ckpts/params_nested \
         /scratch/lijh/exp42/overfit/lr4/params_step00150 \
         /scratch/lijh/exp42/data; do
  printf '%-58s %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
echo
df -h /scratch | tail -1
