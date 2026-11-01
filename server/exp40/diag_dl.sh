#!/usr/bin/env bash
# 诊断并发下载为何停滞：看 part 文件大小/mtime，并手工并发 3 段对比
set -u
P=/scratch/lijh/exp40/ckpt_parts
echo "=== part 文件（前 10） ==="
find "$P" -name '*.part*' -printf '%s\t%TT\t%p\n' 2>/dev/null | head -10
echo "总数: $(find "$P" -name '*.part*' | wc -l)"
echo
echo "=== 手工并发 3 段（每段 8MB，60s 上限） ==="
HOST="https://storage-download.googleapis.com/openpi-assets/checkpoints/pi05_droid/params/ocdbt.process_0/d"
S=213a5648ef56840990e3a9d6fca5eb89
t0=$(date +%s)
for i in 1 2 3; do
  s=$((i * 8388608)); e=$((s + 8388607))
  ( timeout 70 curl -4 -sS -m 60 -r ${s}-${e} -o /tmp/m_$i.bin "$HOST/$S" ) &
done
wait
t1=$(date +%s)
total=$(du -cb /tmp/m_*.bin 2>/dev/null | tail -1 | cut -f1)
echo "3 段合计 $total bytes / $((t1-t0))s"

echo
echo "=== 现有 curl 进程的完整命令行 ==="
ps -eo pid,etime,args | grep '[c]url' | head -4
