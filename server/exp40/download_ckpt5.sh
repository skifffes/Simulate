#!/usr/bin/env bash
# 实验 4.0：官方 pi05_droid checkpoint 下载（v5）。
#
# 实测结论（server/exp40/test_range.sh）：
#   * 不带 Range 的普通 GET 在该出口经常 0 字节超时；
#   * 带 Range 的 GET（-r 0-）能稳定返回；
#   * 并发连接会互相压死，因此始终单连接。
# 因此 v5 全程用显式 Range + 追加写入实现断点续传：
#   curl -r <have>-<size-1> 直接 append 到目标文件，文件大小即为进度。
set -u
WS=/scratch/lijh/exp40
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
CKPT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid"
HOST="https://storage-download.googleapis.com/openpi-assets"
LIST="$WS/pi05_droid_objects.tsv"
LOG="$WS/download_ckpt5.log"
ERRS="$WS/download_errors5.log"
mkdir -p "$CKPT"

{
echo "=== $(date -Is) 下载 v5（Range 续传） ==="
awk -F'\t' '{s+=$2} END {printf "目标总大小 %.2f GB\n", s/1073741824}' "$LIST"

while IFS=$'\t' read -r key size md5hex; do
  rel="${key#checkpoints/pi05_droid/}"
  [ -z "$rel" ] && continue
  dst="$CKPT/$rel"
  mkdir -p "$(dirname "$dst")"
  size=$((size))
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$have" -eq "$size" ]; then echo "SKIP $rel ($size)"; continue; fi
  if [ "$have" -gt "$size" ]; then echo "TRUNC $rel"; : > "$dst"; have=0; fi
  echo "=== $(date +%H:%M:%S) START $rel want=$size have=$have"
  tries=0
  while :; do
    have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    [ "$have" -ge "$size" ] && break
    tries=$((tries + 1))
    if [ "$tries" -gt 300 ]; then echo "GIVEUP $rel tries=$tries have=$have"; break; fi
    t0=$(date +%s)
    curl -4 -sS -r "${have}-$((size - 1))" -m 1800 --connect-timeout 120 \
      "$HOST/checkpoints/pi05_droid/$rel" >> "$dst" 2>>"$ERRS"
    new=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    dt=$(( $(date +%s) - t0 )); [ "$dt" -lt 1 ] && dt=1
    echo "    try=$tries ${new}/${size} (+$((new - have)) B in ${dt}s => $(( (new - have) / dt )) B/s)"
  done
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$have" -eq "$size" ]; then
    echo "OK   $rel $size md5=$(md5sum "$dst" | cut -d' ' -f1)"
  else
    echo "BAD  $rel $have/$size"
  fi
done < "$LIST"

echo "=== 汇总 ==="
du -sh "$CKPT"
echo "=== $(date -Is) 结束 ==="
} > "$LOG" 2>&1
echo "done log=$LOG"
tail -20 "$LOG"
