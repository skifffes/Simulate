#!/usr/bin/env bash
# 实验 4.0：官方 pi05_droid checkpoint 的顺序断点续传下载（v4）。
#
# v3 的问题：对 4.4KB 的小文件也设了 --speed-limit，导致 TTFB 期间被判定超速限而反复中断。
# v4：
#   * 小文件（<=32MB）不做速度限制，只用 -m 3600；
#   * 大文件用 --speed-limit 20000 --speed-time 120，只在真正停滞 2 分钟时中断并续传；
#   * 每 30 秒打印一次进度，便于远程观察。
set -u
WS=/scratch/lijh/exp40
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
CKPT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid"
HOST="https://storage-download.googleapis.com/openpi-assets"
LIST="$WS/pi05_droid_objects.tsv"
LOG="$WS/download_ckpt4.log"
ERRS="$WS/download_errors4.log"
mkdir -p "$CKPT"

{
echo "=== $(date -Is) 顺序断点续传下载 v4 ==="
awk -F'\t' '{s+=$2} END {printf "目标总大小 %.2f GB\n", s/1073741824}' "$LIST"

while IFS=$'\t' read -r key size md5hex; do
  rel="${key#checkpoints/pi05_droid/}"
  [ -z "$rel" ] && continue
  dst="$CKPT/$rel"
  mkdir -p "$(dirname "$dst")"
  size=$((size))
  if [ "$(stat -c %s "$dst" 2>/dev/null || echo 0)" -eq "$size" ]; then echo "SKIP $rel"; continue; fi
  echo "=== $(date +%H:%M:%S) START $rel want=$size"
  tries=0
  while :; do
    have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    [ "$have" -ge "$size" ] && break
    tries=$((tries + 1))
    if [ "$tries" -gt 60 ]; then echo "GIVEUP $rel tries=$tries have=$have"; break; fi
    t_start=$(date +%s)
    if [ "$size" -le $((32 * 1024 * 1024)) ]; then
      curl -4 -sS -C - -o "$dst" -m 3600 --connect-timeout 120 "$HOST/checkpoints/pi05_droid/$rel" 2>>"$ERRS"
    else
      curl -4 -sS -C - -o "$dst" -m 3600 --connect-timeout 120 \
        --speed-limit 20000 --speed-time 120 "$HOST/checkpoints/pi05_droid/$rel" 2>>"$ERRS"
    fi
    new=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    dt=$(( $(date +%s) - t_start )); [ "$dt" -lt 1 ] && dt=1
    echo "    try=$tries $new/$size (+$((new - have)) B in ${dt}s => $(( (new - have) / dt )) B/s)"
  done
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  echo "$([ "$have" -eq "$size" ] && echo OK || echo BAD) $rel $have/$size"
done < "$LIST"

echo "=== 汇总 ==="
du -sh "$CKPT"
echo "=== $(date -Is) 结束 ==="
} > "$LOG" 2>&1
echo "done log=$LOG"
tail -20 "$LOG"
