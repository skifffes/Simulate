#!/usr/bin/env bash
# 实验 4.0：官方 pi05_droid checkpoint 的顺序断点续传下载（v3）。
#
# 结论依据（server/exp40/measure_dl.sh / diag_dl.sh 的实测）：
#   * 单连接偶尔可达 3.7 MB/s，12 路并发时互相把对方压到 13 KB/s —— 该出口对
#     同一目标的并发连接极不友好，所以改为“单连接 + 断点续传”，用 --speed-limit
#     主动放弃停滞连接再续传，而不是靠并发。
set -u
WS=/scratch/lijh/exp40
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
CKPT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid"
HOST="https://storage-download.googleapis.com/openpi-assets"
LIST="$WS/pi05_droid_objects.tsv"
LOG="$WS/download_ckpt3.log"
mkdir -p "$CKPT"

{
echo "=== $(date -Is) 顺序断点续传下载开始 ==="
awk -F'\t' '{s+=$2} END {printf "目标总大小 %.2f GB\n", s/1073741824}' "$LIST"

while IFS=$'\t' read -r key size md5hex; do
  rel="${key#checkpoints/pi05_droid/}"
  [ -z "$rel" ] && continue
  dst="$CKPT/$rel"
  mkdir -p "$(dirname "$dst")"
  size=$((size))
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$have" -eq "$size" ]; then echo "SKIP $rel ($size)"; continue; fi
  echo "=== $(date +%H:%M:%S) $rel want=$size have=$have"
  tries=0
  while :; do
    have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    [ "$have" -ge "$size" ] && break
    tries=$((tries + 1))
    [ "$tries" -gt 400 ] && { echo "GIVEUP $rel tries=$tries"; break; }
    curl -4 -sS -C - -o "$dst" -m 900 \
      --speed-limit 200000 --speed-time 15 \
      --retry 0 "$HOST/checkpoints/pi05_droid/$rel" 2>>"$WS/download_errors.log"
    have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    # 服务器不支持断点时 curl 会报 416，这里截断重下
    if [ -s "$WS/download_errors.log" ] && grep -q "416" "$WS/download_errors.log"; then
      echo "416 on $rel -> restart"; : > "$dst"; : > "$WS/download_errors.log"
    fi
  done
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  md5=$(md5sum "$dst" | cut -d' ' -f1)
  if [ "$have" -eq "$size" ]; then
    echo "OK   $rel  $size  md5=$md5"
  else
    echo "BAD  $rel have=$have want=$size"
  fi
done < "$LIST"

echo "=== 汇总 ==="
du -sh "$CKPT"
find "$CKPT" -type f | wc -l
echo "=== $(date -Is) 结束 ==="
} > "$LOG" 2>&1
echo "done log=$LOG"
tail -20 "$LOG"
