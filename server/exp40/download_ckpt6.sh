#!/usr/bin/env bash
# 实验 4.0：官方 pi05_droid checkpoint 下载（v6，多域名轮询 + Range 续传 + 单连接）。
#
# 该出口对 Google 的连接极不稳定（同一域名时而 1.5 MB/s、时而 25~55 s 完全连不上），
# 因此每次尝试按顺序轮换三个可用端点，成功写入的文件长度即为断点续传进度。
set -u
WS=/scratch/lijh/exp40
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
CKPT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid"
LIST="$WS/pi05_droid_objects.tsv"
LOG="$WS/download_ckpt6.log"
ERRS="$WS/download_errors6.log"
REL_PREFIX="checkpoints/pi05_droid"
HOSTS=(
  "https://openpi-assets.storage.googleapis.com/$REL_PREFIX"
  "https://storage-download.googleapis.com/openpi-assets/$REL_PREFIX"
  "https://storage.googleapis.com/openpi-assets/$REL_PREFIX"
)
mkdir -p "$CKPT"

{
echo "=== $(date -Is) 下载 v6（多域名轮询） ==="
awk -F'\t' '{s+=$2} END {printf "目标总大小 %.2f GB\n", s/1073741824}' "$LIST"
total_want=$(awk -F'\t' '{s+=$2} END {print s}' "$LIST")

while IFS=$'\t' read -r key size md5hex; do
  rel="${key#checkpoints/pi05_droid/}"
  [ -z "$rel" ] && continue
  dst="$CKPT/$rel"
  mkdir -p "$(dirname "$dst")"
  size=$((size))
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$have" -eq "$size" ]; then echo "SKIP $rel"; continue; fi
  [ "$have" -gt "$size" ] && { : > "$dst"; have=0; }
  echo "=== $(date +%H:%M:%S) START $rel want=$size have=$have"
  attempts=0
  while :; do
    have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    [ "$have" -ge "$size" ] && break
    attempts=$((attempts + 1))
    if [ "$attempts" -gt 400 ]; then echo "GIVEUP $rel have=$have"; break; fi
    host="${HOSTS[$(( (attempts - 1) % ${#HOSTS[@]} ))]}"
    t0=$(date +%s)
    curl -4 -sS -r "${have}-$((size - 1))" -m 1200 --connect-timeout 25 \
      "$host/$rel" >> "$dst" 2>>"$ERRS"
    new=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    dt=$(( $(date +%s) - t0 )); [ "$dt" -lt 1 ] && dt=1
    echo "    try=$attempts host=$(echo "$host" | cut -d/ -f3) ${new}/${size} +$((new - have))B/${dt}s => $(( (new - have) / dt ))B/s"
  done
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  [ "$have" -eq "$size" ] && echo "OK   $rel $size" || echo "BAD  $rel $have/$size"
  got=$(du -sb "$CKPT" 2>/dev/null | cut -f1)
  echo "    累计 ${got}/${total_want} bytes ($(( got * 100 / total_want ))%)"
done < "$LIST"

echo "=== 汇总 ==="
du -sh "$CKPT"
echo "=== $(date -Is) 结束 ==="
} > "$LOG" 2>&1
