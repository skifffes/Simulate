#!/usr/bin/env bash
# 实验 4.0：分段并发下载官方 pi05_droid checkpoint（第二版，用 JSON API 生成清单）。
#
# 依据的测速结果：
#   storage.googleapis.com              单连接 ~0.19 MB/s
#   storage-download.googleapis.com -4  单连接 ~1.5 MB/s
# 策略：16MB 分段、12 路并发、可续传，拼接后按 objects.tsv 的 md5 校验。
set -u
WS=/scratch/lijh/exp40
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
CKPT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid"
PARTS="$WS/ckpt_parts"
HOST="https://storage-download.googleapis.com/openpi-assets"
SEG=$((16 * 1024 * 1024))
JOBS="${JOBS:-12}"
LIST="$WS/pi05_droid_objects.tsv"
LOG="$WS/download_ckpt.log"
PY310=/scratch/lijh/.conda/envs/gxc/bin/python3.10

mkdir -p "$CKPT" "$PARTS"

{
echo "=== $(date -Is) 分段并发下载（v2） ==="

if [ ! -s "$LIST" ]; then
  echo "=== 生成对象清单（JSON API） ==="
  "$PY310" "$WS/list_ckpt.py" "checkpoints/pi05_droid/" "$LIST"
fi
echo "对象数 $(wc -l < "$LIST")"
awk -F'\t' '{s+=$2} END {printf "总大小 %.2f GB\n", s/1073741824}' "$LIST"

JOBFILE="$WS/ckpt_jobs.txt"
: > "$JOBFILE"
while IFS=$'\t' read -r key size etag md5hex; do
  rel="${key#checkpoints/pi05_droid/}"
  [ -z "$rel" ] && continue
  size=$((size))
  if [ "$size" -le "$((SEG * 2))" ]; then
    echo "$rel|0-$((size - 1))" >> "$JOBFILE"
  else
    off=0
    while [ "$off" -lt "$size" ]; do
      end=$((off + SEG - 1)); [ "$end" -ge "$size" ] && end=$((size - 1))
      echo "$rel|$off-$end" >> "$JOBFILE"
      off=$((end + 1))
    done
  fi
done < "$LIST"
echo "分段任务数 $(wc -l < "$JOBFILE")"

fetch_seg() {
  local rel="$1" range="$2" start="${2%%-*}"
  local out="$PARTS/$rel.part$start"
  local url="$HOST/checkpoints/pi05_droid/$rel"
  local want=$(( ${range##*-} - start + 1 ))
  mkdir -p "$(dirname "$out")"
  local have
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  [ "$have" -eq "$want" ] && return 0
  for attempt in 1 2 3 4 5 6 7 8; do
    curl -4 -sS -m 900 --retry 2 --retry-delay 3 -r "$range" -o "$out" "$url" >/dev/null 2>&1
    [ "$(stat -c %s "$out" 2>/dev/null || echo 0)" -eq "$want" ] && return 0
    sleep 2
  done
  echo "SEGFAIL $rel $range" >> "$WS/segment_failures.log"
  return 1
}
export -f fetch_seg
export PARTS HOST

echo "=== 并发下载 JOBS=$JOBS SEG=$SEG ==="
t0=$(date +%s)
xargs -P "$JOBS" -I{} bash -c 'IFS="|" read -r rel range <<< "{}"; fetch_seg "$rel" "$range"' < "$JOBFILE"
t1=$(date +%s)
echo "下载阶段耗时 $((t1 - t0)) s"
du -sh "$PARTS" 2>/dev/null

echo "=== 拼接与校验 ==="
while IFS=$'\t' read -r key size etag md5hex; do
  rel="${key#checkpoints/pi05_droid/}"
  [ -z "$rel" ] && continue
  dst="$CKPT/$rel"
  mkdir -p "$(dirname "$dst")"
  size=$((size))
  if [ "$(stat -c %s "$dst" 2>/dev/null || echo 0)" -eq "$size" ]; then
    echo "skip $rel"; continue
  fi
  : > "$dst"
  off=0
  ok=1
  while [ "$off" -lt "$size" ]; do
    end=$((off + SEG - 1)); [ "$end" -ge "$size" ] && end=$((size - 1))
    part="$PARTS/$rel.part$off"
    if [ ! -f "$part" ]; then echo "MISSING $part"; ok=0; break; fi
    cat "$part" >> "$dst"
    off=$((end + 1))
  done
  if [ "$ok" -eq 0 ]; then continue; fi
  got=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  md5=$(md5sum "$dst" | cut -d' ' -f1)
  if [ "$got" -eq "$size" ] && { [ -z "$md5hex" ] || [ "$md5hex" = "-" ] || [ "$md5" = "$md5hex" ]; }; then
    echo "OK   $rel  $got  md5=$md5"
    rm -f "$PARTS/$rel".part*
  else
    echo "BAD  $rel  have=$got want=$size md5=$md5 want_md5=$md5hex"
  fi
done < "$LIST"

echo "=== 汇总 ==="
du -sh "$CKPT"
head -20 "$WS/segment_failures.log" 2>/dev/null
echo "=== $(date -Is) 结束 ==="
} > "$LOG" 2>&1
echo "done log=$LOG"
tail -30 "$LOG"
