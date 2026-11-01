#!/usr/bin/env bash
# 实验 4.0：预下载官方 pi05_droid checkpoint 到 openpi 的默认缓存目录。
#
# 为什么自己下：openpi 的 download.py 对 gs://openpi-assets 优先用 gsutil，缺失时回退 gcsfs，
# 该 bucket 的 gcsfs 匿名访问不稳定；这里直接用 HTTPS XML API 拉取，可续传、可校验，
# 落到 openpi 期望的缓存路径后，maybe_download() 会直接命中并返回。
#
# 目标布局：$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid/<相对路径>
set -u
WS=/scratch/lijh/exp40
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
CKPT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid"
BASE="https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid"
LIST="$WS/pi05_droid_objects.tsv"
LOG="$WS/download_ckpt.log"

mkdir -p "$CKPT"

{
echo "=== $(date -Is) 开始下载 pi05_droid ==="
echo "cache: $CKPT"

echo "=== 列出对象 ==="
curl -sS -m 180 --retry 5 --retry-delay 3 \
  "https://storage.googleapis.com/openpi-assets?prefix=checkpoints/pi05_droid/&max-keys=1000" \
  | tr '<' '\n' | grep -E "^(Key|Size|ETag)>" | sed 's/^[A-Za-z]*>//' > "$WS/pi05_droid_objects.raw"
# raw 是 Key Size ETag 的三元组顺序流；用 paste 每 3 行合成一行
paste - - - < "$WS/pi05_droid_objects.raw" > "$LIST"
echo "对象数: $(wc -l < "$LIST")"
awk -F'\t' '{s+=$2} END {printf "总大小: %.2f GB\n", s/1024/1024/1024}' "$LIST"

echo "=== 下载（断点续传，串行 + 每文件重试） ==="
fail=0
while IFS=$'\t' read -r key size etag; do
  rel="${key#checkpoints/pi05_droid/}"
  [ -z "$rel" ] && continue
  dst="$CKPT/$rel"
  mkdir -p "$(dirname "$dst")"
  want=$((size))
  for attempt in 1 2 3 4 5; do
    have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
    if [ "$have" -ge "$want" ]; then break; fi
    echo "[$attempt] $rel  ($have/$want)"
    curl -sS -C - -m 3600 --retry 3 --retry-delay 5 -o "$dst" "$BASE/$rel" || true
  done
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$have" -lt "$want" ]; then
    echo "FAILED $rel ($have < $want)"
    fail=$((fail+1))
  fi
done < "$LIST"

echo "=== 校验大小 ==="
bad=0
while IFS=$'\t' read -r key size etag; do
  rel="${key#checkpoints/pi05_droid/}"
  [ -z "$rel" ] && continue
  have=$(stat -c %s "$CKPT/$rel" 2>/dev/null || echo -1)
  if [ "$have" != "$size" ]; then echo "SIZE MISMATCH $rel have=$have want=$size"; bad=$((bad+1)); fi
done < "$LIST"

echo "=== 磁盘占用 ==="
du -sh "$CKPT"
echo "失败: $fail  大小不一致: $bad"
echo "=== $(date -Is) 结束 ==="
} > "$LOG" 2>&1

echo "done log=$LOG"
tail -20 "$LOG"
