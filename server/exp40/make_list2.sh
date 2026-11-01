#!/usr/bin/env bash
# 生成 checkpoint 对象清单：用 XML API（已实测 6.8s 返回）+ 正则按 Contents 块解析，
# 避免 JSON API / urllib 在该网络下的长尾延迟。
set -u
WS=/scratch/lijh/exp40
LIST="$WS/pi05_droid_objects.tsv"
PY310=/scratch/lijh/.conda/envs/gxc/bin/python3.10

if [ ! -s "$WS/ckpt_xml.xml" ]; then
  curl -4 -sS -m 300 --retry 5 --retry-delay 3 \
    "https://storage.googleapis.com/openpi-assets?prefix=checkpoints/pi05_droid/&max-keys=1000" \
    -o "$WS/ckpt_xml.xml"
fi
ls -l "$WS/ckpt_xml.xml"

"$PY310" - "$WS/ckpt_xml.xml" "$LIST" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8", errors="replace").read()
blocks = re.findall(r"<Contents>(.*?)</Contents>", text, flags=re.S)
rows = []
for b in blocks:
    key = re.search(r"<Key>(.*?)</Key>", b).group(1)
    size = int(re.search(r"<Size>(.*?)</Size>", b).group(1))
    et = re.search(r"<ETag>(.*?)</ETag>", b)
    md5 = et.group(1).strip('"') if et else ""
    rows.append((key, size, md5))
rows.sort()
with open(dst, "w") as fh:
    for n, s, m in rows:
        fh.write(f"{n}\t{s}\t{m}\n")
total = sum(r[1] for r in rows)
print(f"objects={len(rows)} total_gb={total/1073741824:.3f}")
PY
head -3 "$LIST"
awk -F'\t' '{s+=$2} END {printf "sum=%.2f GB\n", s/1073741824}' "$LIST"
