#!/usr/bin/env bash
# 用 curl -4 抓 GCS JSON 列表（urllib 在该网络下可能走 IPv6 卡住），再用本地文件解析生成 TSV。
set -u
WS=/scratch/lijh/exp40
LIST="$WS/pi05_droid_objects.tsv"
PY310=/scratch/lijh/.conda/envs/gxc/bin/python3.10

curl -4 -sS -m 300 --retry 5 --retry-delay 3 \
  "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_droid/&maxResults=1000&fields=items(name,size,md5Hash)" \
  -o "$WS/ckpt_list.json"
ls -l "$WS/ckpt_list.json"

"$PY310" - "$WS/ckpt_list.json" "$LIST" <<'PY'
import base64, json, sys
src, dst = sys.argv[1], sys.argv[2]
items = json.load(open(src))["items"]
rows = []
for it in items:
    md5 = base64.b64decode(it["md5Hash"]).hex() if it.get("md5Hash") else ""
    rows.append((it["name"], int(it["size"]), md5))
rows.sort()
with open(dst, "w") as fh:
    for n, s, m in rows:
        fh.write(f"{n}\t{s}\t{m}\n")
total = sum(r[1] for r in rows)
print(f"objects={len(rows)} total={total/1073741824:.3f} GB")
PY
wc -l "$LIST"
