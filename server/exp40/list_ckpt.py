"""列出 GCS bucket 中某个前缀下的对象（name, size, md5）。

用 JSON API 而不是 XML API：XML 的 Contents 块里 ETag 出现在 Size 之前，
用文本流切分容易错位；JSON 直接给结构化字段。

用法：
    python list_ckpt.py checkpoints/pi05_droid/ > objects.tsv
输出：name<TAB>size<TAB>md5_b64<TAB>md5_hex
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.parse
import urllib.request

BUCKET = "openpi-assets"


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/pi05_droid/"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    rows = []
    token = None
    while True:
        params = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            params["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for item in payload.get("items", []):
            md5_b64 = item.get("md5Hash", "")
            md5_hex = base64.b64decode(md5_b64).hex() if md5_b64 else ""
            rows.append((item["name"], int(item["size"]), md5_b64, md5_hex))
        token = payload.get("nextPageToken")
        if not token:
            break
    rows.sort()
    lines = [f"{n}\t{s}\t{m5b}\t{m5h}" for n, s, m5b, m5h in rows]
    text = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    sys.stdout.write(text)
    total = sum(r[1] for r in rows)
    print(f"# objects={len(rows)} total_bytes={total} total_gb={total / 1073741824:.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
