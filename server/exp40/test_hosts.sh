#!/usr/bin/env bash
# 试更多 GCS 访问域名/路径（各拉 4MB range，25s 上限）
set -u
cd /tmp || exit 1
OBJ="checkpoints/pi05_droid/params/ocdbt.process_0/d/3d66a3d641ef2a488f1a3dad19dcc9fa"
try() {
  name=$1; url=$2
  timeout 30 curl -4 -sS -m 25 -r 0-4194303 -o /tmp/e_$name.bin \
    -w "$name http=%{http_code} size=%{size_download} speed=%{speed_download} t=%{time_total}\n" "$url" || echo "$name FAIL"
}
try vhost   "https://openpi-assets.storage.googleapis.com/$OBJ"
try jsonapi "https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/${OBJ//\//%2F}?alt=media"
try gapis   "https://www.googleapis.com/download/storage/v1/b/openpi-assets/o/${OBJ//\//%2F}?alt=media"
try dl      "https://storage-download.googleapis.com/openpi-assets/$OBJ"
