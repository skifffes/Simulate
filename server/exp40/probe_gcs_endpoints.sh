#!/usr/bin/env bash
# GCS 备选端点/IP 族测试（4MB range，30s 上限）
set -u
cd /tmp || exit 1
P="openpi-assets/checkpoints/pi05_droid/params/ocdbt.process_0/d/213a5648ef56840990e3a9d6fca5eb89"
try() {
  name=$1; url=$2; extra=${3:-}
  timeout 45 curl -sS $extra -m 40 -r 0-4194303 -o /tmp/t_$name.bin \
    -w "$name http=%{http_code} size=%{size_download} speed_Bps=%{speed_download} time=%{time_total}\n" "$url" \
    || echo "$name FAIL"
}
try std        "https://storage.googleapis.com/$P"
try dl         "https://storage-download.googleapis.com/$P"
try http       "http://storage.googleapis.com/$P"
try v4         "https://storage.googleapis.com/$P" "-4"
try dl_v4      "https://storage-download.googleapis.com/$P" "-4"

echo
echo "=== 解析到的 IP ==="
getent ahosts storage.googleapis.com | head -6
getent ahosts storage-download.googleapis.com | head -6

echo
echo "=== 直连某个 IP（绕过 DNS） ==="
ip=$(getent ahosts storage.googleapis.com | awk 'NR==1{print $1}')
echo "using $ip"
timeout 45 curl -sS -m 40 -r 0-4194303 --resolve "storage.googleapis.com:443:$ip" \
  -o /tmp/t_ip.bin -w "ip http=%{http_code} size=%{size_download} speed_Bps=%{speed_download}\n" \
  "https://storage.googleapis.com/$P" || echo "ip FAIL"
