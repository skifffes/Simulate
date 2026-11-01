#!/usr/bin/env bash
# 用 GCS JSON API 列出对象（比 XML 分页快），并测试单文件下载速度
set -u
B=openpi-assets
P=checkpoints/pi05_droid/

echo "=== XML API（10s 超时）实测 ==="
timeout 12 curl -sS -m 10 -o /tmp/xml_list.txt -w "http=%{http_code} size=%{size_download} time=%{time_total}\n" \
  "https://storage.googleapis.com/openpi-assets?prefix=${P}&max-keys=1000" || echo "XML: timeout/fail"
head -c 300 /tmp/xml_list.txt 2>/dev/null; echo

echo
echo "=== JSON API（30s 超时）实测 ==="
timeout 35 curl -sS -m 30 -o /tmp/json_list.txt -w "http=%{http_code} size=%{size_download} time=%{time_total}\n" \
  "https://storage.googleapis.com/storage/v1/b/${B}/o?prefix=${P}&fields=items(name,size,md5Hash)&maxResults=1000" || echo "JSON: timeout/fail"
head -c 500 /tmp/json_list.txt 2>/dev/null; echo

echo
echo "=== 单文件下载测速（_METADATA） ==="
timeout 60 curl -sS -m 55 -o /tmp/_METADATA -w "http=%{http_code} size=%{size_download} speed=%{speed_download}\n" \
  "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/params/_METADATA" || echo "dl: timeout/fail"
ls -l /tmp/_METADATA 2>/dev/null
