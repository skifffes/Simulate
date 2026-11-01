#!/usr/bin/env bash
# 1) 打印 pi05_droid 真实对象清单（之前 XML 已抓到的 /tmp/xml_list.txt）
# 2) 查 HF 上可能的 pi05 完整 checkpoint 仓库文件树
# 3) hf-mirror 大文件吞吐测速
set -u
cd /tmp || exit 1

echo "=== pi05_droid 对象清单 ==="
if [ -s /tmp/xml_list.txt ]; then
  tr '<' '\n' < /tmp/xml_list.txt | grep -E "^(Key|Size)>" | sed 's/^[A-Za-z]*>//' | paste - - > /tmp/objs.tsv
  cat /tmp/objs.tsv
  awk -F'\t' '{s+=$2} END {printf "TOTAL %.2f GB (%d objects)\n", s/1024/1024/1024, NR}' /tmp/objs.tsv
else
  echo "no xml_list.txt"
fi

echo
echo "=== HF: lerobot 组织里与 pi05 / droid 相关的仓库 ==="
timeout 60 curl -sS -m 55 "https://hf-mirror.com/api/models?author=lerobot&search=pi05&limit=50" \
  | tr '}' '\n' | grep -o '"modelId":"[^"]*"' | head -20
timeout 60 curl -sS -m 55 "https://hf-mirror.com/api/models?author=lerobot&search=droid&limit=50" \
  | tr '}' '\n' | grep -o '"modelId":"[^"]*"' | head -20

echo
echo "=== HF: 候选仓库文件树 ==="
for repo in "s3y/pi05_droid_pytorch" "april5129/pi05_droid" "zaper27/pi05_real_droid"; do
  echo "--- $repo ---"
  timeout 60 curl -sS -m 55 "https://hf-mirror.com/api/models/${repo}" \
    | tr '}' '\n' | grep -o '"rfilename":"[^"]*"' | head -25
done

echo
echo "=== hf-mirror 吞吐测试（smolvla_base 权重前 16MB） ==="
timeout 200 curl -sS -L -m 190 -r 0-16777215 -o /tmp/hf_head.bin \
  -w "http=%{http_code} size=%{size_download} speed_Bps=%{speed_download} time=%{time_total}\n" \
  "https://hf-mirror.com/lerobot/smolvla_base/resolve/main/model.safetensors" || echo "fail"
ls -l /tmp/hf_head.bin 2>/dev/null
