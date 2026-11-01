#!/usr/bin/env bash
# 停掉任何下载脚本（脚本内部匹配，避免命令行里出现模式串误杀自己）
set -u
for pat in 'download_ckpt' 'make_list' 'list_ckpt'; do
  pkill -f "$pat" 2>/dev/null
done
for pid in $(ps -eo pid,args | grep -F 'openpi-assets/checkpoints' | grep -v grep | awk '{print $1}'); do
  kill -9 "$pid" 2>/dev/null
done
sleep 1
echo "剩余 curl 进程: $(ps -eo args | grep -c '[c]url')"
