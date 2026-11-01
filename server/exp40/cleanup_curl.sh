#!/usr/bin/env bash
# 清理旧版并发下载残留的 curl（它们和正式下载抢同一个不稳定出口）
set -u
killed=0
for pid in $(ps -eo pid,args | grep -F 'ckpt_parts' | grep -v grep | awk '{print $1}'); do
  kill -9 "$pid" 2>/dev/null && killed=$((killed+1))
done
for pid in $(ps -eo pid,args | grep -F 'ckpt_par' | grep -v grep | awk '{print $1}'); do
  kill -9 "$pid" 2>/dev/null && killed=$((killed+1))
done
sleep 1
echo "killed=$killed"
echo "剩余与 openpi-assets 相关的 curl:"
ps -u "$(whoami)" -o pid,etime,args | grep -F 'openpi-assets' | grep -v grep | head -5
tail -2 /scratch/lijh/exp40/download_ckpt6.log
