#!/usr/bin/env bash
# 结束遗留的推理进程（它们仍占用 GPU 显存）
set -u
for P in 42943 42944 42945 42946; do
  if kill "$P" 2>/dev/null; then echo "killed $P"; else echo "gone $P"; fi
done
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
ps -u lijh -o args | grep -c '[e]xpert_state_query' || true
