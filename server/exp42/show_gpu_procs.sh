#!/usr/bin/env bash
set -u
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo '--- processes ---'
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  ps -p "$P" -o pid,etime,args --no-headers 2>/dev/null | cut -c1-140 || echo "$P gone"
done
