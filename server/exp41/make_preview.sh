#!/usr/bin/env bash
# 从数据集中抽取少量可视化样本（Git 里只放小图，视频留在服务器）
set -u
REPO=/scratch/lijh/exp40/repo
OUT="$REPO/results/experiment_4_1/dataset/preview"
mkdir -p "$OUT"
FF=$(command -v ffmpeg || true)
echo "ffmpeg: ${FF:-none}"
ls "$REPO/data/exp41/A90/videos" 2>/dev/null | head -3
V_TOP_A=$(find "$REPO/data/exp41/A90/videos" -name 'episode_000000.mp4' | grep -m1 'top' || true)
V_WRI_A=$(find "$REPO/data/exp41/A90/videos" -name 'episode_000000.mp4' | grep -m1 'wrist' || true)
V_TOP_B=$(find "$REPO/data/exp41/B90/videos" -name 'episode_000000.mp4' | grep -m1 'top' || true)
V_WRI_B=$(find "$REPO/data/exp41/B90/videos" -name 'episode_000000.mp4' | grep -m1 'wrist' || true)
echo "$V_TOP_A"; echo "$V_WRI_A"; echo "$V_TOP_B"; echo "$V_WRI_B"
if [ -n "$FF" ]; then
  for spec in "A90_top:$V_TOP_A" "A90_wrist:$V_WRI_A" "B90_top:$V_TOP_B" "B90_wrist:$V_WRI_B"; do
    name=${spec%%:*}; path=${spec#*:}
    [ -z "$path" ] && continue
    # 末帧（终态）：取倒数第 2 帧
    "$FF" -y -loglevel error -sseof -0.15 -i "$path" -frames:v 1 "$OUT/${name}_final.png" || echo "fail $name"
    "$FF" -y -loglevel error -i "$path" -vf "select=eq(n\,0)" -frames:v 1 "$OUT/${name}_first.png" || true
  done
fi
ls -la "$OUT"
