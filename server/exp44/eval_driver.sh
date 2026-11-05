#!/usr/bin/env bash
# 实验 4.4：增量离线评测驱动（按 step 轮转 A/B，等 checkpoint 出现且大小稳定后评测，GPU2）
# 已完成的 analysis json 会跳过，可安全重启。
set -u
STEPS="1200 1800 2400 3600 4800 6000 8000 10000 12000"
LOG=/scratch/lijh/exp44/eval_driver.log
for S in $STEPS; do
  for G in A B; do
    AN="/scratch/lijh/exp40/repo/results/experiment_4_4/offline/${G}_step${S}_val_analysis.json"
    if [ -f "$AN" ]; then
      echo "[$(date +%H:%M:%S)] skip $G $S (done)" >> "$LOG"
      continue
    fi
    for i in $(seq 1 720); do
      F="/scratch/lijh/exp44/$G/ckpts/lora_step$(printf '%05d' "$S").npz"
      if [ -f "$F" ]; then
        S1=$(stat -c %s "$F"); sleep 10; S2=$(stat -c %s "$F")
        if [ "$S1" = "$S2" ] && [ "$S1" -gt 1000000000 ]; then break; fi
      fi
      sleep 10
    done
    echo "[$(date +%H:%M:%S)] evaluating $G $S" >> "$LOG"
    EVAL_GPU=2 bash /scratch/lijh/exp44/eval_ckpt.sh "$G" "$S" 0,1,2,3 >> "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] finished $G $S" >> "$LOG"
  done
done
echo "[$(date +%H:%M:%S)] driver done" >> "$LOG"
