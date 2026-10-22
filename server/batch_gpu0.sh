#!/usr/bin/env bash
# GPU0 批处理：assisted 对照 → 扩充 40 种子 chunk50 → OOD shift/yaw → 专家 IID。
cd "$HOME/simulate_vla"
export GPU=0
run() { echo "[batch0] start $* $(date +%T)"; "$@"; echo "[batch0] rc=$? $* $(date +%T)"; }

run env MODE=assisted SS=1 OUT=assisted_chunk50_iid20 VIDEOS=0 bash server/e11_run.sh
run env MODE=strict SS=1 SEEDS="$(seq -s, 30000020 30000059)" OUT=strict_chunk50_iid40new VIDEOS=0 bash server/e11_run.sh
run env MODE=strict SS=1 SEEDS="$(seq -s, 30000000 30000009)" SHIFT_X=0.03 OUT=ood_shiftx_chunk50 VIDEOS=0 bash server/e11_run.sh
run env MODE=strict SS=1 SEEDS="$(seq -s, 30000000 30000009)" YAW=0.5 OUT=ood_yaw_chunk50 VIDEOS=0 bash server/e11_run.sh
run env EXPERT=1 SS=1 OUT=expert_iid20 VIDEOS=0 bash server/e11_run.sh
echo "[batch0] ALL DONE $(date +%T)"
