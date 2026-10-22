#!/usr/bin/env bash
# GPU3 批处理：strict ss2 对照 → 扩充 40 种子 n10 → OOD friction → 专家 OOD 三组。
cd "$HOME/simulate_vla"
export GPU=3
run() { echo "[batch3] start $* $(date +%T)"; "$@"; echo "[batch3] rc=$? $* $(date +%T)"; }

run env MODE=strict SS=2 NA=10 OUT=strict_n10_ss2_iid20 VIDEOS=0 bash server/e11_run.sh
run env MODE=strict SS=1 NA=10 SEEDS="$(seq -s, 30000020 30000059)" OUT=strict_n10_iid40new VIDEOS=0 bash server/e11_run.sh
run env MODE=strict SS=1 SEEDS="$(seq -s, 30000000 30000009)" FRICTION=0.4 OUT=ood_friction_chunk50 VIDEOS=0 bash server/e11_run.sh
run env EXPERT=1 SS=1 SEEDS="$(seq -s, 30000000 30000009)" SHIFT_X=0.03 OUT=expert_ood_shiftx VIDEOS=0 bash server/e11_run.sh
run env EXPERT=1 SS=1 SEEDS="$(seq -s, 30000000 30000009)" YAW=0.5 OUT=expert_ood_yaw VIDEOS=0 bash server/e11_run.sh
run env EXPERT=1 SS=1 SEEDS="$(seq -s, 30000000 30000009)" FRICTION=0.4 OUT=expert_ood_friction VIDEOS=0 bash server/e11_run.sh
echo "[batch3] ALL DONE $(date +%T)"
