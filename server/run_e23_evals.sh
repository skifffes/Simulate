#!/usr/bin/env bash
# 实验 2.3：D1/D2 训练完成后的评测编排（4 个 GPU 并行）。
# 用法（服务器）：bash server/run_e23_evals.sh
set -u
cd /scratch/lijh/simulate_vla
VENV=venv/bin/python
export HF_HUB_OFFLINE=1
export PYTHONPATH=src
E=results/experiment_2_3/eval
P1=outputs/smolvla_e23_D1/checkpoints/020000/pretrained_model
P2=outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model
SEED=40000000

for p in "$P1" "$P2"; do
  if [ ! -f "$p/config.json" ]; then
    echo "missing checkpoint: $p" >&2
    exit 1
  fi
done

echo "[eval] D1/D2 standard atomic (GPU0/4)"
CUDA_VISIBLE_DEVICES=0 setsid nohup "$VENV" -u scripts/eval_multi_atomic_vla.py \
  --policy "$P1" --skills A0 A90 B C --episodes 20 --seed-base $SEED \
  --out "$E/atomic_std/D1" > logs/e23_eval_D1_std.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 setsid nohup "$VENV" -u scripts/eval_multi_atomic_vla.py \
  --policy "$P2" --skills A0 A90 B C --episodes 20 --seed-base $SEED \
  --out "$E/atomic_std/D2" > logs/e23_eval_D2_std.log 2>&1 &

echo "[eval] D1 composition+snapshot (GPU1), D2 composition+snapshot (GPU3)"
CUDA_VISIBLE_DEVICES=1 setsid nohup bash -c "
  '$VENV' -u scripts/eval_composition_e23.py --policy '$P1' --chain A0_B --episodes 20 \
    --tag D1 --out '$E/composition/D1' &&
  '$VENV' -u scripts/eval_composition_e23.py --policy '$P1' --chain A90_B --episodes 20 \
    --tag D1 --out '$E/composition/D1' &&
  '$VENV' -u scripts/eval_snapshot_b_e23.py --policy '$P1' --a-skill A0 --episodes 20 \
    --tag D1 --out '$E/snapshot/D1' &&
  '$VENV' -u scripts/eval_snapshot_b_e23.py --policy '$P1' --a-skill A90 --episodes 20 \
    --tag D1 --out '$E/snapshot/D1'
" > logs/e23_eval_D1_chain.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 setsid nohup bash -c "
  '$VENV' -u scripts/eval_composition_e23.py --policy '$P2' --chain A0_B --episodes 20 \
    --tag D2 --out '$E/composition/D2' &&
  '$VENV' -u scripts/eval_composition_e23.py --policy '$P2' --chain A90_B --episodes 20 \
    --tag D2 --out '$E/composition/D2' &&
  '$VENV' -u scripts/eval_snapshot_b_e23.py --policy '$P2' --a-skill A0 --episodes 20 \
    --tag D2 --out '$E/snapshot/D2' &&
  '$VENV' -u scripts/eval_snapshot_b_e23.py --policy '$P2' --a-skill A90 --episodes 20 \
    --tag D2 --out '$E/snapshot/D2'
" > logs/e23_eval_D2_chain.log 2>&1 &

echo "[eval] started"
