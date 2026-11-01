#!/usr/bin/env bash
# 官方 checkpoint 到位后的完整验证序列（Level 1–5）。
# 用法: bash run_all_checkpoint.sh [gpu_for_level1]
set -u
WS=/scratch/lijh/exp40
REPO=$WS/repo
VENV=$WS/openpi/.venv
SIM=/scratch/lijh/simulate_vla/venv/bin/python
GPU_L1="${1:-3}"
LOG=$WS/run_all_checkpoint.log

export PYTHONPATH="$WS/lerobot_stub:$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8
export OPENPI_DATA_HOME="$HOME/.cache/openpi"
export OPENPI_COMMIT=$(git -C "$WS/openpi" rev-parse HEAD 2>/dev/null || echo unknown)

cd "$REPO" || exit 1

{
echo "=========== $(date -Is) 开始 checkpoint 全量验证 ==========="

echo "--- 0. 校验 checkpoint 完整性 ---"
$VENV/bin/python "$WS/make_ckpt_manifest.py" \
  --ckpt "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid" \
  --list "$WS/pi05_droid_objects.tsv" \
  --out "$REPO/results/experiment_4_0/checkpoint_manifest.json" 2>&1 | tail -3

echo "--- 1. 停掉旧 server ---"
pkill -f "serve_policy.py" 2>/dev/null
sleep 2
echo "剩余 serve_policy 进程: $(pgrep -fc 'serve_policy.py' || true)"

echo "--- 2. 起 checkpoint server（GPU2 :8000） ---"
bash "$WS/run_server.sh" checkpoint 2 8000 2>&1 | tail -6

echo "--- 3. Level 1：独立推理（GPU$GPU_L1） ---"
CUDA_VISIBLE_DEVICES=$GPU_L1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
  $VENV/bin/python scripts/exp40/level1_model_inference.py --mode checkpoint \
  --out results/experiment_4_0/model_smoke 2>&1 | tail -30

echo "--- 4. Level 2：真实观测离线推理 ---"
for sc in B90 A90; do
  $SIM scripts/exp40/level2_offline_sim_obs.py --scene $sc --port 8000 2>&1 | tail -12
done

echo "--- 5. Level 5：A90/B90 接口检查 ---"
$SIM scripts/exp40/level5_env_interface.py --port 8000 2>&1 | tail -25

echo "--- 6. Level 4：有限闭环（B90 与 A90 各一次，短时） ---"
$SIM scripts/exp40/level4_closed_loop.py --scene B90 --ticks 20 --port 8000 2>&1 | tail -12
$SIM scripts/exp40/level4_closed_loop.py --scene A90 --ticks 20 --port 8000 2>&1 | tail -12

echo "--- 7. 服务器调用记录 ---"
ls results/experiment_4_0/model_smoke/calls_8000/calls 2>/dev/null | wc -l

echo "=========== $(date -Is) 结束 ==========="
} > "$LOG" 2>&1
echo "done log=$LOG"
grep -E "PASS|FAIL|Level|output shape|steady|checkpoint|stop_reason|ticks|OK |BAD " "$LOG" | tail -60
