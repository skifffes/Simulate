#!/usr/bin/env bash
# 在 A40 上用仿真 venv 跑实验 4.0 里不需要 GPU 的部分：环境核验 + Level 3 + Level 2(--no-server) + Level 5(--no-server)
set -u
REPO=/scratch/lijh/exp40/repo
PY=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONIOENCODING=utf-8

echo "=== python / pybullet ==="
"$PY" -V
"$PY" -c "import pybullet, numpy, PIL; print('pybullet', pybullet.getAPIVersion(), 'numpy', numpy.__version__, 'PIL', PIL.__version__)"

echo
echo "=== Level 3：适配器单元测试（含真实仿真 FK/夹爪方向） ==="
"$PY" scripts/exp40/level3_adapter_checks.py 2>&1 | tail -40

echo
echo "=== Level 2：真实观测链路（不调用模型） ==="
"$PY" scripts/exp40/level2_offline_sim_obs.py --scene B90 --no-server 2>&1 | tail -15
"$PY" scripts/exp40/level2_offline_sim_obs.py --scene A90 --no-server 2>&1 | tail -15

echo
echo "=== Level 5：A90/B90 接口（不调用模型） ==="
"$PY" scripts/exp40/level5_env_interface.py --no-server 2>&1 | tail -30

echo
echo "=== 产物 ==="
find "$REPO/results/experiment_4_0" -type f | sort | head -40
