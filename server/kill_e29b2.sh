#!/usr/bin/env bash
pkill -f 'scripts/run_b90_snapshot.py' || true
pkill -f 'scripts/run_b90_e1.py' || true
sleep 2
rm -rf /scratch/lijh/simulate_vla/results/experiment_2_9/b2_low_near/runs
pgrep -af 'run_b90_snapshot|run_b90_e1|run_e29b2' | grep -v pgrep | wc -l
