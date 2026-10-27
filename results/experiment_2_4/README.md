# 实验 2.4 结果目录

## 这里有什么

- `G0_a0b_direct/` … `G5_a0b_local_retreat/`：每组 10 个正式 rollout，每个 `seed_*/` 含
  `top.mp4`、`side.mp4`、`contact.mp4`、`comparison.mp4`、`events.mp4`，
  以及 `physics_trace.csv`、`contact_trace.csv`、`policy_trace.jsonl`、
  `success_timeline.csv`、`frame_index.csv`、`event_summary.json`、`metadata.json`。
- `summary.csv`：六组成功率、抬升率、墙接触率、双指接触率。
- `video_manifest.csv`：全部视频索引（组、seed、成功、失败类型、最大抬升、视频路径）。
- `analysis.json` / `case_analysis.json`：分组统计、代表性案例、Q5 列表、首次接触 link、抬升前 EE 运动。
- `figures/`：Figure 1–8。
- `models_D0/`、`models_D1/`：Q8 诊断的 5 次 G0 rollout（不同 checkpoint）。
- 大体积视频保留在服务器：`/scratch/lijh/simulate_vla/results/experiment_2_4/`。

## 先看哪几个视频（按顺序）

1. **最典型的抓取抬升**：`G0_a0b_direct/seed_40000002/comparison.mp4`
   - side 视角能直接看到：切换后夹爪闭合、把物体夹住、抬离桌面；top 视角看到物体被带向 +X。
2. **A90→B 的最严重抬升**：`G1_a90b_direct/seed_40000005/side.mp4`
3. **同一 B 的干净推动**：`G3_b_home_yaw90/seed_40000000/side.mp4`
4. **yaw0 的翻倒/推转**：`G2_b_home_yaw0/seed_40000000/side.mp4`
5. **退回 HOME 后的失败**：`G4_a0b_retreat_home/seed_40000000/side.mp4`
6. **局部退让后完全不接触**：`G5_a0b_local_retreat/seed_40000002/side.mp4`

每个目录里的 `events.mp4` 是最大抬升、首次墙接触、双指接触等事件前后各 2 s 的剪辑；`top.mp4` 适合看平面轨迹，`contact.mp4` 适合看双指是否夹住。

## 如何精确定位某一时刻

1. 打开 `frame_index.csv`，找到目标 `frame_index`；
2. 同一行给出 `physics_step`、`physics_time_sec`、`policy_step`、`skill`；
3. 在 `physics_trace.csv` 中用 `physics_step` 查机器人/物体状态；
4. 在 `contact_trace.csv` 中用 `physics_step` 查具体 body/link 接触；
5. 在 `policy_trace.jsonl` 中用 `policy_step` 查当时的语言指令、状态与动作；
6. 在 `success_timeline.csv` 中用 `policy_step` 查 B 的各子条件。

视频帧率 20 fps，`frame_index.csv` 的帧数与视频帧数一致；`metadata.json` 记录了 checkpoint、sha256、commit、PyBullet 版本、分辨率与判据。

## 复现

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src venv/bin/python \
  scripts/run_experiment_2_4.py --group G0_a0b_direct --episodes 10 \
  --seed-base 40000000 \
  --policy outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model \
  --out results/experiment_2_4
```

完整命令见 `docs/experiment_2_4.md` 第 11 节。
