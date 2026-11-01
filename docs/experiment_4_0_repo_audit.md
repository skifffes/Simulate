# 实验 4.0：Simulate 仓库接口审计

审计对象：本仓库现有的 **SmolVLA D2** 观测 / 动作 / 控制接口。目的不是评价模型，而是把"另一个 VLA（π₀.₅ / openpi）要接进来时必须逐项对齐的量"写成可核对的清单。

审计方法：只读源码与已有结果文件，每条结论给 `路径:行号`。凡本仓库没有的东西写 **not found**，不做推测。行号对应当前工作副本。

**D2 身份（后文反复引用）**

| 项 | 值 | 依据 |
| --- | --- | --- |
| 训练集 | `data/lerobot_panda_e23_D2`，480 集 / 68 892 帧 | `docs/experiment_2_3.md:60`、`docs/experiment_2_3.md:107` |
| 其中 B 分片 | `data/_shards_e23/B_D2`，120 集 / 14 653 帧（60 HOME + 59 多样起点 + 1 次 HOME 回退） | `docs/experiment_2_3_data.md:10`、`docs/experiment_2_3.md:60`、`docs/experiment_2_5_d2_coverage.md:76` |
| checkpoint | `outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model`，sha256 `34a6e308…` | `docs/experiment_2_4.md:30`、`docs/experiment_2_5.md:37` |
| 基座与预算 | `lerobot/smolvla_base`，20 000 步，batch 8，lr 1e-4 | `server/train_smolvla_multi.sh:31-45`、`docs/experiment_2_3.md:109` |
| 本仓库自身的 π₀.₅ 适配代码 | **not found**（`src/`、`scripts/`、`server/` 内无 `openpi/pi05` 引用） | — |
| π₀.₅ 参考副本 | 审计期间出现在仓库根的 `.openpi_ref/openpi`（含 π₀.₅：`src/openpi/training/config.py:126,573-574`）。本审计**不覆盖**它，因此迁移表 π₀.₅ 列保持 TBD | `.openpi_ref/openpi/README.md:3-8` |

---

## 1. Panda 7 关节模型与索引

**机器人 body**：`p.loadURDF('franka_panda/panda.urdf', useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)`，body id 存在 `sim.robot`，名字字符串只有 `'franka_panda/panda.urdf'`，没有单独的 body 名（`src/simulation.py:35-36`）。同一 `reset()` 里还创建 `plane.urdf`（`src/simulation.py:30`）和可选参考 `cube_small.urdf`（`src/simulation.py:31-32`，VLA 场景关闭：`src/vla/scenario.py:87`、`src/vla/atomic_tasks/scene2.py:96`）。

**7 个手臂关节**：`self.joints = list(range(7))`（`src/simulation.py:45`），即 joint index 0–6 = `panda_joint1..panda_joint7`。

**`ee` link index = 11**，注释写明是 `panda_grasptarget`（`src/simulation.py:46`）。实测（只读加载同一 URDF，打印 `getJointInfo` 的 joint 名/child link 名/限位）：joint 11 = `panda_grasptarget_hand`（child link `panda_grasptarget`，z≈0.821 m，比双指低 4.7 cm）；**link 8 = `panda_hand`**（z≈0.926 m）；link 9/10 = `panda_leftfinger` / `panda_rightfinger`（z≈0.868 m）。这与代码里 `ignored_contacts = [(6, 8)]` 的注释"panda_link7 与 panda_hand"互相印证：**link 6 = panda_link7、link 7 = panda_link8、link 8 = panda_hand**。两个固定关节：`panda_joint8`（child `panda_link8` = link 7）与 `panda_hand_joint`（child `panda_hand` = link 8）（`.conda312/Lib/site-packages/pybullet_data/franka_panda/panda.urdf:225-235`）。

**夹爪关节 = 9/10**，即 `panda_finger_joint1` / `panda_finger_joint2`（`src/simulation.py:52-54`；URDF 声明见 `.../panda.urdf:310-324`）。`manipulation.py` 里固化为 `FINGER_LINKS = (9, 10)`、`HAND_LINK = 8`（`src/manipulation.py:14-15`）。**易错点：在本 URDF 里"第 i 个 joint 的 child link 索引 = i"（base 为 −1），所以 joint 9/10 与 link 9/10 数值相同但含义不同**；代码在两处混用同一组数字（`src/manipulation.py:54` 用 joint 9/10，`src/manipulation.py:60-63` 用 link 9/10）。另注：`sim.joints = range(7)` 是 joint 索引 0–6，而对应 link 是 0–6 = `panda_link1..panda_link7`，**不包含 link 7/8**。

**关节限位来源**：运行时从 URDF 读 `getJointInfo(...)[8]`（lower）与 `[9]`（upper），只对 7 个手臂关节（`src/simulation.py:47-48`）。因此夹爪限位不在 `sim.lower/upper` 里，只在 URDF/`FINGER_MAX_WIDTH` 里。URDF 实际数值：

| joint | 名称 | lower | upper | effort | velocity |
| --- | --- | --- | --- | --- | --- |
| 0 | panda_joint1 | −2.9671 | 2.9671 | 87 | 2.175 |
| 1 | panda_joint2 | −1.8326 | 1.8326 | 87 | 2.175 |
| 2 | panda_joint3 | −2.9671 | 2.9671 | 87 | 2.175 |
| 3 | panda_joint4 | −3.1416 | 0.0 | 87 | 2.175 |
| 4 | panda_joint5 | −2.9671 | 2.9671 | 12 | 2.61 |
| 5 | panda_joint6 | −0.0873 | 3.8223 | 12 | 2.61 |
| 6 | panda_joint7 | −2.9671 | 2.9671 | 12 | 2.61 |
| 9 | panda_finger_joint1 | 0.0 | 0.04 | 20 | 0.2 |
| 10 | panda_finger_joint2 | 0.0 | 0.04 | 20 | 0.2 |

（`.conda312/Lib/site-packages/pybullet_data/franka_panda/panda.urdf:53,80,108,135,162,189,216,315,322`）

**HOME 构型**：`HOME = [0, -0.5, 0, -2.1, 0, 1.65, 0.78]`（`src/simulation.py:9`）。

**自碰撞处理**：`ignored_contacts = [(6, 8)]`，即 `panda_link7` 与 `panda_hand` 在动力学层关闭碰撞对（`src/simulation.py:41-44`）；失败判定里再排除固定邻接对（`src/simulation.py:100-106`，`getJointInfo(...)[16]` 是 parentIndex，不是名字）。

---

## 2. 夹爪：关节数、驱动方式、标量语义、单位

**关节数 = 2**，都是 prismatic（`panda_finger_joint1/2`，joint 9/10），每指行程 **0–0.04 m**、effort **20 N**、velocity **0.2 m/s**（`.../panda.urdf:315,322`）。

**驱动方式**：对两指**分别**下同一个 `targetPosition`（不是靠 URDF mimic 自动跟随）：

```python
for joint in FINGER_LINKS:            # (9, 10)
    p.setJointMotorControl2(sim.robot, joint, p.POSITION_CONTROL,
                            targetPosition=width, force=force, maxVelocity=max_velocity)
```
`src/manipulation.py:43-50`

URDF 里虽然写了 `<mimic joint="panda_finger_joint1"/>`（`.../panda.urdf:323`），但 PyBullet 不走 mimic，代码显式给两指发同一条命令，所以两指位移对称、总开口 = 2×width。

**"gripper" 标量的含义 = 单指位移（每根手指相对手掌的位移），不是总开口、不是 `panda_finger_joint1` 的"宽度"**。同一数量在四处出现，语义一致：

| 位置 | 语义 | 单位 | 依据 |
| --- | --- | --- | --- |
| `action[7]` / `gripper.pos` | 单指目标位移（命令） | m | `src/vla/config.py:24`、`src/vla/atomic_tasks/config2.py:49` |
| `observation.state[7:9]` | 两指**实际**位移 | m | `src/vla/config.py:22` |
| `set_gripper(width, ...)` | 两指共同目标位移 | m | `src/manipulation.py:43-50` |
| `.bullet` 快照 `width_m` / `grip_mm` | 单指位移 | m / mm | `results/experiment_3_controlled/initial_states/A_ref.json:10`、`scripts/exp3_controlled_states.py:16` |

**min/max 与裁剪**：`width = clip(width, 0.0, FINGER_MAX_WIDTH=0.04)`，力 `clip(force, 0, FINGER_MAX_FORCE=20)`（`src/manipulation.py:16-17,45-46`）；评测时再按 `FINGER_MAX_WIDTH` 裁一次（`scripts/eval_vla_policy.py:202`；实验 2/3 的脚本硬编码 `np.clip(action[7], 0.0, 0.04)`，`scripts/eval_multi_atomic_vla.py:171`、`scripts/run_experiment_2_4.py:369`）。

**默认/常用开度**：初始 0.035（`src/simulation.py:53`，= 70 mm 总开口）、张 open 0.04（`src/grasp_task.py:31`）、预闭合 0.024（`src/grasp_task.py:32`）、闭合 0.017（`src/grasp_task.py:33`）、C 放托盘 0.023（`src/vla/atomic_tasks/experts.py:263`）、B 推动 0.014（`src/vla/atomic_tasks/experts.py:24`）。**注意 `reset()` 里给两指的力是 30 N**（`src/simulation.py:54`），超过 URDF 的 20 N 上限；`set_gripper` 才是 20 N。

---

## 3. 相机定义与 RGB 获取路径

**两台相机，定义在 `CAMERAS` 字典里，顺序即采集顺序 `top` → `wrist`**（`src/vla/atomic_tasks/config2.py:14-36`；实验 1 版本见 `src/vla/config.py:27-51`）。

| 相机 | 类型 | 关键参数（D2） | 依据 |
| --- | --- | --- | --- |
| `top` | fixed（第三人称固定） | target=(0.55, 0.21, 0.02)、distance=1.05、yaw=50°、pitch=−52°、fov=50°、near=0.02、far=5.0 | `src/vla/atomic_tasks/config2.py:15-24` |
| `wrist` | wrist（挂在手掌上） | link=8（panda_hand）、offset=(0.09, 0.0, −0.01) m、aim=`grasp`、look_ahead=0.02、target_down=0.02、fov=75°、near=0.005、far=3.0 | `src/vla/atomic_tasks/config2.py:25-35` |

腕部相机视线目标 = 两指尖（link 9/10）中点 + 接近方向 ×0.02，再整体下移 0.02（`src/vla/cameras.py:70-77`），`up` 固定为世界 +Z。

**RGB 获取**：

```python
pixels = p.getCameraImage(width, height, view, proj,
                          renderer=p.ER_TINY_RENDERER,
                          flags=p.ER_NO_SEGMENTATION_MASK,
                          physicsClientId=sim.client)[2]
rgb = np.asarray(pixels, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
```
`src/vla/cameras.py:31-38`

- 渲染器：**TINY_RENDERER**，无窗口（`src/vla/cameras.py:33`）。
- 返回缓冲区是 RGBA，取前 3 通道 → **RGB**（PyBullet 该接口是 RGB 顺序，仓库里没有任何通道交换）。
- **BGR：not found**。全仓无 `BGR` / `cvtColor`；写视频时显式 `-pix_fmt rgb24`（`src/vla/lerobot_writer.py:85`），`_write_video` 用 `format='rgb24'`（`scripts/eval_vla_policy.py:139-140`）。
- dtype = `uint8`，像素范围 **0–255**，形状 **(H, W, 3) = (256, 256, 3)**。

**超采样路径**：`CameraRig(supersample)` → 渲染 `(256*ss, 256*ss)` → PIL **LANCZOS** 下采样回 256×256（无 PIL 时用步进切片）（`src/vla/cameras.py:17-26,29-38,44-50`）。默认 `supersample=2`（`src/vla/cameras.py:44`），但：

- **D2 采集用 1**：`scripts/collect_multi_atomic_data.py:64`、`scripts/collect_b_e23.py:95`；
- **D2 评测用 1**：`scripts/eval_multi_atomic_vla.py:245`（`--supersample` 默认 1）；
- **实验 1 评测脚本默认 2**：`scripts/eval_vla_policy.py:387-388`。

批量抓图接口：`CameraRig.capture() -> {name: uint8 (H,W,3)}`（`src/vla/cameras.py:92-97`）。

---

## 4. 送入模型前的 RGB 预处理

**仓库侧只有两步：`/255.0` 与 HWC→CHW，没有任何 resize/pad/mean-std**：

```python
x = torch.from_numpy(np.ascontiguousarray(frames[name])).to(self.device)
x = x.float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
batch[f'observation.images.{name}'] = x
```
`scripts/eval_vla_policy.py:77-80`（D2 评测同构：`scripts/eval_multi_atomic_vla.py:61-64`）

- 输入范围因此是 **[0,1]**；**[-1,1]：not found**；均值/方差归一化**不在仓库里**。
- resize / padding / 图像统计量在 **SmolVLA 自己的 preprocessor** 里：`configs/experiment_1/train.yaml:17-19` 记录基座默认 `resize_imgs_with_padding=(512,512)`（并列出 `chunk_size=50`、`n_action_steps=50` 等 checkpoint 默认参数）；`use_imagenet_stats` / `normalization_mapping` 只能从 checkpoint 的 `config.json` 读（`server/probe_ckpt.py:10-14`、`server/probe_policy.py:8-11` 打印这些字段）。**本仓库未记录 D2 checkpoint 的 `use_imagenet_stats` 实际值 → not found。**
- 数据集写入侧：图像以 **uint8、256×256、H.264/yuv420p、20 fps** 存成视频（`src/vla/lerobot_writer.py:41-57,311-322`），`add_episode` 强制 `uint8` 且形状必须 `(T,256,256,3)`（`src/vla/lerobot_writer.py:282-287`）。
- 数据集统计量（会被 lerobot 聚合成 Normalize 用的 stats）按 **/255 后**的像素算，归约到每通道 `(3,1,1)`（`src/vla/lerobot_writer.py:145-167`）。

**不一致点（迁移时要留意）**：采集用 `supersample=1`（第 3 节），而 `scripts/eval_vla_policy.py` 默认 2 —— 图像锐度/抗锯齿不同；D2 评测脚本已改回 1。

---

## 5. D2 本体状态（9 维）构造

**顺序与单位（数据集里写死的特征名）**：

| index | 名称 | 含义 | 单位 | 依据 |
| --- | --- | --- | --- | --- |
| 0–6 | `panda_joint1.pos` … `panda_joint7.pos` | 7 个手臂关节**实际**角 | rad | `src/vla/config.py:22`、`src/vla/atomic_tasks/config2.py:48` |
| 7 | `finger1.pos` | 左指实际位移 | m（0–0.04） | 同上 |
| 8 | `finger2.pos` | 右指实际位移 | m（0–0.04） | 同上 |

落盘特征定义可直接核对：`data/lerobot_panda_pick_place/meta/info.json:22-32`（`observation.state`，shape `[9]`，names 即上表）。

**构造点（采集与评测完全同一行代码）**：

```python
state = np.concatenate([sim.q(), np.asarray(finger_widths(sim))])
```

- 采集：`src/vla/atomic_tasks/recording.py:37-38`（实验 1 版：`src/vla/expert.py:47`）
- D2 评测：`scripts/eval_multi_atomic_vla.py:165`；实验 2.4：`scripts/run_experiment_2_4.py:363`；实验 2.1/2.2：`scripts/collect_handoff_snapshots.py:113`、`scripts/eval_handoff_timing.py:132`
- `sim.q()` = 7 个关节位置（`src/simulation.py:66-67`）；`finger_widths()` = 两指 `getJointStates` 的位置（`src/manipulation.py:53-55`）

**状态里没有**：关节速度、力矩、EE 位姿、物体位姿、接触力（`src/vla/diagnostics.py:4` 明确写"只会进入离线分析，绝不进策略输入"）。**EE 位姿 / 物体真值：not found（不在观测里）**。

**归一化在评测时的实际路径**：仓库侧不再做任何状态缩放，直接用 checkpoint 内保存的 `Normalize`/`Unnormalize` 统计量（`scripts/eval_vla_policy.py:58-60` 的注释 + `SmolVLAPolicy.from_pretrained`），`select_action(batch)` 内部完成（`scripts/eval_vla_policy.py:85-86`）。

统计量的产生：写入时按维度算 min/max/mean/std（`src/vla/lerobot_writer.py:324-326` → `_feature_stats`，`src/vla/lerobot_writer.py:133-142`），存 `meta/episodes_stats.jsonl`。

**具体数值样例（本地只有实验 1 的单任务数据集，D2 的 stats 在服务器 → D2 数值 not found）**，`data/lerobot_panda_pick_place/meta/episodes_stats.jsonl` 第 1 集：

| 维度 | state mean | state std |
| --- | --- | --- |
| q1 | 0.1110 | 0.1277 |
| q2 | 0.1335 | 0.2692 |
| q3 | 0.0493 | 0.0601 |
| q4 | −2.2996 | 0.1131 |
| q5 | −0.0165 | 0.0379 |
| q6 | 2.4384 | 0.3109 |
| q7 | 0.9600 | 0.2077 |
| finger1 | 0.0307 | 0.0095 |
| finger2 | 0.0319 | 0.0083 |

（同文件 action mean/std：q1 0.1126/0.1279 … q7 0.9623/0.2078，gripper 0.0301/0.0106；图像每通道 mean≈0.80/0.81/0.89、std≈0.165/0.158/0.163。）

---

## 6. D2 动作接口（8 维）与 chunk 执行

**8 维语义**：

| index | 名称 | 含义 | 单位 | 绝对值/增量 |
| --- | --- | --- | --- | --- |
| 0–6 | `panda_joint1.pos` … `panda_joint7.pos` | 7 个关节**绝对**位置目标 | rad | **绝对** |
| 7 | `gripper.pos` | 单指目标位移 | m | **绝对** |

依据：`src/vla/config.py:23-24`、`src/vla/atomic_tasks/config2.py:49`、`data/lerobot_panda_pick_place/meta/info.json:39-48`。

"绝对"有三个硬证据：① 专家录制时写的就是 `self._setpoint`（绝对关节角）而非差值（`src/vla/expert.py:48`、`src/vla/atomic_tasks/recording.py:39`）；② 评测直接把 `action[:7]` 当 position target 下发（`scripts/eval_multi_atomic_vla.py:173`）；③ 分析脚本把 `chunk[:,:7]` 直接喂 FK 得到 EE 轨迹（`scripts/exp29b2_policy.py:46-47`），只有绝对角才能这样用。

**夹爪通道的含义**：单指目标位移（与第 2 节同一标量）；`0.04` = 每指 40 mm（总开口 80 mm），`0.014–0.017` = 闭合抓取（总开口 28–34 mm）；不是"开/合二值"、不是力矩（`src/manipulation.py:43-50`）。

**chunk 长度 = 50**：

- `chunk_size=50, n_action_steps=50` 是基座/训练默认（`configs/experiment_1/train.yaml:17-19`；`scripts/eval_vla_policy.py:62-64` 注释"训练时按 50 步 chunk 优化"）；
- 分析脚本按 50 步遍历 chunk：`for k in range(50)`（`scripts/exp29b2_policy.py:52`）；
- 延迟基准假定每 chunk 只有 1 次真前向 + `n_action_steps-1` 次取队列（`server/bench_policy_latency.py:71-74`）；
- 实验记录里写死"n_action_steps=50 inferred"（`scripts/exp29b2_analyze.py:61`）。

**评测时如何消费 chunk**（`scripts/eval_vla_policy.py`）：

| 项 | 实现 | 依据 |
| --- | --- | --- |
| 每 tick 的调用 | 每控制周期调一次 `runner.act(frames, state)`，返回**一条** 8 维动作 | `scripts/eval_vla_policy.py:190-196` |
| 每条动作执行多久 | `for _ in range(STEPS_PER_ACTION=12): sim.step()` → 1 tick = 12 物理步 = **0.05 s @20 Hz** | `scripts/eval_vla_policy.py:210-211`、`src/vla/config.py:8-10` |
| 重新查询策略的时机 | 由 `policy.select_action` 内部队列决定：队列空则重跑一次前向，否则取缓存。仓库侧唯一的控制手段是 `--n-action-steps`（默认不传，沿用 checkpoint 配置：50） | `scripts/eval_vla_policy.py:61-64,402-403` |
| 关节目标裁剪 | `target = clip(action[:7], sim.lower+1e-4, sim.upper-1e-4)` | `scripts/eval_vla_policy.py:200-201` |
| 夹爪裁剪 | `width = clip(action[7], 0.0, 0.04)`，force=20 N、maxVel=0.05 | `scripts/eval_vla_policy.py:202,209` |
| 非有限值 | `np.nan_to_num` 兜底 | `scripts/eval_vla_policy.py:197-198` |
| 动作维度兜底 | 不足 8 维补零、超出截断 | `scripts/eval_vla_policy.py:89-92` |
| 提前结束 | `tick>40` 且物体回到初始高度+1 cm 内、`width>0.035`、线速度 < `rest_linear_speed=0.01` | `scripts/eval_vla_policy.py:228-231` |
| 收尾 | 停发新动作、物理继续 1.0 s（`settle_seconds`），严格模式要求策略自己 `release_cmd` | `scripts/eval_vla_policy.py:234-273` |
| 总预算 | `--max-ticks 260` → 260×12 = 3120 物理步 = **13.0 s** | `scripts/eval_vla_policy.py:381` |

**左余 chunk 的处理**：评测主循环里**不保留**——每次 `select_action` 自己管队列，脚本拿不到 chunk；只有实验 2.2 的交接时机对照显式读取 lerobot 的动作队列，并给出三种口径：H0 丢弃剩余 chunk、H1 执行完剩余 chunk 再切换、H2/H3 保持最后一条命令等待 0.6/1.2 s（`scripts/eval_handoff_timing.py:6-12,152-155,188-207`，读取 `runner.policy._queues[ACTION]`）。**"跨技能保留 chunk"在正式评测中：not found**。

**注意 `EVAL_EXEC_STEPS`**：`src/vla/config.py:14-16` 定义了 `EVAL_EXEC_STEPS=8`、`EVAL_EXEC_STEPS_RANGE=(5,10)`（"同一动作连续执行 5~10 个控制周期再观察"），但全仓无任何引用 → **该常量当前未生效，实际执行是 1 tick/动作**。

---

## 7. 关节位置控制器 `sim.command()`

```python
p.setJointMotorControlArray(self.robot, self.joints, p.POSITION_CONTROL,
                           targetPositions=list(q),
                           forces=[87, 87, 87, 87, 12, 12, 12],
                           positionGains=[0.2] * 7, physicsClientId=self.client)
```
`src/simulation.py:61-64`

| 项 | 值 | 依据 |
| --- | --- | --- |
| 控制模式 | `p.POSITION_CONTROL`（纯位置环，无速度/加速度前馈） | `src/simulation.py:62` |
| `forces` | `[87,87,87,87,12,12,12]` N·m（与 URDF effort 一致） | `src/simulation.py:63` |
| `positionGains` | `0.2`（7 轴统一） | `src/simulation.py:64` |
| `maxVelocity` | **未设置**（用 PyBullet 默认值） | `src/simulation.py:61-64` |
| 每条动作保持 | 12 个物理步 | `src/vla/config.py:8-10` |
| 策略频率 | **20 Hz** | `src/vla/config.py:9` |
| 物理频率 | **240 Hz**，`dt=1/240` | `src/vla/config.py:8`、`src/simulation.py:21,28` |
| 求解器 | `deterministicOverlappingPairs=1`、`numSolverIterations=100` | `src/simulation.py:29` |
| 重力/时间步 | −9.81 m/s²，`setTimeStep(1/240)` | `src/simulation.py:27-28` |
| 夹爪电机 | `setJointMotorControl2(... POSITION_CONTROL, force=20, maxVelocity=0.05~0.06)` | `src/manipulation.py:43-50`、`src/vla/atomic_tasks/experts.py:26-27` |
| 复位后稳定步数 | 120 步 | `src/simulation.py:58-59` |

**没有的东西**：力矩控制、阻抗/导纳、速度限制、动作平滑/插值（除专家离线插值 `joint_step=0.004`，`src/vla/expert.py:22-23,96`），以及**动作增量积分**（动作就是绝对目标）。π₀.₅ 若输出 delta 或 EE 增量，需要在适配层显式积分 + 限幅。

---

## 8. 正运动学与逆运动学接口

| 功能 | 函数 | 约定 | 依据 |
| --- | --- | --- | --- |
| FK / 当前 EE 位姿 | `Simulation.pose() -> (position(3) ndarray, quaternion(4) ndarray)` | 用 `getLinkState(robot, ee=11, computeForwardKinematics=True)` 的 `state[4]`（位置）与 `state[5]`（四元数），**四元数顺序 xyzw**（PyBullet 约定） | `src/simulation.py:69-71` |
| IK | `RobotController.solve(position, orientation, rest=None) -> (q, ok, wrapped)` | `p.calculateInverseKinematics(robot, ee, position, orientation, lowerLimits, upperLimits, jointRanges, restPoses, maxNumIterations=200, residualThreshold=1e-5)` | `src/robot_controller.py:42-51` |
| IK（起点生成器版） | `starts.ik_solve(sim, position, orientation, rest) -> (q, ok)` | `maxNumIterations=300`，不做 2π 折算 | `src/vla/atomic_tasks/starts.py:71-80` |
| `q` 读取 | `Simulation.q() -> (7,)` | `getJointStates(robot, joints=0..6)` 的 `[0]` | `src/simulation.py:66-67` |
| 尖端 FK | `finger_tips(sim) -> [pos_left, pos_right]`；`ee_position(sim)` | link 9/10 / link 11 | `src/manipulation.py:58-64`、`src/vla/atomic_tasks/starts.py:83-86` |
| 2π 等价折算 | `wrap_into_limits(q, lower, upper, rest, tol=1e-5)` | 只在关节角时用，不改构型 | `src/robot_controller.py:8-27` |

**orientation 约定**：全部是 `p.getQuaternionFromEuler([roll, pitch, yaw])` 生成的 **xyzw 四元数**，最常见是"朝下抓" `[pi, 0, 0]`（`src/grasp_task.py:24`、`src/vla/atomic_tasks/scene2.py:45`）与"朝下 + 绕 Z 转 yaw" `[pi, 0, yaw]`（`src/vla/atomic_tasks/scene2.py:48-49`、`src/two_state_task.py:76-77`）。IK 的 `position` 是**世界系 EE（link 11，panda_grasptarget）位置**，不是指尖、不是手掌。

**"到达"判据（三级）**：

1. 关节层：`err = ‖q_actual − q_target‖ < 0.06 rad`（`src/robot_controller.py:82-83`；专家侧同阈值 `src/vla/expert.py:129-130`）
2. 位姿层：`position_error < 0.025 m` **且** `orientation_error = 2·arccos(|dot(q_actual, q_target)|) < 0.15 rad`（`src/robot_controller.py:93-97`）
3. 运动安全：目标越限 → `joint_limit_target`；运动中越限 0.015 rad → `joint_limit_during_motion`；碰撞 → `collision_during_motion`（`src/robot_controller.py:57-58,75-79`）

**已知不一致（迁移要避开）**：`diagnostics.py` 用 **link 8（panda_hand）** 当 EE 位置算事件偏移（`src/vla/diagnostics.py:55-57`），与 `sim.ee=11` 差一个固定的手部偏移；只有 `grip_*`（两指尖中点）与 `sim.pose()` 是同一参考。若 π₀.₅ 侧要求"EE 位姿"作为本体输入，必须先明确用 11 还是 8。

---

## 9. 场景复位 / 物体状态 / 接触记录

**body id 一览（均在 `reset()`/`build_scene()` 内新建）**

| body | 创建处 | 说明 |
| --- | --- | --- |
| `sim.plane` | `src/simulation.py:30` | 地面 |
| `sim.robot` | `src/simulation.py:35-36` | panda，fixed base |
| `sim.cube` | `src/simulation.py:31-32` | 仅 `reference_cube=True`；VLA 场景为 −1（`src/vla/scenario.py:87`、`src/vla/atomic_tasks/scene2.py:96`） |
| `block` | `src/manipulation.py:20-35`（`create_block`） | 唯一可移动工件，返回 body id |
| `wall` | `src/vla/atomic_tasks/scene2.py:59-65` | B 的挡板，内表面 x=0.78（`src/vla/atomic_tasks/scene2.py:30`） |
| `tray`（2 个 body） | `src/vla/atomic_tasks/scene2.py:68-78` | C 的托盘导轨，内宽 0.080 m（`src/vla/atomic_tasks/scene2.py:39`） |

**允许接触集合**：场景构建后 `sim.allowed_contacts = {(block,8),(block,9),(block,10)}`（`src/vla/atomic_tasks/scene2.py:105`）；实验 1 场景只允许双指 `{(block,9),(block,10)}`（`src/vla/scenario.py:92`）。

**接触查询函数**

| 函数 | 查询 | 返回 | 依据 |
| --- | --- | --- | --- |
| `Simulation.collisions(allowed=None)` | `getContactPoints(bodyA=robot)` | 非预期接触 `[{body, link_a, link_b, penetration_m}]`，过滤桌面支撑、允许集合、固定邻接对 | `src/simulation.py:83-108` |
| `contact_points(sim, body)` | `getContactPoints(bodyA=body, bodyB=robot)` | `[{link, normal_force, position, normal, distance}]`，含零力接触 | `src/manipulation.py:86-98` |
| `finger_contact_forces(sim, body)` | 按 link 汇总法向力 | `(f1, f2, links_with_contact)` | `src/manipulation.py:101-108` |
| `non_finger_contacts(sim, body)` | 手/臂接触 | 列表，抓取时视为失败 | `src/manipulation.py:111-114` |
| `PushTask._wall_contact()` | `getContactPoints(bodyA=block, bodyB=wall)` | 是否触墙 | `src/successor_tasks.py:328-330` |

**记录了什么**

- 逐采样（每 4 步，`record_interval=4`）：`t, stage, ee_x/y/z, obj_x/y/z, obj_yaw_deg, obj_lin_speed, obj_ang_speed, tilt_deg, finger1, finger2, contact_force_1/2, contact_links`（`src/two_state_task.py:122-139`、`src/grasp_task.py:101-123`）。
- 逐接触采样（每 8 步，`contact_interval=8`）：`t, stage, link, normal_force, px/py/pz`（再版加物体局部坐标 `lx/ly/lz`）（`src/two_state_task.py:142-152`、`src/successor_tasks.py:90-102`）。
- 仿真轨迹（实验 2.4 的 240 Hz 记录）：关节 `q1..q7/dq/q_target`、`grip_actual/grip_target`、EE 位姿+速度、物体位姿/速度、`obj_tilt_deg/obj_yaw_rad/obj_wall_gap_m/obj_lift_m`、`left/right/palm/arm/table/wall` 接触标志与力（`scripts/run_experiment_2_4.py:221-280`）；逐物理步接触点 CSV（body/link/名称/点/法向/力/距离/摩擦）（`scripts/run_experiment_2_4.py:167-190`）。

**A90 的复位发生在哪**：`build_scene(seed, jitter=0.008, yaw=0.0)` → `Simulation(gui=False, reference_cube=False)` → `reset()`（重建世界、HOME、手指 0.035、`command(HOME)`、120 步稳定）→ 再 `create_block(...)` 放在 `START=(0.55, 0.0, 0.021)`（`src/vla/atomic_tasks/scene2.py:81-118`、`src/simulation.py:24-59`）。评测调用点：`scripts/eval_multi_atomic_vla.py:79-83,144-149`。

**B90 的复位发生在哪（两条路）**：

1. **独立原子起点**：`build_scene(seed, start=PLACE=(0.55,0.18,0.021), jitter=0.012, yaw=base±0.10)`，B90 时 `base=π/2`（`src/vla/atomic_tasks/scene2.py:26,99`；`scripts/eval_multi_atomic_vla.py:81-83`；`scripts/collect_b_e23.py:92-93` 的 spec：x=0.55±0.012、y=0.18±0.012，`scripts/collect_b_e23.py:51-78`）。
2. **从 `.bullet` 快照恢复**：先建同 seed 场景 → `p.restoreState(fileName=..., physicsClientId=...)` → `sim.command(sim.q())` + 夹爪按实际宽度重发 → 2~30 步稳定（`scripts/run_experiment_2_6.py:51-76`、`scripts/run_b90_snapshot.py:52-85`、`scripts/run_b90_controlled.py:21-64`）。

---

## 10. B90 指令文本与任务成功判据

**指令文本（唯一来源，`TASKS` 字典）**：

| 技能 | 精确字符串 | 依据 |
| --- | --- | --- |
| A0 | `pick up the orange block and place it in the target area, keeping its long side aligned with Y` | `src/vla/atomic_tasks/config2.py:39-40` |
| **A90** | `pick up the orange block and place it in the target area, with its long side aligned with X` | `src/vla/atomic_tasks/config2.py:41-42` |
| **B / B90** | `push the orange block along the positive X direction to the wall` | `src/vla/atomic_tasks/config2.py:43` |
| C | `pick up the orange block and place it into the tray, with its long side aligned with Y` | `src/vla/atomic_tasks/config2.py:44-45` |
| （实验 1 单任务） | `pick up the orange block and place it on the target position` | `src/vla/config.py:18` |

**"B90" 没有独立指令**：B90 = 同一 `TASKS['B']` 文本 + yaw≈90° 的物体/起点条件（`scripts/exp210_local.py:9` 记为 `'task':'B90_push_yaw90','language':'push ... positive X'`；调用处 `scripts/run_experiment_2_4.py:480`、`scripts/run_b90_snapshot.py:70-71`）。**独立的 "B90" 指令字符串：not found。**

**A90 成功判据（v2.2 口径）**：

```python
return (err <= 0.02 and yaw_ok and st['tilt'] <= 12
        and not finger_contacts(sim, block) and (max_z - initial_z) >= 0.03
        and st['speed'] < 0.01)
```
`scripts/eval_handoff_timing.py:74-81`（`err` = 物体 xy 到 `PLACE` 的 L2，`yaw_ok` = `axis_error_x(yaw) <= 12°`）

同口径的独立实现见 `scripts/collect_handoff_snapshots.py:88-89`（额外显式 `released`）；D2 原子评测 `scripts/eval_multi_atomic_vla.py:110-119`（A0/A90，`err<=0.02`、`axis_error_*<=12°`、`tilt<=12`、`released`、`rest<0.01`）。

**B / B90 成功判据（当前 v2.2，near-wall 阈值）**：

```python
def task_ok(skill, st, y0, yaw0):
    if skill == 'B':
        depth = BLOCK_HALF[0]*abs(cos(st['yaw'])) + BLOCK_HALF[1]*abs(sin(st['yaw']))
        gap = WALL_INNER_X - (st['position'][0] + depth)
        dyaw = abs(atan2(sin(st['yaw']-yaw0), cos(st['yaw']-yaw0)))
        return (0.0 <= gap <= 0.02 and abs(st['position'][1] - y0) <= 0.02
                and dyaw <= radians(15) and st['speed'] < 0.01)
```
`scripts/eval_handoff_timing.py:84-90`

```python
def final_ok(skill, final, released, stable, y0, yaw0):
    ... return (0.0 <= gap <= 0.02 and abs(final['position'][1] - y0) <= 0.02
                and dyaw <= radians(15) and final['tilt'] <= 12 and stable)
```
`scripts/eval_handoff_timing.py:98-105`

流程（v2.2 完整定义）：首次满足 `task_ok` → **停发新动作**（保持最后一条命令）→ 物理静置 **1.0 s** → 复核 `final_ok` + 连续 5 次 `speed < 0.01`（`scripts/eval_handoff_timing.py:241-260`；同口径注释见 `scripts/run_experiment_2_4.py:637` 的 `'success_judge': 'v2.2 (1.0 s settle, gap<=2cm, |dy|<=2cm, dyaw<=15deg, tilt<=12deg)'`）。

逐条件的完整版本（实验 2.4）：

```python
'cond_gap':     bool(0.0 <= gap <= 0.02),
'cond_lateral': bool(abs(dy) <= 0.02),
'cond_yaw':     bool(dyaw <= math.radians(15)),
'cond_speed':   bool(st['speed'] < 0.01),
'cond_tilt':    bool(st['tilt'] <= 12),
```
`scripts/run_experiment_2_4.py:518-527`；其中 `gap = WALL_INNER_X - (x + depth)`、`depth = 0.02·|cos yaw| + 0.05·|sin yaw|`（`scripts/run_experiment_2_4.py:513-517`，`WALL_INNER_X=0.78`、`BLOCK_HALF=(0.02,0.05,0.02)`，`src/vla/atomic_tasks/scene2.py:20,30`）。

**判据漂移（重要）**：D2 原子评测里 B 用的是 `abs(gap) <= 0.02`（允许负 gap，即允许压进墙里 2 cm），`scripts/eval_multi_atomic_vla.py:120-128`；而 B90 相关实验统一用 `0.0 <= gap <= 0.02`。**两份判据不等价**，迁移时以 v2.2（`0.0 <= gap`）为准。

**已知判据与物理的一致性限制**：`v2.2 成功 ≠ 真实推墙成功`，wall contact 只在 A1/B2/C2 出现且全部失败（`results/experiment_3_5/final_stage_report.md:19,35,96-97`）。也就是说 B90 的"成功"主要是"前缘间隙进入 0–2 cm"，不是"确实压到墙"。

---

## 11. 实验 3 的状态快照 / 审计工具（可复用清单）

**保存**

| 工具 | 做的事 | 依据 |
| --- | --- | --- |
| `p.saveBullet(path)` | 保存 PyBullet 完整世界状态 | `scripts/collect_handoff_snapshots.py:159-160`、`scripts/eval_handoff_timing.py:154-155`、`scripts/exp29_prepare_states.py:93-94`、`scripts/exp3_controlled_states.py:32`、`scripts/exp3_a90_terminals.py:15`、`scripts/reconstruct_b_midstates.py:78-79` |
| 状态 JSON（交接快照） | `skill/seed/success/ticks/max_z/lift_m/release_tick` + `robot{q,qd,ee_position,ee_quaternion,finger_positions,finger_velocities,limit_margin_*}` + `object{position,quaternion,linear_velocity,angular_velocity,yaw_rad,tilt_deg}` + `finger_contacts/table_contacts` + `policy_state(9)` + `last_action(8)` + `task` + `next_skills` + `scene` + `bullet_file` + `top_png/wrist_png` | `scripts/collect_handoff_snapshots.py:62-76,145-180` |
| 状态 JSON（实验 3 受控起点，极简） | `{seed, bullet, group:"yaw90", tag, state_id, kind:"B_reference"/"B_place"/"transition", source, obj[3], width_m}` | `results/experiment_3_controlled/initial_states/A_ref.json:1-11`、`results/experiment_3_controlled/initial_states/C_transition.json:1-11`、`scripts/exp3_controlled_states.py:33` |
| 重建中间状态 | 用数据集里记录的**绝对动作** replay 到第 t 帧再 `saveBullet`，并报 replay 误差（关节 ≤5.3e-6 rad、夹爪 ≤1e-6 m） | `scripts/reconstruct_b_midstates.py:35-92`、`docs/experiment_2_6_midstate_validation.md:7` |
| 起点生成（Low/High B90） | 采 5+5 个合法起点，逐宽度 14/25/35/40 mm 验合法性后存 `.bullet` + CSV | `scripts/exp29_prepare_states.py:39-118` |
| 状态清单 | 把快照目录汇总成 `state_manifest.csv`（C3/B90 两张表 + 视频清单） | `server/make_b90_manifests.py:6-49` |
| 中间状态 JSON 批量生成 | 从 `reconstruction_report.json` 生成 `ep{EE}_f{FF}.json`（source/seed/episode/frame/group/mode/bullet/q_error_rad/gripper_error_m） | `server/make_b90_states.py:6-20` |

**恢复**

| 工具 | 做的事 | 依据 |
| --- | --- | --- |
| `p.restoreState(fileName=..., physicsClientId=...)` | 恢复世界状态（需先建同 seed 场景） | `scripts/verify_snapshot_restore.py:80`、`scripts/eval_handoff_timing.py:180`、`scripts/run_b90_controlled.py:31,58`、`scripts/exp3_controlled_states.py:27,38`、`scripts/exp31_audit.py:25`、`scripts/analyze_b90_extension.py:71` |
| 恢复一致性审计 | 逐快照比对 `q / 指宽 / 物体位置 / 四元数 / 接触数 / 两路图像 MAE`，判据 `dq<1e-4, dfing<1e-3, dobj<1e-3, dquat<1e-3, dcontacts=0, image_mae<2.0`，输出 `restore_check.json` | `scripts/verify_snapshot_restore.py:84-131`（结果：40/40 通过，`docs/experiment_2_1_snapshots.md:32-45`） |
| 快照恢复审计（C3） | 输出 `state_restore_audit.csv`（q_error_rad / gripper_error_m / obj_error_mm） | `server/audit_c3_restore.py:10-29` |
| 首推理观测保存 | `initial_state.json`（q/dq/ee/ee_quat/obj_*/seed/group/mode/snapshot/q_target/grip_target）+ `initial_observation.npz`（top/wrist）+ 观测哈希 | `scripts/run_b90_controlled.py:70-77`、`scripts/exp3_controlled_states.py:41-49`、`results/experiment_3_controlled/data/observation_hashes*.csv` |

**已知限制**：`restoreState` 不恢复接触求解器的 warm-start 历史，恢复瞬间位姿/接触/图像一致，但长时间 rollout 可能与原轨迹分叉（`docs/experiment_2_1_snapshots.md:49`）。

**`.bullet` 文件位置（仓内已有）**：`results/experiment_3_controlled/initial_states/{A_ref,B_place,C_transition}.bullet`（与同名 `.json`、`_images.npz` 配对）。

---

## 迁移兼容性表（D2 侧已知项）

| Component | Existing D2 | π₀.₅ Required | Gap | Action |
| --- | --- | --- | --- | --- |
| Image input | 2 路 RGB uint8 256×256（PyBullet TINY_RENDERER，`supersample=1`）；仓库只做 `/255` + HWC→CHW（`src/vla/cameras.py:29-38`、`scripts/eval_multi_atomic_vla.py:61-64`） | TBD | TBD | TBD |
| Camera order | 固定顺序 `top` → `wrist`，键名 `observation.images.top` / `.wrist`（`src/vla/atomic_tasks/config2.py:14-36`、`data/lerobot_panda_pick_place/meta/info.json:50-95`） | TBD | TBD | TBD |
| Proprioception | 9 维：`q1..q7`(rad) + `finger1/finger2`(m)；无速度/力矩/EE 位姿/物体真值（`src/vla/config.py:22`、`scripts/eval_multi_atomic_vla.py:165`） | TBD | TBD | TBD |
| Language | 每 episode 一条英文指令，经 `task_index` 存 `meta/tasks.jsonl`；A90/B 文本见第 10 节（`src/vla/atomic_tasks/config2.py:38-46`、`src/vla/lerobot_writer.py:240-246`） | TBD | TBD | TBD |
| Action representation | 8 维：7 个**绝对**关节角目标(rad) + 单指目标位移(m)（`src/vla/config.py:24`、`scripts/eval_multi_atomic_vla.py:169-174`） | TBD | TBD | TBD |
| Action horizon | chunk=50；`n_action_steps` 默认 50（可覆盖）；每 tick 消费 1 条、执行 12 物理步（`scripts/eval_vla_policy.py:61-64,210-211`、`scripts/exp29b2_policy.py:52`） | TBD | TBD | TBD |
| Coordinate frame | 世界系 +X 横向 / +Y 搬运 / +Z 上；动作为关节空间绝对角，EE 参考 link 11（`src/vla/atomic_tasks/scene2.py:3-10`、`src/simulation.py:46`） | TBD | TBD | TBD |
| Gripper convention | 单指位移 0–0.04 m，两指同目标，总开口 = 2×；0.04=全开、0.014–0.017=闭合（`src/manipulation.py:14-17,43-50`） | TBD | TBD | TBD |
| Normalization | checkpoint 内 `Normalize/Unnormalize`（用数据集 stats，逐维 mean/std）；图像 `/255` 后由模型 transform 处理（`scripts/eval_vla_policy.py:58-60`、`src/vla/lerobot_writer.py:324-326`） | TBD | TBD | TBD |
| Control frequency | 策略 20 Hz、物理 240 Hz、12 物理步/动作（=50 ms）（`src/vla/config.py:8-10`） | TBD | TBD | TBD |
| Controller interface | `sim.command()` → `POSITION_CONTROL`，forces `[87×4,12×3]`，`positionGains=0.2`，无速度/力矩环（`src/simulation.py:61-64`） | TBD | TBD | TBD |
