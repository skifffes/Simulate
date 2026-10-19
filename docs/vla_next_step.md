# VLA 下一阶段可行性分析

核对日期：2026-10-19。仅检查官方资料，没有安装 RoboTwin、ROS 或 VLA 权重，没有真机实验。

## 官方证据和限制

### RoboTwin 2.0 与 Piper

官方仓库：https://github.com/RoboTwin-Platform/RoboTwin

官方文档：https://robotwin-platform.github.io/doc/

原始 README：https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/README.md

当前 main 为 2.0。README 2025-07-02 记录 Piper wrist bug 修复并要求重新下载 embodiment asset，证明官方确有 Piper 资产维护记录，但**不能据此确认所有任务、控制模式及检查点均已适配 Piper**。需核对当前资产下载可用性、机器人配置、双臂坐标、关节/夹爪命名、末端定义及控制接口。本阶段没有下载资产，资产实际可用性未验证。

当前 README 指向安装页 `https://robotwin-platform.github.io/doc/usage/robotwin-install.html`，本次该页抓取未取得可依赖的完整系统要求。因此不能确认原生 Windows 或 WSL2 是官方支持环境；不能把“WSL2 能运行某些 Linux 程序”写成 RoboTwin 官方兼容保证。后续优先独立 Ubuntu 环境，或先在 WSL2 做最小渲染/资产/控制冒烟测试；显式验证 WSLg、NVIDIA 驱动、CUDA、SAPIEN 和离屏渲染后再扩大安装。Python/CUDA/SAPIEN 版本必须按届时可访问的官方安装文档锁定，本阶段不猜版本。

README 2026-08-03 的当前评估体系通过 XPolicyLab，策略适配目录：
https://github.com/XPolicyLab/XPolicyLab/tree/main/policy

支持远程策略服务器与本地模拟器分离。不能把旧版 ACT/DP3 修复记录或 StarVLA 支持更新直接当成任意 VLA 检查点即插即用保证。默认动作配置 arx_x5 对应默认 aloha-agilex，并不等于 Piper 配置。应先选定当前官方策略目录内确有适配器和公开权重的组合，复现一个官方任务，再更换机器人。

### OpenPI

官方仓库：https://github.com/Physical-Intelligence/openpi

原始 README：https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/README.md

官方已测试 Ubuntu 22.04，并明确暂不支持其他操作系统；原生 Windows 和 WSL2 不应当作已验证的官方目标。

官方单 NVIDIA GPU 估算：推理 >8 GB，LoRA 微调 >22.5 GB，全量微调 >70 GB。还应为模拟器、图像和系统预留余量。本机约 8 GB 的 RTX 5060 Laptop **达不到官方 >8 GB 门槛，不能承诺本机完整推理**；16 GB 系统内存也是大型环境的风险。量化/卸载若使用需单独验证，不属于官方原始资源保证。优先远程 >=24 GB GPU 推理、本机仿真；首阶段不训练。

官方已有 ALOHA/DROID 等检查点，README 明确不同机器人适配具有实验性质，不能保证自有机器人成功。PyTorch 后端当前不支持 LoRA/FSDP 训练，不能不区分后端地引用 LoRA 资源。

## 推荐路线

1. 保留本项目作为底层执行与日志教学基线，先补多起点 IK 对照。
2. 在有足够 GPU 资源的 Ubuntu 环境选择一个官方已经复现的环境、机器人与策略检查点三元组；先复现公开的单任务评估。不优先同时更换机器人、任务和策略。
3. 如果目标必须是 Piper，先验证官方 Piper 资产与控制配置，再收集/转换少量匹配数据；不能把 ALOHA/DROID 权重直接用于 Piper 并宣称迁移成功。
4. RoboTwin 若暂无法验证 Windows/WSL2，可考虑 OpenPI 官方示例中的匹配环境（例如 LIBERO 路线）作为 VLA 学习起点；仍需检查该版本对应的 action adapter、权重和环境依赖，不能视为 Piper 模拟替代品。

安装难度：当前 PyBullet 低；RoboTwin 中高（机器人资产、渲染和策略服务版本匹配）；OpenPI 高（GPU、系统、模型与动作/数据适配）。没有从官方 README 确认 RoboTwin 的统一最低显存，因此不提供猜测的数字。

## 策略适配必须完成的检查

`Policy.act(observation)` 当前只是占位接口。真正接入时必须固定：

- 相机名称、分辨率、RGB 顺序、外参与帧率；本体状态、语言指令和历史窗口。
- 动作维度及关节顺序、rad/m 单位、位置或增量、世界/工具坐标系、旋转表示及夹爪开合定义。
- 检查点配套动作归一化统计与反归一化、action chunk 长度、控制频率及执行延迟。
- 通过已知输入测试输出范围；先在匹配任务离线回放，再仿真闭环，最后才讨论真机安全。

PyBullet Panda 的 7 关节与双 Piper、本体相机和训练任务均不匹配。通用 VLA 权重不是通用的 Panda 关节控制器。

## 原子技能和两技能组合评估

先评估 A/B/C 各自在标准起点的成功率；再固定控制预算，从真实 A 终态评估 A→B、A→C，记录独立的 A 成功、切换、后继成功、整链成功及恢复成本。与 Reset-to-Canonical、多起点 IK/规划器比较，避免把底层求解器局限解释为模型语义失败。

- Oracle 顺序：外部给定先做哪个技能、后做哪个，不评估任务分解。
- Oracle 切换：外部真值判据决定何时结束前一技能，不评估自主终止。
- 自动切换：策略或可观测判定器决定何时切换，需要单独报告误切、迟切和超时。

先 Oracle 顺序 + Oracle 切换隔离交接问题，然后保持任务/检查点不变，只替换切换方式。不要同时改变多个变量。

## 怎样论证“未见组合”

对自己可审计的数据：先定义技能词表和有序边，按完整组合/轨迹切分，再做去重、近重复搜索、指令同义表达检查；保留数据清单、哈希、生成种子、时间和训练配置。原子技能可在训练出现，测试组合不得进入训练/验证、模型选择或示范缓冲；测试前锁定评估与控制参数。

对无法完全审计的互联网预训练 VLA，**通常无法证明组合从未出现在其全部预训练数据中**。只能准确声明“在本次可审计的下游微调数据中留出的组合”，并披露预训练数据不透明和潜在污染。不能通过新指令措辞或随机测试场景自动获得严格的未见组合证明。

结论：本项目证明的是一个局部脚本控制器的起始状态依赖示例，没有得到交叉偏好，也没有提供 VLA 未见组合泛化证据。
