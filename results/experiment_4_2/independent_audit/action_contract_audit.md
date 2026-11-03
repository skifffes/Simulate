# 动作契约与时间索引

以下行号对应审计HEAD；本轮未修改生产链路。

| 层 | 实际路径/位置 | shape、单位与时序 |
|---|---|---|
| 专家 | `src/vla/atomic_tasks/experts.py` 66–178 | IK位置setpoint，7关节rad；夹爪每指m；目标可能跨周期持续不变 |
| 原始记录 | `src/vla/atomic_tasks/recording.py` 33–46、63–84 | 命令前记录state/RGB，state=(9,)是7关节+2指位置；action=(8,)是控制器setpoint，不是实际位移；然后调用原控制器推进 |
| 4.1导出 | `scripts/exp41/export_lerobot.py` 220–290 | state=(8,)为q和平均指位置/0.04；action=(8,)前7为target-q，夹爪/0.04；rad、无量纲 |
| 4.2缓存 | `scripts/exp42/export_data.py` | 从原始Parquet和对应视频导出9D状态、8D绝对目标及RGB NPY；每技能episode0/1/2的状态动作逐值一致，两路视频前40帧同索引逐像素一致，见raw_audit.json |
| window | `scripts/exp42/train_lora.py` 93–149 | stride4、H16；首行对应t而非t+1；`dq[k]=target[t+k]-q[t]`，夹爪连续归一化；尾部重复末帧 |
| transform | 同文件45–84、152–173 | DroidInputs→Normalize→model inputs；最终state32、actions16×32由配置/历史工件报告，安装版实现待复核；不是本地已运行shape测试 |
| FM | 同文件202–230 | 自写分维版本：x_t=t*noise+(1-t)*a，目标noise-a；与安装版官方compute_loss数值等价尚未独立验证 |
| sampling/反变换 | `scripts/exp42/serve_finetuned.py`、`src/exp42/registry.py` | 调create_trained_policy，必须核对实际OpenPI版本的Unnormalize及DroidOutputs；当前无本地安装，不能声称逐张量闭合 |
| adapter | `src/exp40/action_adapter.py` 106–162 | ±1rad接口检查；±0.25rad安全clip；关节裕度0.02rad；旧chunk递推，与4.2窗口固定锚点不符 |
| 电机 | `scripts/exp42/eval_rollouts.py` 88–144 | q_target送POSITION_CONTROL；g值错误当m下发；每控制周期12次240Hz物理step，20Hz控制 |
| 实际运动 | 同文件日志 | 缺少逐控制周期q/qd记录，不能判定为日志过期；实际物理q_next、EE和接触需专门测量 |

## 必须区分

`target[t]-q[t]` 是当前到控制setpoint误差，通常不等于 `q[t+1]-q[t]`。正确部署可以还原setpoint后使用同一位置控制器，不需要偷偷把标签换成真实位移。控制跟踪误差必须由原始帧验证，脚本的可选raw_first40输出包括这两项及其差。

对固定锚点窗口，正确目标是 `q[t]+dq[k]`；旧代码是 `q[t]+sum(dq[0:k+1])`（忽略clip）。合成测试专门选无clip、无限位触发轨迹以隔离该问题。安全限制遇到大锚点偏移时应单独审查，不能靠取消安全限制换取通过。

## 时间对齐结论边界

源码显示命令前拍摄/记录，并在原始控制器前hook。服务器实际读取每技能前三条原始Parquet、NPZ及两路视频：状态动作与缓存相等，两路视频前40帧与缓存同索引逐像素相等，原始timestamp前40帧亦归档。支持抽查范围内导出保留索引，但不能独立证明采集物理时序无偏差或所有帧无丢失。训练_valid_mask被收集但不进入loss；标准长episode窗口通常完整，短episode的重复尾帧风险另查。
