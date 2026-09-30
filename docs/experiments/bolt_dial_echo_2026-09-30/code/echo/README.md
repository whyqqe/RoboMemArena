# ECHO — Evidence-conditioned Commitment and Horizon-aware Orchestration

独立实验臂：Planner + 冻结 VLA + Harness + Proactive Memory。默认保留 nomem 的真实
VLA 指令与 Harness 恢复路径，**不拦截/重写 primitive**；只在 Planner 通道按需提供由
先前观察保存的、决策相关的图像证据。

- `core.py`：episode-scoped 事件证据、执行承诺和验证账本。`DELIVERED` 不等于
  `VERIFIED`；在线 scorer 的 `stage_done` 绝不用于记功或改写当前决策。
- `bind.py`：延迟绑定 planner `_build_messages`、`reset_episode`，并观察
  `override_vla_prompt` 的实际返回值；运行中每个解释器写绑定收据。
- `selftest.py`：缺失 harness 的迟导入、planner 提供历史图像、观察器 identity、
  reset、不可信 scorer 不结算、证据失效测试。
- `../arms/echo.sh`：臂配置；`../run_echo_t8_1x10.sh`：t8 seed100 的严格早停入口。

第一版实际启用：从 Planner 可用的帧库预先记录候选证据；仅当连续 stall≥3 或
当前 Planner 子目标切换时，按当前决策索引读出最多两帧历史图像。正常执行时消息
字节序列沿用原 planner 路径，VLA 指令沿用原 Harness 返回值；不把 BDDL/stage
标签渲染成可复制的 VLA 命令。每个进程记录绑定、错误和实际发射次数。

**能力边界**：现有接口没有可信的在线物理验证器。`independent_verification` 仅接受
可信验证来源，但本实验臂尚未接入视觉/本体感受验证器；因此账本可以保持
`AMBIGUOUS`，不能宣称已实现可信计数、几何纠偏或基于已验证结果的在线重规划。
事件图像库只覆盖真实到达 Planner 的帧，不声称包含所有 VLA 步；下一阶段需要
独立的采帧接口和预注册的影子运行验证。此版本检验的是非破坏性前瞻记忆路径，
不是完整 ECHO 研究假说的证明。

**实验标准**：t8 1×10，h0，seed 100–109，严格早停（前三次全零或此后连续三次
零分）；0.0 早停不可作为 10 trial 的均值。GPM seed100 归档参考 46.7 分；
跨作业比较只能作探索性证据。必须查收据中两个模块已绑定、`n_plans>0`，
并核查 `n_saved` 与 `n_recalled`；否则不能解释分数。
