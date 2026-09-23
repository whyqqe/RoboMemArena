# Stage 0 — 旧结论 / 证据等级 / 能否复用

> 日期：2026-09-23。核清前不合并冲突数字；阶段 2 才做同版重跑。

| ID | 旧结论（原文大意） | 证据 | 等级 | 阶段 1 能否复用 |
|---|---|---|---|---|
| C1 | 密集关键帧 `memory_kf` 相对基线约 +8.2 pp | dual_track_report | 报告级；本地 Qwen planner | **否**作 API 记忆因果；仅作「记忆可能有用」背景 |
| C2 | 执行恢复 `harness_exec` 约 +8.6 pp | dual_track_report | 报告级 | **否**作记忆收益；说明 recovery 是混淆 |
| C3 | 组合 `harness_v21` 约 +13.4 pp | dual_track_report | 报告级 | **否**当无记忆底座；含 memory+recovery |
| C4 | Tier II 主实验约 +5.3 pp | REPRODUCIBILITY.md | 协议可复现，与 C3 不同来源 | **不得**与 C3 合并成同一结论 |
| C5 | 修复后 pullmem（595262）= 31.2 > push 28.8 > nomem 26.2 | 原始 aggregate + 机制表 | **原始可追溯**，但 pull 与对照**跨作业** | 机制（读帧率 0.20→0.88）可复用为「pull 通道修好了」；分数趋势须 **Hlegacy 同版三臂重跑** 后才能当复核 |
| C6 | pull 工具修复使枚举调用归零、观测帧/调用 ↑4.4× | 595132 vs 595262 census | **机制级证实** | **是**；作为继续完善 pull 的理由 |
| C7 |「无记忆底座」可用旧 harness_v21 / Harness VLA | 旧习惯 | **错误** | 明确拒绝；底座是 H0（VLM+VLA，recovery OFF） |
| C8 | VoLo 全任务基线可发表 | 部分任务污染 | 仅 20/23 干净 | 本阶段**不重跑**；不作主创新对照 |

## 阶段 1 主张边界（本目录允许写什么）

允许：

- 已建立 Hlegacy / H0 配置与预检；task5 smoke 能跑通
- pull 读取通道在机制上有效（引用 C6）
- 默认 Planner 改为 `gpt-6-luna`（成本），与 gemini-3.8-flash **分数不可直接比**，除非同作业对照

不允许：

- 用 595262 单臂分数宣称 pull 统计显著优于 push/nomem
- 把 dual_track 的 +13.4 写成 Proactive Memory 收益
- 在 H0 打开 retry / `MEM_STAGE_ANCHOR=1` 仍声称因果记忆实验
