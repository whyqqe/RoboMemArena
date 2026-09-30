# 记忆机制谱系：问题背景、PMH、`nomem`/`pushmem`/`pullmem` 系列、GPM 与 AOM

- 日期：2026-09-29
- 范围：RoboMemArena `mem_efficacy` 这条研究线，从最朴素的 `nomem` 基线，到 PMH 的主动记忆、推送式、检索式、谓词式（GPM），再到图仲裁式（AOM）
- 目的：把**问题为什么难**、**每个设计各自怎么想的**、以及**实验里实际暴露了什么缺陷**讲清楚，不重复各子目录的逐次报告
- 配套材料：[`PMH.md`](PMH.md)、[`PMH_new.md`](PMH_new.md)、[`PMH_IMPL_587161.md`](PMH_IMPL_587161.md)、[`PMH_V2_GRAPH_EVIDENCE.md`](PMH_V2_GRAPH_EVIDENCE.md)、[`PMH_V3_ARCH.md`](PMH_V3_ARCH.md)、[`docs/experiments/mem_efficacy_pull_push_2026-09-18/REPORT.md`](docs/experiments/mem_efficacy_pull_push_2026-09-18/REPORT.md)、[`docs/experiments/evmem_gpm_hard4_2026-09-27/REPORT.md`](docs/experiments/evmem_gpm_hard4_2026-09-27/REPORT.md)、[`docs/experiments/aom_2026-09-28/REPORT.md`](docs/experiments/aom_2026-09-28/REPORT.md)、[`AOM_HANDOFF.md`](AOM_HANDOFF.md)

> **阅读顺序建议**：先读 §1–§2（背景与噪声底线），再读 §3–§6（五个设计，按时间顺序）。§7 是横向结论。**§2 的噪声数字必须先看**，否则后面每一张表都会被过度解读。

---

## 1. 问题背景

### 1.1 被测系统：一个异步快慢双系统

这条线上所有实验都在同一个系统上跑，只有**记忆**这一项是自变量：

| 组件 | 选择 | 是否冻结 |
|---|---|---|
| 评测协议 | RMA 官方，26 任务 | 是（上游） |
| Harness | RMA 的 HarnessVLA 式恢复层 | 是 |
| 慢系统（规划器） | `gemini-3.8-flash`，走外部 API | 是 |
| 快系统（底层） | `pi05_robomemarena`，来自 `checkpoints/PrediMem` | 是（从不微调） |
| **记忆** | **实验变量** | — |

结构上是**异步**的：VLA 以高频执行动作块，VLM 规划器以低频（`VLM_INTERVAL=5`）在另一个线程里决策下一个子目标。规划器不是每步都在场，因此**状态信息必须在规划器不在场时被保存下来，并在它回来时被交还给它**——这就是"记忆"这个变量的物理含义。

### 1.2 任务集：记忆被设计成"必需"的四个维度

26 个任务按 `memory_type` 分类，官方元数据（`evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json`）给出四类挑战：

| 标记 | 套件 | 记忆为什么必需 |
|---|---|---|
| `O` | Occlusion | 遮挡/顺序观察。抽屉只能逐个打开看一眼；**哪个是空的只有在全部看完之后才知道**，且信息在被使用前会再次被关掉 |
| `C` | Counting | 计数。需要记住"已经倒过几次"——动作本身在视觉上不可区分（`Pour_One` 与 `Pour_Two` 是同一动作） |
| `T` | Transferring | 搬运。三类物体按序转移，需记住**哪些已完成、下一个该搬哪个** |
| `S` | Sequence | 纯长程顺序，无遮挡 |

本研究线聚焦的 hard4 子集：

| 任务 | 套件 | 内容 | 评分阶段数 |
|---|---|---|---|
| **5** | `O` | 按序开关三个抽屉，记住**空**的那个，再开中间抽屉、把黄油放进去、关上 | 9 |
| **8** | `C` | 把番茄酱抬起来，**朝巧克力布丁上倒两次** | 3 |
| **19** | `T` | 把番茄酱、牛奶、橙汁从 cabinet1 搬到 cabinet2 | 3 |
| **22** | `S`+计数 | 把番茄酱抬起来，**朝饼干上倒两次**，再把饼干放进微波炉 | 3 |

这四类各自对应一种不同的记忆难题，这也是它们成为主要试验田的原因：t5 考"信息什么时候才存在"，t8/t22 考"动作次数不可从视觉区分"，t19 考"已完成集合"。

> 顺带一个容易被忽略的事实：**t8 与 t22 的评分阶段结构完全相同**（都是 `Lift → Pour_One → Pour_Two`），只差倾倒目标物（`chocolate_pudding_1` vs `cookies_1`），而 `nomem` 基线在 t8 上是 26.7、在 t22 上是 73.3。同一个信息结构，基线难度可以差 2.7 倍——**"任务名里的记忆类型"并不能预测模型实际会不会踩坑**。

### 1.3 评分与主指标

主指标是 `stage_score_pct`：每个任务把它自己的评分阶段列表跑完，得分 = 已完成阶段数 / 阶段总数 × 100，再对 10 个 trial 取宏平均。多任务时再对任务取宏平均。

这个指标的性质很重要：**它是离散的、带大台阶的**。3 阶段任务的取值只可能是 0 / 33.3 / 66.7 / 100，9 阶段任务则是 11.1 的整数倍。因此"提升 3.3 分"在 3 阶段任务上等于"多了一个 episode 完成了一整个阶段"，而不是连续量的 3.3%。

### 1.4 评测的受控约定

这条线上有一条比代码更重要的纪律：**任何两个被测臂，除声明的差异外必须逐字节一致**。具体做法是每个臂一个自包含文件，基线 `arms/nomem.sh` 开头**主动清空** `PMH_* / PACT_* / SEAM* / MEM_* / MEMEXP_*` 等变量，然后处理臂只改动它在 `MEMEXP_EXPECTED_DIFF` 里声明的那几个键，由 `validate_arm.py` 在花 GPU 之前强制核对。

这不是洁癖。已实测过一次真实事故：同一个作业里先跑 `pullmem` 再跑 `nomem`，因为 `MEMEXP_PULL_ENABLE` 不含 `MEM_` 前缀（`^MEM_` 匹配不到它）而残留为 1，**基线规划器被悄悄挂上了钩子**——分数上看不出来，只在进程级 census 里可见。这个洞现在是靠"清空 + 断言"两条一起堵的。

---

## 2. 噪声底线：先看这一节，再看任何表

所有单任务结果都是 **seed 100 / 1×10**（10 个 trial，覆盖 seed 100..109）。实测单个 trial 的 `stage_score_pct` 标准差约 **32 pp**（非饱和任务上 36.8 pp），因此 10 个 trial 的均值标准误约 **14 pp**。

这条约束推出三个必须是结论一部分的推论：

1. **所有已报告的跨臂差值都落在噪声内。** push/pull 系列的 +2.4 / +5.0 / +8.7 pp，GPM 的 +6.2 / +20.0 / +13.4 pp，AOM 的 ±16 pp——没有一个达到 2SE。
2. **要分辨 25 pp 需要约 13 trial/臂，分辨 15 pp 需要约 35 trial/臂**（按 SD=32 pp 估算）。当前是 10。
3. **同一配置跨作业复现的离散度已经很大。** `nomem` 在 task5 / seed100 / 10 trials 这个同一配置上给出过 **26.2**（job 595132）与 **13.8**（`task5_h0_controls`）两个数，相差约 12 pp。两次作业的代码有差异，所以这不是一次干净的重复实验；但它给出了一个量级概念：**12 pp 的差异可以完全由运行时环境产生**。更极端的例子来自 PMH 线：同一份代码、同一 seed，t4 的逐 attempt 分布测得 `[75,0,25,75]`（job 567876）与 `[25,25,25,0]`（job 567948），**单任务漂移实测 25–37.5 pp**。

因此这条线上的正确读法是：

> **机制指标（计数、比值、覆盖率）是可解释的强证据；分数差是弱证据。**
> 任何一次单任务的分数变化，只有在机制指标同向移动、并且改动是唯一变量时才值得归因。

这条纪律在 AOM 那一轮被直接用来推翻了一个已有的错误叙述（见 §6.3），而违反它的案例在这条线上一共出现过**三次**（见 §7.3）。

---

## 3. PMH：主动记忆 Harness —— 这条线的起点

PMH（Proactive Memory Harness）是这条线最早的完整设计，日期 2026-09-04。**后面每一个设计都是在回应 PMH 暴露出的失败**，所以它是理解整条线的钥匙。

### 3.1 核心思想：State / Evidence 分离

PMH 的一句话方案：

> 系统用短期 buffer 保持当前连续感知，把完成的事件整理成多层级长期记忆；VLM 始终看到一份简洁的当前任务状态，并在该状态不足以支持决策时，**主动**从文本摘要逐层下钻到关键帧和完整视觉证据，最后生成 subtask 交给冻结 VLA 执行。

它要解决的是这一代之前的通病：**"把所有记忆固定塞进上下文"**。因此整个框架建立在**一个区分**之上——把长期记忆分成两种不同角色：

| | **Task State** | **Episodic Evidence** |
|---|---|---|
| 回答的问题 | 系统当前**相信**什么 | 这些判断由**什么历史**支持 |
| 形态 | 纯文本、体量小 | 按 event segment 组织，含语义摘要 + 关键帧 + 完整视觉历史 |
| 更新 | 随新的 event segment 更新 | 随 segment 提交归档 |
| 注入 | **每轮持续注入**，保证基本任务连续性 | **默认不注入**，只有 VLM Agent 发现状态不足以支持决策时才主动检索 |
| 寿命 | 当前任务 | 整个 episode |

> **State 用于快速行动，Evidence 用于按需核实。**

这一区分同时也解释了"高层文本状态持续注入"为什么与"不要把历史塞满上下文"不矛盾：被持续注入的只是一份小型当前状态，真正占空间、含细节的 episodic memory 仍由 agent 自己访问。

**三层长期记忆**：

| 层 | 内容 | 作用 |
|---|---|---|
| 高层 **Task State** | 当前阶段、已完成/未完成 subtask、对象已被处理、位置/容器状态变化、未解决的失败或不确定性 | 记录 **current state / belief**，驱动执行 |
| 中层 **Segment Memory** | 每个 segment 一张"事件卡片"：ID + 时间范围、一句 caption、主要对象/动作/结果、状态变化、少量代表关键帧、指向低层的链接 | 既可用文本快速搜索，也能给少量视觉预览 |
| 低层 **Visual Archive** | 该 segment 的完整视觉证据（完整采样帧或短视频） | 默认不进上下文；用于查物体外观与身份、回看遮挡发生前的场景、比较操作前后状态、验证某动作是否真正完成 |

三层之间**必须保持指针**，这样文本摘要即使不完整或出错，Agent 仍能回到原始视觉证据重新判断：

```
Task State claim
      ↓ supported_by
Segment Card
      ↓ visual_pointer
Full Visual Evidence
```

**一个 Agent，两个 VLM 角色**（PMH 明确反对引入第二个 autonomous agent）：

```text
同一个 VLM backbone
    ├── Planning role：VLM Memory Agent，持续进行搜索与规划（唯一的主动决策者）
    └── Consolidation role：Segment Consolidator，按边界被调用一次，无状态
```

Segment Consolidator 在逻辑上是独立组件，但**不是第二个 agent**：它是 `commit_segment` 内部的一次**无状态** VLM 调用，只读已结束的局部 segment、输出结构化记忆后立即结束，不维护自己的 context / memory / action loop。这样避免让 Planner 在同一次输出里同时承担规划、分段、摘要、状态更新与关键帧选择。

### 3.2 写路径：记忆如何形成

```
连续观察 → Short-Term Buffer（近期 frames + 当前活动 segment）
              │  event boundary
              ▼
      VLM-backed Segment Consolidator（无状态、单次调用）
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
Task State  Segment    Visual
 (纯文本)     Memory    Archive
```

- **边界信号**：第一版不训练 event detector，用少量直观信号提候选——subtask 发生变化、一次操作成功或失败、操作对象发生变化、场景/物体状态明显变化。Harness 提候选，VLM Consolidator 判断前一事件是否语义完整，确认则提交。
- **`commit_segment` 由 Harness 触发**，而不是由主 Planner 在同一次规划输出里完成。这是刻意的职责分离。
- 每个 segment 代表一个相对完整的局部事件（"打开某抽屉并观察内部"、"抓取并放置一个物体"、"一次失败的抓取"、"任务阶段切换"）。

### 3.3 读路径：coarse-to-fine 主动下钻

每轮 VLM Agent **默认**只看到：原始任务 instruction、当前 observation、Short-Term Buffer 中的少量近期帧、持久化的高层 Task State。**中层 segment 与低层视觉历史不默认注入。**

如果这份输入不足以支持下一条 subtask，Agent 从高层到低层逐步查询：

```
Step 0：Task State 是否足够？              是 → 直接规划
    ▼ 否
Step 1：搜索 Segment Caption / State Transition   足够 → 规划
    ▼ 不足
Step 2：查看相关 Segment 的关键帧                足够 → 规划
    ▼ 仍不足
Step 3：展开该 Segment 的完整视觉证据
    ▼
基于证据规划，或明确保持不确定
```

判断标准刻意保持简单，不设计 entropy/uncertainty 模型，只要求 Agent 回答一个问题：**当前证据是否足以唯一支持下一步决策？**

**最小函数集**（只有三个）：

| 函数 | 谁调用 | 用途 |
|---|---|---|
| `commit_segment` | **Harness** 在候选边界处触发 | 把结束的 active segment 写入长期记忆：Task State 更新 + Segment Card + 关键帧 + 视觉档案指针 |
| `search_memory` | VLM Agent 主动 | 用自然语言问题搜索高层状态变化与中层 caption，只返回少量相关 segment 的文本与 ID，**不默认返回大量图像** |
| `inspect_segment` | VLM Agent 主动 | 展开已定位的 segment，只有两个 detail level：`keyframes` / `full_visual` |

### 3.4 明确不做的（这些"不做"后来变成了重要证据）

PMH 第一版明示不实现：**复杂 graph traversal**、object-level ROI search、独立 compare/merge/delete 工具、learned memory router、memory update 的训练策略、以及把所有层级 memory 同时塞进 Planner context。

理由写得很清楚：这些能力**只在三个基础工具无法覆盖真实失败案例时再增加**。

> 这条自我约束后来被证明是**正确的**：v2.0 的图/森林层被加进来之后测出来是**惰性的**，而 v3.0 的第一句结论就是"**初版的方向是错的**"——"图层是规范明确排除的东西，而它测出来是惰性的。初版却在给它加固。正确的第一步不是优化图层，而是回答：原始架构的四个模块，有几个真的在工作？"（见 §3.7）

**另一个核心创新表达**：PMH 把与 PACE / ACE 的区别定义为

```text
PACE：每轮固定读取 visual + semantic memory
ACE ：检测 stall / stage switch 后扩大固定读取量
PMH ：Agent 提出当前问题，定位相关 segment，再逐层展开证据
```

一句话：**现有方法预先决定 VLM 每轮能看到什么历史；PMH 让 VLM 先维护当前状态，再围绕具体决策主动定位 event，并从语义 gist 逐步下钻到视觉 evidence。**

整个创新位于 **inference-time harness**：不训练 memory VLA，不改变冻结 VLA 的动作能力，不训练 memory controller。

### 3.5 统一化版本：`PMH_new.md`（Belief / Claim / VOI）

`PMH_new.md` 把上面的框架重写成一套**基于 belief、证据和预测误差的主动记忆控制系统**：

```text
Memory  = Belief + Evidence + Validity Conditions
Harness = Belief-driven Information-and-Action Controller
```

三处实质推进：

- **记忆的基本单位从 Segment 变成 Claim**，Segment 只是轨迹索引。Claim 带来源、置信度、时间范围与**失效条件**：

  ```yaml
  claim: red_block is inside left_container
  evidence: [segment_17, frame_238]
  confidence: 0.86
  timestamp: after_segment_17
  scope: kitchen_table
  invalidation: [red_block_visible_on_table, container_moved]
  ```

  并要求状态可追溯，且**区分 Observed / Believed / Committed**。

- **读取决策变成一个期望值比较**（VOI）：Agent 先形成 Decision Need（目标、所需事实、缺失信息、风险、证据阈值），再在"检索历史 / 物理观察动作 / 任务动作"之间选择：

  ```text
  VOI(m) = decision improvement / (latency + risk + action cost)
  ```

  并按风险分级证据要求：低风险动作可依赖 State；中风险至少需要事件或关键帧；高风险必须由当前观测或视觉证据确认。

- **写入条件从"边界触发"变成事件触发**：`state_change OR prediction_error OR failure_relevance OR future_decision_value`。

- **执行侧变成结构化动作 + 独立验证**：VLM 输出带 `preconditions / predicted_transition / success_observation / abort_conditions / recovery` 的结构化动作；执行后**独立验证**预测的状态变化，再更新或撤销 Claims。核心纪律是 **"VLA 停止不等于成功"**，失败轨迹必须反哺记忆。

最小接口：`observe()` / `recall(decision_need)` / `act(structured_action)` / `verify(action_trace, expected_transition)`。

> 值得注意：`PMH_new.md` 是**设计文档，不是实测结果**。它描述的 VOI、Claim 失效条件、独立验证在归档作业里没有对应的实测读数，因此不能用它来支持任何"效果更好"的说法。

### 3.6 PMH 的实测结果

| 作业 | 范围 | 结果 |
|---|---|---|
| 567743 | v1.0 | `PG 61.5 / HM 81.2` → **−19.7 pp** |
| **587161** | **v2，26 任务 × seed100** | macro **69.11**，HM **70.20**，**−1.09 pp**（胜 6 / 负 6 / 平 14） |

**但这一线最重要的发现不是这个宏平均，而是一条剔除分析**：

> **PMH 的记忆层实际上是有效的。** 剔除 t22 后，PMH 在 **3 个 seed 上全部为正**（+8.2 / +5.8 / +1.8，均值 **+5.3 pp**，对 Harness+redaction）。而胜负被 **t22 一个任务**决定（PMH `0/3` seed，每次精确停在 `01_Lift_Tomato_Sauce`）。

这条发现的方法论意义大于它的数值：**一个 26 任务宏平均把"一个任务的归零"和"其余 25 个任务的稳定正收益"混在了一起**，差一点就把后者误判为不存在。同类现象在 AOM 那一轮以"t19 的整张表都在噪声带内"的形式再次出现。

**已被证实的正向杠杆**（作为对照，说明这条线上什么真的有效）：

| 杠杆 | 效应 | 机制 |
|---|---|---|
| **尝试放大**（`MAX_RETRIES=3`, `require_progress=0`） | **+9.4 (Harness) / +10.3 (PMH)** | `require_progress=1` 会在某次 attempt 未超过历史最好时**终止整条重试链**。t4 实录 `[0,0,25,50]`，旧策略记 0 而非 50 |
| **PMH 视觉证据通道**（在真缺口任务上） | **t4: +12.5，2/2 seed 完全一致** | stage 索引关键帧让 planner 找回"哪个抽屉是空的" |
| Retry Brief（上一轮进展的文本摘要） | 正 | 让重试不重复已完成的阶段 |

**已被证实的有害项**：

| 项 | 效应 | 机制 |
|---|---|---|
| **注入处方性 stage name** | **t4 −43.8, t5 −31.2**（方差为零） | 直接把答案键当指令喂进去 |
| **无条件检索** | **t22 −50, t11 −20** | 记忆无用任务上仍烧掉 53–70 次 `retrieve_visual`，挤占决策预算 |
| "加更多记忆内容"（v0.10–v0.14） | **净负**（5 轮复现） | 与已免费给出的答案重复，且增加上下文噪声 |
| 在遮蔽基础上再加 PMH 到 t5 | **−25 pp（87.5 → 62.5）** | PG 换来 46 张无用卡片 + 19 次强制注入的陈旧帧，净负 |

### 3.7 PMH 的缺陷

**（a）核心缺陷：把"意识到自己缺了什么"交给了模型，而模型几乎从不开口**

这是 PMH 最根本的结构问题，也是整条线后来所有设计的分水岭。

PMH 的读路径定义是"**agent 主动**判断证据是否足够，再逐层下钻"。实测普查（`PMH_V3_ARCH.md` 对 v2.0 线的普查）给出的读路径状态是：

| 模块 | 规范要求 | 实测 | 结论 |
|---|---|---|---|
| **VLM Memory Agent（读路径）** | `PMH.md` §5.2/§9.3：agent 主动判断证据是否足够，再从 gist 下钻到视觉 | `search_memory=0` · `retrieve_visual=19，forced=19 (100%)` · **`none` = 85%** | ❌ **不工作** |

三个数字合起来是一条完整的因果链：

- **`search_memory = 0`** —— 语义检索工具**一次都没被调用**；
- **`none` = 85%** —— 模型在绝大多数决策点上主动选择"不需要看"；
- **`retrieve_visual = 19` 且 `forced = 19（100%）`** —— 唯一真正发生的 19 次视觉检索，**全部是被 harness 强制触发的**，没有一次是 agent 自己要求的。

> 也就是说：这一层的**读接口基本处于静默状态**，而它唯一的产出完全来自一个绕过模型的兜底机制。

在另一次普查（job 587161，`PMH_IMPL_587161.md`）里同一个量测得 **`n_decide_none` = 119（55.9%）**，工具调用密度只有 87/213。**两个数字（85% 与 55.9%）来自不同作业、不同臂的普查，不应互相替代**；一致的是结论方向：**模型自己开口的比例远低于设计假设**。

这个缺陷的根源不在模型，在**设计**：它要求模型做一件它没有可靠能力做的事——**在自己还缺信息的时候意识到自己缺信息**。后续设计的应对方式都是**把判断变成结构性事实**（`pullmem` 推送"开放地址列表"、GPM 用 harness 自己观测到的 stall/attempts 行为信号门控、AOM 用图可达性），而不是继续请模型自省。

**（b）token 天花板事故（job 585835）：一个测量假象差点被写成一条定律**

这是整条线上最值得记住的一次事故，因为它同时是"启用 ≠ 使用"和"测量假象固化成结论"两个病的极端案例。

事件链：

1. `PMH_DECIDE_TOKENS` 设得过小 → **token 上限截断了全部 68 次决策**；
2. 每一次截断都让决策**退化成 `none`**；
3. **防闭眼机制**（`PMH_ANTI_COLLAPSE_NONE`，连续 6 次 `none` 就强制检索）把这个现象读成"agent 一直选择不看"；
4. 于是它开始**制造 agent 从未请求的检索**（`forced=True, reason=anti_collapse_none`）；
5. 而事后普查把这些强制检索读成"**读路径在工作**"。

`PMH_IMPL_587161.md` 对它的记录是：

> 这个假象一度被写进模型当成"**agent 自主生成查询的能力 ≈ 0**"的定律。实际原因是 **token 天花板**，planner 从未被允许选任何东西。

两个直接后果被写进了后来的代码纪律：

- `PMH_DECIDE_TOKENS` 设为 4096（与 `PLANNER_API_MAX_TOKENS` 一致，且代码以回退保证两者不会再分叉），此后**实测解析失败为 0**；
- **解析失败必须与"沉默"分开计数**。原先 `parse_pmh_decision` 把无法解析的响应对到与主动 `{"tool":"none"}` **同一个对象**上，下游无法区分两者——这正是 585835 被放大的原因。

**（c）94% 的阶段锚点在归档陈旧帧**

`n_stage_write_stale = 67/71`（**94%**）。阶段锚点仍在归档陈旧帧，而视觉银行把它当**第二优先级**来源。P1 修复（把候选池锚到 plan step 实际展示的帧）**没有解决它**。

**（d）没有查询条件化 —— 检索退化为"取最近一段"**

PMH 的证据包构造是：

```
out = preferred(最近 12 个 stage 的帧 ∩KF) + relevant(默认关闭) + kf(按帧号升序 = 最早的在前)
      → 截断到 8
```

即**按"最近 + 最早"填满固定的 8 格**。三个后果全部被实测证实：

| 后果 | 实测证据 |
|---|---|
| 无查询条件化，检索退化为"取最近一段" | `retrieve_visual` 的 `by_salience` 实现就是 `segments[-1]`；t22 上 attempts 1–3 的 8/9、8/8、6/6 次决策全是它，**零动作** |
| 冗余不抑制 | `preferred` 会取同一开抽屉事件的多个近重复帧（v1.0 gate 给过 `frames=[315,795,1000,1220]`） |
| 互补证据收不回 | 唯一的查询条件化通道 `PMH_RELEVANCE_RERANK` **默认值就是 `"0"`**，每一条日志都是 `relevant=0 cands=0` |

这是一条**"机制存在但被默认值关掉"**的缺陷：能力写好了，开关默认是关的，于是所有普查都显示它没在工作，而"没在工作"又被读成"这个机制没用"。

**（e）图层：规范明确排除、且实测惰性的东西，却被加固了一整份文档**

v2.0 的图/森林层（节点/边/效用/Kruskal 证据森林，参考 GraphMemix）在被加进来之后测出来是**惰性的**。而 `PMH.md` §6.4 明确写"暂时不需要：复杂 graph traversal"，§10 写"单 episode manipulation 历史规模更小，第一版只需要 Task State → Segment Card → Visual Archive 的**简单指针结构，不需要复杂图或树**"。

v3.0 的开篇自我纠错把这条教训写得比任何总结都好：

> **初版的方向是错的。** 初版把 v2.0 的图/森林层当作既定事实，然后花整份文档去修它。……**图层是规范明确排除的东西，而它测出来是惰性的。初版却在给它加固。**
> 正确的第一步不是优化图层，而是回答：**原始架构的四个模块，有几个真的在工作？**

**（f）`n_language_noop = 176（83%）`：语言档在空转**

恢复阶梯的语言档测的是"planner 有没有说**新**话"，而 t22 类失败里 planner 说的是**对的**——它没说新话，是因为它已经说对了，但物理执行没有推进。**指标测错了东西**：它把"没有进步"和"没有新表述"当成同一件事。

**（g）武装 ≠ 生效**

`n_hard_recovery_request = 134` 而 `n_tv_release = 0`：一个机制被**武装了 134 次但完全不可达**。普查必须把"武装"与"生效"分开读，否则会把一次都没生效的机制算成在工作。

**（h）检索层级未被测量**

`n_inspect` 与 `n_retrieve_visual` 由**同一个函数**自增，因此它们的相等是**代码产物**，不是"下钻阶梯成链"的证据。要判断 `keyframes → full_visual` 是否真的成链，需要新增按 `detail` 分级的计数器。

**（i）v1.0 的 Gap Gate：算得对，但触发 0 次**

v1.0 的核心机制是"只在控制器算出真实信息缺口时才放行证据"。实测**触发 0 次**（`PMH_OPTIMAL_ARCH.md` §1.7 推断 B）。同一份文档还记录了另外两处推断被自己的数据证伪、以及"封锁"分支被删除。教训与 (d) 相同：**一个机制触发 0 次，和它不存在是等价的**，而文档容易把它写成"设计正确但环境不配合"。

**（j）把单次读数当定论（这一线犯过两次）**

- **t22 的 −50 pp 曾被归因为"过度检索"**，并被据此设计了"封锁"分支。事后撤回：封锁/放行两种条件下 t22 **都是 0.0**，该诊断不成立；真正病因是 `_consecutive_same_subtask` 语义下的**软提交打断动作流**。
- **"封锁是 t4 的病根"只有方向性支持，未确立。** 同一份 v1.1 代码、同一 seed=100，t4 在 job 567876 测出 75.0，在 job 567948 测出 25.0，逐 attempt 分布 `[75,0,25,75]` vs `[25,25,25,0]`。**t4 的逐次尝试离散度（0–75）远大于待测效应**。上表把单次读数当作定论是错的。

这与 §6.3 里 AOM t19 v4 的错误是**同一类错误**，在一条线上重复了三次。

---

## 4. `nomem` / `pushmem` / `pullmem` 系列

这是 PMH 之后的第二代，问的是最朴素的问题：**"把历史观测交给规划器"这件事，哪种交付方式更有效？** 这一代的工作在 2026-09-18 打包为 `docs/experiments/mem_efficacy_pull_push_2026-09-18/`。

### 4.1 `nomem` —— 基线

**原理**：官方协议 + Harness 恢复层 + 固定规划器 + 冻结 VLA，**减去一切让历史观测抵达规划器的通道**。机制上就是两个开关关掉：

- `HARNESS_VLM_CONTEXT=0` → `HarnessController._refresh_vlm_context()` 提前返回空串，`"Harness memory context"` 提示块永不出现；
- `VLM_USE_KEYFRAME_MEMORY=0` → `J_hist` 停止累积，`"Historical keyframes from moments before the current step"` 提示块永不出现。

`N_RECENT / K_MAX / D_MERGE` 保持官方值，因此所有臂看到的**近期窗口**完全一致——那属于"当前观测"，不属于记忆。

**它的作用不是"最差的那个"，而是"参照坐标系"**：只有基线与处理臂只差记忆通道，才谈得上归因。

### 4.2 `pushmem` —— 推送式（HM 风格）

**原理**：在 `nomem` 之上，把 RMA **自己就会推送**的提示块打开。规划器不发起任何请求、不做任何检索决定：**由 Python 层决定每步往 prompt 里塞什么**。两个块都存在于官方 `api_vlm_planner._build_messages`：

```
"Harness memory context (read-time evidence; use with historical keyframes):"
"Historical keyframes from moments before the current step in the same execution (N):"
```

两个块都**不以规划器是否发出了什么为门控**——这就是"推送"的定义，也正是官方 README 描述机制的方式（"unlimited historical keyframes (`K_MAX=0`)"）。

**它相对官方协议必须先打的一个补丁**：官方把 `MEM_STAGE_ANCHOR` 留空（默认 0），此时图库是

```
K_indices_abs = build_visual_memory(J_hist, ...)
self.J_hist.append(j_abs)          # J_hist = 规划器自己提名的 keyframe_positions
```

官方本地 PrediMem VLM 被训练成会输出 `keyframe_positions`，而**通用 API 规划器不会**，于是 `J_hist` 永远是空列表、图库永远为空、**图像块在每一步都不发射**。代码里已经记录了这个状态（`api_vlm_planner.py:1489-1491`，以及 job 586700 的 `kf_n=0`）。

`MEM_STAGE_ANCHOR=1` 把图库构建切到官方的 `merge_keyframe_bank`，候选来自两个规划器无法压制的信号：**阶段边界**（评估循环在阶段推进时 pin）与**子任务变化**。因为 `bank_max=0` 不设上限，规划器提名依然会被并进去，所以这是官方行为的**严格超集**，不是替代。

> 这是 §3.7(a) 那个病的第二次出现，形态完全相同：**通道被打开了，不等于通道会发射**。一个依赖"模型自愿做某件事"的机制，在换成通用模型后静默失效，而**在分数上完全看不出来**——它只表现为"记忆没用"。

### 4.3 `pullmem` —— 小推送 + 规划器自己检索

**原理**：`pushmem` 必须在**事前**判断什么重要，而它只能通过代理信号（阶段边界、子任务变化）来判断，因此**只要相关性是在事后才确定的，它必然失败**。遮挡任务是纯粹的这一情形：三个抽屉各看一次，**哪个是目标取决于哪个是空的**——这个事实在三个都看完之前根本不存在。任何先验显著性规则都无法把第二次观测排在第一次之上，因为在那一刻两者同等重要。

`pullmem` 把 `pushmem` 混同的两个职责拆开：

- **获取**证据改为按需，由规划器选择——只有它知道此刻哪个地址重要；
- **保存**仍然由 harness 负责，所以小推送保留。

这一拆分同时解决上面那个"模型必须自愿"的问题：**推送使检索可被发现**（它列出存在哪些地址），而**打开哪个地址的决定权仍在规划器**。

**推送什么**（两端都设上限，且从不推答案）：

- 规划器自己发出的最近几个动作，带步号；
- **已被观测、现在可读**的地址，带帧数；
- **已被观测、但尚不可读**的地址，并显式标注为不可读。

推地址列表是把检索变得可用的关键：它把"意识到自己缺了什么"——PMH 把这一判断交给模型，而结果见 §3.7(a)（普查测得 `none` **85%**，另一作业测得 **55.9%**）——转化成了"读一个开放地址列表"，一个结构性事实。

**不推送什么**：关键帧库本身。`VLM_USE_KEYFRAME_MEMORY=0`，库被构建但绑定层把那些帧从 prompt 里剥掉，所以规划器除了实时窗口外看到的**只有它自己要来的图**。这让 census 的 `n_frames_served_off_context` 可以被解释为"收益"而不是"体量"。

**怎么问**：规划器输出严格 JSON 而不是 primitive：

```json
{"tool": "query_world", "address": "contents(middle drawer)"}
{"tool": "list_unbound"}
{"tool": "list_facts"}
```

`query_world` **一次调用同时返回记录和要看的那几帧**，并刻意不回答问题：规划器是 VLM，它靠读图来解析地址。这保持了写入路径不含模型，把感知判断放在它该在的地方。

**为什么是一次而不是两次**：PMH 的 `search_memory` 只返回文本、从不加载图像，所以需要**看**东西的规划器必须先 `search_memory`、从文本里读出 segment id、再 `inspect_segment(full_visual)`。那是**每个事实两次模型决策**，而在模型大概率选 `none` 的区间里，走完这个序列的概率是两个小数之积。这里压缩成一次决策、一次往返。

**两个注册表，以及为什么这不是细节**：`pullmem` 明确选择 `memexp_tools_search`：

| 注册表 | 工具 | 谁在用 |
|---|---|---|
| `memexp_tools` | `query_world`(地址) / `list_unbound` / `list_facts` | 默认无人用，保留作消融 |
| `memexp_tools_search` | 上述 **加上** `search_memory(query)`（自由文本） | `pullmem` |

**只支持地址的检索不是一个中性的默认值**：`query_world` 收的是一个**地址**——一段模型必须从推送块里原样抄出来的精确字符串——这**颠倒了检索的目的**。能写出 `contents(middle drawer)` 的模型通常已经能看见它指的东西；写不出来的模型则根本没有路径抵达那条记录。

### 4.4 `pullmem_er` —— 在 `pullmem` 上叠证据检索

**原理**：`pullmem` + Evidence Retrieval。

- 小推送里加 **OpenEvidence 块**：把地址标成 `UNKNOWN / PENDING / DETERMINED`；
- 规格文本告诉规划器**在存在缺口时先 `query_world`，再编造结论**；
- `query_world` 真正送出帧时记 commit 计数（serve→commit 账本）。

**刻意不加**：命令绑定器 / VLA 子任务改写（那是 `pullmem_alh`，在 task 1 上实测 CSR 为 0）；不新增第二个记忆库。

**为什么另开一个臂而不是改 `pullmem`**：让 `nomem / pushmem / pullmem` 三者保持逐字节可比。`pullmem_er` 相对 `pullmem` 的任何增益，才能归因于"缺口→检索→送达→提交"这条纪律，而不是对检索的静默改写。

### 4.5 结果

task5 / seed100 / 1×10：

| 臂 | `stage_score_pct` | 来源 |
|---|---:|---|
| **`pullmem`（工具重构后）** | **31.2** | job 595262 |
| `pushmem` | 28.8 | job 595132 |
| `nomem` | 26.2 | job 595132 |
| `pullmem`（工具重构前） | 22.5 | job 595132 |

配对差值：`pullmem(后) − pushmem` = **+2.4 pp**（重构前为 −6.3 pp）；`pullmem(后) − nomem` = **+5.0 pp**（重构前为 −3.7 pp）。

重构后顺序成为 `pullmem > pushmem > nomem`，这是这一轮**唯一符合设计预期**的顺序。

**但重点是机制证据，不是分数**——分子分母同时向预期方向移动：

| 指标 | 重构前 | 重构后 | 方向 |
|---|---:|---:|---|
| 工具调用总数 | 95 | **67** | ↓ |
| **枚举类调用占比** | **49%** | **0%** | ↓ |
| 送回上下文的观测帧 | 19 | **59** | ↑ 3.1× |
| **每次工具调用换到的帧数** | **0.20** | **0.88** | **↑ 4.4×** |
| `cap_hits` | 2 | **0** | ↓ |
| 已完成阶段数 | 20 | 27 | ↑ |
| 关键帧库中 `frame 1` 占比 | 44.8% | 2.6% | ↓ |

### 4.6 这一代发现的缺陷

**结构性缺陷（不是 bug，是设计层面的）**

- **推送必须事前判断相关性。** 只能用代理信号，因此在"相关性事后才确定"的任务类型（尤其是遮挡）上**必然**失败。这是 `pullmem` 存在的理由，也是 `pushmem` 无法通过调参解决的部分。
- **通道依赖模型的自愿行为。** `J_hist` 依赖规划器自愿提名 keyframe，通用 API 规划器不提，于是**图像通道开着但从不发射**（`kf_n=0`）。这是"启用 ≠ 使用"的又一次出现。
- **只支持地址的检索颠倒了检索目的**（见 §4.3）。
- **harness 的文本记忆通道是净损失。** 本地双轨研究实测：加入 `HARNESS_VLM_CONTEXT` 后增益减半、阳性结果从 5/5 降到 3/5。因此三臂对照**故意把它统一关掉**，让这一代实验只关于**视觉**通道。

**机制缺陷 D1–D5（前四个是机制，第五个是测量）**

| 编号 | 缺陷 | 后果（实测） | 修复 |
|---|---|---|---|
| **D1** | 关键帧库候选池太小：`frame_store_main` **只从成功抵达模型的规划器调用**写入，而 `VLM_INTERVAL=5` + `VLM_QUEUE_SIZE=1` 且队列满时**驱逐旧的**，env 循环远快于推理，绝大多数提交从未抵达 | ~2470 步的 episode 只写入约 **35 帧（1.4%）**，且偏向开局——一个前提是"记录过去"的通道几乎没有过去可记录 | 新增 `MEM_KF_STORE_INTERVAL`，在 env 循环里按固定间隔直接写帧，**把"记录"与"调用"解耦**（默认 0，官方路径不变）。修复后库中位数 10 → 197~227 |
| **D2** | `_kf_spread_union` 端点吸附：`older[int(k*step)]` 的 `k=0` 项恒为最旧帧，而官方 `merge_keyframe_bank` 已把 `frame 0` 选为锚点，所以这一格实际落在近似重复的 `frame 1` 上 | **8 个库位里有 1 个几乎每步都花在 episode 开局上**（`frame 1` 占比 70.1% / 44.8%）——正是该机制本应避免的组成 | 改为分层取样取**中点**而非端点；`frame 0` 是官方锚点不动。占比降到 2.6% |
| **D3** | 工具说明**招揽**枚举类工具：`list_facts` / `list_unbound` 是纯枚举，**按构造不可能返回观测帧**，而每步的系统推送块**已经把这批清单渲染进 prompt 了** | 95 次调用中 **49%（47 次）**是这两个工具，它们贡献了 19 帧中的 **0 帧** | 从工具说明里**移除推荐**（能力保留、仍注册、被调用仍应答）。占比降到 0% |
| **D4** | 说明承诺的预算 ≠ 循环实际执行的预算：说明写 "as many times as you need"（无限），实际 `MEMEXP_PULL_MAX_ROUNDS=3` | 模型被告知可以自由调用，然后被掐断，`cap_hits=2` | 让说明从**同一个变量**读取并写明真实预算（3→2）；新增闸门比对说明与 fuse 这两个独立产物 |
| **D5** | `search_memory` 的 miss 是死胡同：`_rank` 要求字面 token 重叠，模型用自己的措辞提问时只要措辞没字面出现就 miss；而 miss 分支回复的**又是一份清单，不是证据** | 25 次检索中 **12 次 miss（48%）**——该臂价值最高的路径有一半空手而归，且那一半往返还花在列表上 | 新增 `_recent_readable()` 回退：按时间倒序、并**过滤出真正有可读帧的记录**，让 miss 也变成可读证据 |

> **D5 的测量口径警告**：修复使 `n_search_miss` 结构性趋近 0。因此"检索命中率提好了"是**定义改变的结果，不是排序质量的改善**——`_rank` 的排序逻辑一行未改。这是"**修好了一个指标，而不是修好了一个能力**"的第一次出现；AOM 一轮里几乎重复了同样的错误（§6.3），而且代价更大。

**还有两个"提示在骗模型"的缺陷**（`pullmem_er` 侧，job 614210 / 614301）：

- `serve` 会从**帧窗口可用性**去提交任务簿记（`stage(...)`），把"能看见"当成"已完成"——直接违反 `SEEN ≠ VERIFIED`；
- 在存在限定兄弟（qualified sibling）时仍然广告未限定的 `contents(X)`；
- 系统提示在三处**硬编码"recent 5-frame window"**，而窗口大小是运行时参数（该 profile 里 `N_RECENT=7`）。模型被告知 keyframe 位置索引一个 5 帧窗口、同时被喂进 7 帧，结果在 job 614301 的 **76 个 plan step 里有 72 个**输出了空的 `keyframe_positions`。修复（`MEM_PROMPT_WINDOW_SYNC`）让提示里的帧数等于实际发送的帧数。

> 最后这条值得单独记住：**它不是模型失败，是 prompt 在说谎。** 当模型输出的字段长期为空，第一个要怀疑的是"我们有没有告诉它一个和现实一致的接口"。这是"启用 ≠ 使用"的第三次出现。

### 4.7 这一代**不能**声称的结论

必须明确，否则上表会被过度解读：

1. **不是配对设计。** `pushmem`/`nomem` 来自 595132，`pullmem` 来自 595262——不同作业、不同墙钟时段、不同 API 负载，跨作业比较有系统性差异风险。
2. **单任务单种子。** 不外推到其他任务或完整 26 任务协议。
3. **检索命中率的改善是定义性的**（见 D5 警告），不是排序质量改善。
4. 因此**不能**说 `pullmem` 显著优于 `pushmem`；**可以**说的是：机制指标变化幅度大、方向一致、与分数变化同向——这构成"改动确实改变了机制"的强证据，而**不是**"改动提升了分数"的统计证据。

---

## 5. GPM：`EvMem-GPM`（Goal-Predicate Memory）

打包于 `docs/experiments/evmem_gpm_hard4_2026-09-27/`。它是这条线上**目前分数最高的设计**（hard4 宏平均 41.7 vs `nomem` 31.8），也是 AOM 之前的实际主线。

### 5.1 原理：三通道，默认"流利"

**核心设计约束：默认行为 ≈ `nomem`（fluent）。** 记忆不是每步都注入，而是**在需要时**注入。这一点是刻意与 `pushmem` 相反的：`pushmem` 每步都推，`GPM` 默认什么都不推。

在 `nomem` 之上的增量分三路：

| 通道 | 作用 |
|---|---|
| **G** | 谓词脚手架：按 scorer 对齐的 NEED 谓词推进；对 identity / 容器"答案键"做 redact-safe 处理 |
| **E** | 门控证据：**仅在 stall / attempts / attractor 三种行为信号触发时**注入；失败帧打标签并随时间衰减；模板受 admissible 约束 |
| **C** | 硬原语控制器：拦截 off-stage、过早 pour、stage-label / ordinal 粘贴等非法动作 |

三通道的关键设计判断：

- **门控证据用的是行为信号，不是模型的自述。** `attempts / stall / attractor` 都是 harness 自己观测得到的，**不需要规划器意识到自己卡住了**——而"模型意识不到自己卡住"正是 §3.7(a) 那个 PMH 失败。这是 GPM 相对 PMH / `pullmem` 最重要的结构改进。
- **`SEEN ≠ VERIFIED`。** 看得见不等于做完了。任务阶段只能由 harness 的 scorer 判定完成，不能由"帧窗口里有相关画面"推断。
- **不写 notes，不打开 keyframe bank**（与 `nomem` 一致）。

实现见 `experiments/mem_efficacy/memexp_evmem.py` + `memexp_evmem_bind.py`，由 `arms/evmem.sh` 在 `nomem.sh` 之上启用（`MEMEXP_EVMEM=1`，`PMH_REDACT_STAGE=1`；`MAX_ROUNDS=2`、`LOOKAHEAD=4`、`EVIDENCE_CAP=6`、`GATE_ATTEMPTS=3`、`GATE_STALL=3`、`ATTRACTOR_REPEAT=2`）。

### 5.2 结果（seed100 / 1×10）

| 任务 | `nomem` | EvMem-GPM | Δ |
|---|---:|---:|---:|
| 5 | 13.8 | 20.0 | +6.2 |
| 8 | 26.7 | 46.7 | +20.0 |
| 19 | 13.3 | 26.7 | +13.4 |
| 22 | 73.3 | 73.3 | 0.0 |
| **宏平均** | **31.8** | **41.7** | **+9.9** |

**注：这些数字来自 job 615169/615189/615190/615191，与 push/pull 那一代不是同一批作业**，跨代比较同样有 §2 的噪声问题。

读数：

- **t8 抬升最大（+20）**：计数倾倒族（Lift → Pour_One → Pour_Two）上，C 门禁与 attractor 负反馈抑制了过早 pour 和简报吸引子漂移；
- **t19 / t5 中等抬升**，残留 lift stall 与物体定位失败；
- **t22 与 `nomem` 持平**（基线已经 73.3，处于饱和区间）。

### 5.3 GPM 的缺陷

**（a）`C` 通道本质上是一张正则表，不是理解**

`C` 拦截的是 off-stage、过早 pour、stage-label / ordinal 粘贴等，实现是模式匹配。它能挡住的正是它被写出来挡的那些，**换一个任务形态就需要新写一条规则**。AOM 的设计出发点就是替代它：把"哪个原语此刻合法"从一张表变成**可达性事实**。

**（b）完全没有 DERIVED 表示——计数问题在架构上没有位置**

这是 GPM 最根本的缺口。`Pour_One` 与 `Pour_Two` 在视觉上不可区分（同一动作、同一目标），所以"已经倒过几次了？"这个问题**必须有一个人工表示**。GPM 的三个通道里没有任何一个承担这个职责，直接后果是：

> `extra_pour_detected` 在**每一个归档的 task-8 臂上都是 0**。

也就是说：这个已知的、被明确编码进评分协议的失败模式，在这条线的**全部历史 run 里从未被计数过一次**。不是"很少发生"，是"没有表示，所以无法发生（在账本里）"。

AOM 的 `DERIVED` 模式与 `count:pick` / `count:place` 就是为补这个洞而设的，这也是 AOM 唯一一条被明确声称为**结构性新增**的能力（见 §6.1）。

**（c）`G` 通道的 redact-safe 规则被无差别继承，在搬运任务上变成错误**

`G` 的谓词脚手架对 identity / 容器"答案键"做 redact 处理，措辞是"**不要说出抽屉的身份**"。这条规则在**遮挡任务**里是对的：物体身份正是被 withheld 的答案。但它被当作全局策略继承下去之后，在**搬运任务（`T`）**上就反了——那里物体身份**就是指令本身**。AOM 继承这条规则时产生的实际输出是：

```
Place the object into the open container.
```

一个没有任何宾语、无法执行的命令。见 §6.3。

**（d）同代/前代的失败谱系**

同一条记忆线上还有更早的设计，它们的失败模式正好构成 PMH 与 GPM 之间的空白地带，值得作为"这个问题的求解空间长什么样"的证据：

| 设计 | 失败模式 |
|---|---|
| PMH | 读路径里 **`search_memory` = 0 次**；模型 85% / 55.9% 的时间选 `none`（把"意识到自己缺了什么"交给模型，是结构性错误，见 §3.7a） |
| PIC-MEM v1 | PIN flood（把历史钉得太多，淹没当前观测） |
| PIC-MEM v2 | unreachable operators（算子在该状态下不可达） |
| PIC-MEM v3 | evidence starvation：**4 个证据步/episode**（GPM 是 8–15），同 seed 下 **27.8** 而 GPM 是 **66.7** |
| PIC-MEM v4 | net-zero：叠在 GPM 上，净收益为零 |

PIC-MEM 四代连续失败的原因后来被诊断为：**它是把多个各自重新决定策略的账本/调度器/prompt 块拼在一起**，于是产生两个账本、两个调度器、三个 prompt 块。AOM 的"一条法、一块板"正是对这个诊断的直接回应。

---

## 6. AOM：`Arbitrated Obligation Memory`

打包于 `docs/experiments/aom_2026-09-28/`，交接文档在 [`AOM_HANDOFF.md`](AOM_HANDOFF.md)。AOM 是当前**机械上最干净**的实现（自检 56 项断言全过），但**分数不是最高的**。

### 6.1 原理：一种义务、一条仲裁法、一块板、可达性派生的合法性

**义务（obligation）**：每个任务阶段对应一条义务。三条轴：

| 模式 | 含义 | 何时了结 |
|---|---|---|
| `ACTUAL` | 由物理动作推进 | harness 自己的 `stage_done` 判定通过 |
| `PERCEPTUAL` | 由检视/读取推进 | `look` 返回了**新帧** |
| `DERIVED` | 在已了结集合上**计算**得出（`count:place`、`count:steps`） | 目标计数达成 |

外加正交的 `needs_look` 标志（"这个阶段在动手前想要视觉证据"）。

**仲裁法**返回 `advance / retrieve / act / derive / stagnant` 之一，按固定分支顺序：

```
if active is None or settled  -> advance     (并推动指针)
if force_retrieve             -> retrieve    (吸引子重复信号)
if needs_look and no evidence -> retrieve    (有界单次 look)
if not DERIVED and not sat_act-> act
if DERIVED                    -> derive
if not sat_ret                -> retrieve
else                          -> stagnant
```

两个刻意的设计点：

- **`stagnant` 排在 retrieve 之后**（安全地板 F）：一条在两个通道都饱和的义务仍然会检索，所以"宣布停滞"永远不会让证据通道少拿一帧。
- **合法性来自图可达性，不是模式表。** 规划器发出的 primitive 会被图门控检查；被拒时 `apply_control` 把它**改写成当前 ACTIVE 义务的第一条可采纳模板**，所以一次门控拒绝产生的是**纠正性命令**，而不是一个空操作。这是 AOM 相对 GPM 的 `C` 通道的核心区别。

**统一性声明**：AOM 声称 GPM 与 `pullmem_er` 的行为是**同一条法的两个退化点**：

- **GPM = 这条法在 `{ACTUAL, PERCEPTUAL}` × `{retrieve = harness 自动 look}` 上的取值**；
- **ER = 这条法去掉 `act` 分支、把 `retrieve` 交给模型**的取值。

所以"把 GPM 和 ER 拼起来"会得到两个账本、两个调度器、三个 prompt 块——正是 PIC-MEM 失败的原因。AOM 的回应是不拼接，而是**取两者的共同母集**。

**三个可单独消融的轴**：

| 旋钮 | 关掉之后在测什么 |
|---|---|
| `MEMEXP_AOM_DERIVED=0` | 移掉 DERIVED 模式。这是头条声明：没有它，"已经倒过两次了吗"**没有任何表示**（§5.3b） |
| `MEMEXP_AOM_GRAPH_GATE=0` | 只剩空/标签检查，即 GPM 的正则门。测"义务图相对模式表到底买到了什么" |
| `MEMEXP_AOM_FLOOR=0` | 移掉安全地板 F。默认**保持开启**，因为关掉会复现 PIC-MEM v3 的饥饿节奏（4 个证据步/episode vs GPM 的 8–15），同 seed 下 27.8 vs 66.7 |

**作者对 task 8 的诚实预期（写在臂文件里，值得原样保留）**：

> Task 8 的评分阶段全是 `ACTUAL`，因此 PERCEPTUAL 与 DERIVED **都不会被触发**。所以 AOM 在 t8 上应当**落在 GPM（46.7）附近而不是之上**。有信息量的输出是**计数器**（`n_stagnant`、`n_dep_rejects`、`derived_values`），它们把四种今天都渲染成"分数 0"的失败模式区分成四种。**在 n=10 上比 GPM 高 20pp 以上，应当被当作"法之外的现象"去调查，而不是去报告。**

这条自我约束后来被实验证实是对的：t8 上 AOM v6 = 50.0，GPM = 46.7，差 3.3（见 §6.2）。

### 6.2 结果（seed100 / 1×10）

| 任务 | `nomem` | `pullmem_er` | EvMem | **EvMem-GPM** | AOM v2 | v3s | v4 | v5 | **v6** |
|---|---|---|---|---|---|---|---|---|---|
| **5** | 13.75 | – | – | 20.00 | **45.00** | – | – | – | 36.25 |
| **8** | 26.67 | – | – | 46.68 | 43.35 | – | – | – | **50.02** |
| **19** | 13.32 | 16.65 | 19.99 | **26.67** | 23.32 | 16.66 | 14.80* | 23.33 | 19.99 |
| **22** | 73.34 | – | – | **73.33** | – | – | – | – | 33.33* |

`*` = 被早停截断，**不是** 10 trial 均值。

**v6（当前代码）对 GPM，同种子：t5 +16.25 胜 · t8 +3.34 胜 · t19 −6.68 负 · t22 早停。**

### 6.3 AOM 的缺陷

**（a）核心缺陷：用 SCORING 划分建图，而执行需要的是 EXECUTION 划分**

这是整个 AOM 系列里**唯一结构性**的缺陷。Task 19 有两套长度不同的分解：

```
scoring   _task_specs(19)     = 3 步  Place_Tomato_Sauce / Place_Milk / Place_Orange_Juice_Cabinet2
execution primitive_order(19) = 6 步  pick_TS, place_TS, pick_milk, place_milk, pick_OJ, place_OJ
```

AOM 的 `_seed_obligations` 读的是 **scoring** 表，于是图里是三条 `place` 义务、**一条 grasp 义务都没有**。而板子的规则是"只服务 ACTIVE 义务"，所以**整个 episode 都在命令一个 `place`**——而 `place` 预设手里有东西。**命令的动作物理上不可满足。**

实测（归档的 `task_trace` 流）：

| 臂 | pick 提示数 | place 提示数 |
|---|---:|---:|
| `nomem` | 344 | 151 |
| AOM v2 | **15** | **615** |
| AOM v4 | **10** | **504** |

以及每个 episode 的第一条命令：

| 臂 | 首条发出的 primitive |
|---|---|
| `nomem` | `Reach toward and grasp the tomato sauce from cabinet1.` |
| AOM v2 | `Place the object into the open container.` |
| AOM v4 | `Place the tomato sauce into cabinet2.` |

**可推广的部分（这才是可迁移的发现）**：评分划分省略的执行阶段，**当它在所有评分阶段下游时是无害的，当它是前置条件时是致命的**：

| 任务 | 执行有而评分没有的阶段 | 位置 | 结果 |
|---|---|---|---|
| **5** | 无（两套划分重合） | — | AOM 45.0 vs nomem 13.8 |
| **8** | `place` | 在评分的 pour **之后** | AOM ≈ GPM（中性） |
| **19** | `pick` | 在每一个评分的 `place` **之前** | AOM 低于基线 |

用真实执行划分（`fullvlm_v2_26_memory_tasks.json` 的 `primitive_order`）对全部 26 个任务实跑门控 `should_seed_execution`，结果是 **26 个任务里 16 个命中**：

```
ON  (16): 1, 2, 3, 11, 12, 13, 14, 17, 18, 19, 20, 21, 23, 24, 25, 26
OFF (10): 4, 5, 6, 7, 8, 9, 10, 15, 16, 22
```

（t4/t5 与评分划分重合；t6–t10、t15、t16、t22 的评分划分里含 `Lift_*`，即已经有一条 grasp 类阶段，所以门控判 OFF。）

也就是说：**t19 只是这 16 个任务中的一个**，这个缺陷在 26 个任务里有 16 个潜在存在，而不是 t19 的怪癖。

**修复**：从 `task_info.primitive_labels`（BDDL 执行划分）建图，并用 harness **自己的**桥 `harness.stage_mapper.expected_primitive_for_stage` 把每个执行步绑定到验证它的评分阶段——**这个桥本来就已经存在**（用于 stall 阶梯与 ledger guard），**是 AOM 一直绕开了它**。grasp 步通过 `settles_with` **派生式了结**（你倒了它，必然先拿了它），不需要新传感器。

顺带：在**搬运任务**上关掉继承来的容器/物体 redact（见 §5.3c）。

**（b）修复引入的 bug：抓放循环死锁**

v5（第一次跑 `seed_execution`）发出 **719 次抓取、0 次放置**。原因：`place#1` 要求 `pick#1` 是 `SETTLED`，但 `pick#1` **只在 `place#1` 了结时**才了结——一个环。于是 ACTIVE 指针整个 episode 都没离开 grasp，板子每步都在广告抓取。

**修复**：`Ledger::_dep_ready`——grasp→place 边在 grasp 被**尝试过**（`status != OPEN` 或 `tau_act > 0`）时即满足，而不是在了结时。普通顺序边（pour→pour、place→place）继续保持硬 `SETTLED` 要求。修复后 v6 在 t19 发出 122 抓 / 397 放。

**（c）两处记账错误（必须纠正，不要继续传播）**

1. **对照组用错了。** 早期笔记把**单种子**臂与**多种子宏平均**（nomem 23.3 / 25.6）比较。正确的 seed-100 对照组是 **13.32**，所以"**AOM 低于 `nomem`**"是**假的**：v2（23.32）与 v5（23.33）都在它之上。
2. **AOM t19 v4 的 "14.8" 是截断，不是测量。** 它以 `FUTILITY_MEAN_FLOOR_PCT=15` 在 k=9 被判停；而同种子对照 `hard3_t19_nomem_s100` 在 k=9 处的均值**也正好是 14.8**（最终 13.32）——门控是对**对照组自己的轨迹**开火的。因此这个系列**并不支持**"23.3 → 16.7 → 14.8，单调变差"这个叙述：那三个数里有两个来自门限不同的 run，其中一个还是截断。

> 这是"修好了一个指标而不是一个能力"（§4.6 D5）的第二次出现，而且这次代价更大：它把一个**运行错误**写成了一个**架构退化趋势**。

**（d）早期版本的其他缺陷**

| 版本 | 缺陷 |
|---|---|
| v2 | 自带 `place` **自锁**（`dep_rejects=47`）；错误的评分划分建图；物体 redact |
| v3s | 探测 EXECUTION 划分的实验版本，16.66 |
| v4 | 门限设置错误导致自我截断（见上） |
| v5 | 抓放环死锁（见上） |

### 6.4 AOM 这一轮**被数据证伪**的假设

这两条是这一轮最重要的产出，因为它们**否定了一整类后续优化方向**。

**假设一："primitive 配比驱动分数"——被证伪。**

工作假设是"规划器必须先抓再放"。数据拒绝把它当作分数的解释：t19 上**最强**的臂（GPM，26.67）抓取次数属于**最少**的一档（22，与 v2 的 15 基本同量级）；而抓取次数远超其他所有臂的 v5（**719** 次）得分 23.33。

**假设二："把图修对就能提升 t19"——被证伪。**

图现在机械上是正确的：`exec_labels` 是 6 步阶梯、`name_objects` 为真、`gate_rejects`/`dep_rejects`/`offgraph_rejects` **全为 0**、第一条命令是抓取——**而分数没有改善**（v6 19.99 vs GPM 26.67）。

### 6.5 仍然开放的：瓶颈在物理执行，不在文本通道

这是这一轮最有用、也最该被后续 Agent 接受的一件事：

- **每个臂都能到 33.3（三个评分阶段中的一个），没有任何一个臂到 100。** 臂与臂之间的全部分离度来自"有多少 episode 到了 66.7（三个中的两个）"。
- GPM 相对其对照是 **3/10 vs 0/10**——Fisher 精确检验 **p ≈ 0.21**。**整张 t19 表都在噪声带内。**
- 因此 t19 的剩余差距是**第二段搬运的物理执行方差**，一个**文本约束通道无法推动**的东西。

### 6.6 v6 的图 telemetry（可复核，也说明"门开在哪里"）

| 任务 | 执行建图 | `n_actual` | `n_needs_look` | `n_derived` | gate_rej | dep_rej | offgraph | unmatched | settled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | OFF | 9 | 8 | 3 | 5 | 21 | 0 | 15 | 27 |
| 8 | OFF | 3 | 0 | 2 | 0 | 0 | 0 | 2 | 13 |
| 19 | **ON** | 6 | 3 | 3 | **0** | **0** | **0** | 8 | 6 |
| 22 | OFF | 3 | 0 | 2 | 0 | 0 | 0 | 7 | 7 |

t19 是唯一一个执行划分建图的单元，也是唯一一个门计数**全为零**、同时 derived/pick 覆盖完整的单元。t5 残余的 `dep_rejects=21` 说明旧的自锁路径在那里仍被执行，但没有阻止它取得对 GPM 的 +16.25。

---

## 7. 横向总结

### 7.1 谱系表

| 设计 | 一句话原理 | 主结果 | 结构性缺陷 |
|---|---|---|---|
| **PMH** | **State/Evidence 分离**：小 Task State 持续注入，Episodic Evidence 默认不注入、由 agent 逐层下钻 | 587161：26 任务 macro 69.11 vs HM 70.20（−1.09）；**剔除 t22 后 3 seed 全正，+5.3 pp** | 把"意识到自己缺了什么"交给模型 → `search_memory=0`、`none` 85%/55.9%、19 次检索 **100% 强制** |
| `nomem` | 官方协议减去全部记忆通道 | t5 13.8 / t8 26.7 / t19 13.3 / t22 73.3 | —（它是参照系） |
| `pushmem` | 每步推送文本证据 + 历史关键帧 | t5 28.8 | **必须事前判断相关性**；依赖模型自愿提名 `J_hist` → 通道静默不发射 |
| `pullmem` | 小推送（列开放地址）+ 规划器自己检索 | t5 31.2 | 只支持地址的检索颠倒目的；枚举工具白吃往返；miss 是死胡同 |
| `pullmem_er` | `pullmem` + 缺口广告 + serve→commit | t19 16.7 | `serve` 用"能看见"提交"已完成"，违反 `SEEN ≠ VERIFIED` |
| **GPM** | 默认流利；谓词脚手架 + **按行为信号门控**的证据 + 正则硬门 | hard4 宏 41.7（+9.9） | 无 DERIVED 表示（计数问题无位置）；硬门是模式表；redact 规则在搬运任务上反向 |
| **AOM** | 一种义务 + 一条仲裁法 + 一块板；**合法性 = 图可达性** | t5 36.3 / t8 50.0 / t19 20.0 / t22 截断 | 用 SCORING 划分建图（前置条件缺失）；修复引入抓放死锁 |

链式关系值得单独指出：**PMH 证明"请模型自省"不可靠 → GPM 把门控信号换成 harness 自己观测的行为量 → AOM 进一步把"合法动作"从模式表换成图可达性。** 每一步都在减少对模型自觉性的依赖。

### 7.2 反复出现的失败模式

这条线上真正的知识不在于哪一版分数高，而在于**同一类错误换了个壳反复出现**。

**① "启用 ≠ 使用"——通道开了但从不发射。**

同一个病在**至少六处**出现过，而**每一次在分数上都看不出来**，只表现为"记忆没用"：

| # | 位置 | 表现 |
|---|---|---|
| 1 | PMH 读路径 | `search_memory = 0`；19 次视觉检索 **100% 是 harness 强制的** |
| 2 | PMH `RELEVANCE_RERANK` | 唯一的查询条件化通道，**默认值就是 `"0"`**，所有日志 `relevant=0 cands=0` |
| 3 | PMH v1.0 Gap Gate | **触发 0 次** |
| 4 | `pushmem` / `nomem` | `J_hist` 依赖模型自愿提名 → 图像块**每一步都不发射**（`kf_n=0`） |
| 5 | `pullmem_er` 系统提示 | 硬编码 5 帧窗口而实发 7 帧 → 76 步里 **72 步**输出空 `keyframe_positions` |
| 6 | `pullmem` 工具说明 | 枚举类工具占 **49%** 的调用、贡献 **0 帧** |

> **推论：任何"某个通道是否真的在工作"的问题，必须有独立于分数的计数器来回答。** PMH 线为此付出了最惨的一次代价（§3.7b：token 天花板把 68 次决策全变成 `none`，而普查把它读成"读路径在工作"，这个假象一度被写成一条"定律"）。AOM 的 `coverage`（`n_actual` / `n_needs_look` / `n_derived` / `n_stagnant_steps`）与 `gate_rejects`/`dep_rejects` 正是为此设计，它把四种都渲染成"分数 0"的失败模式区分成四种。

**② 把"意识到缺了什么"交给模型。**

PMH 的 85% / 55.9% `none` 是这条线的原初证据，`pullmem` 的臂文件把它总结为"**模型在 85% 的情况下得到 `none`**"。凡是设计里出现"模型应当注意到自己缺少某个信息"的，都失败了。有效的替代是**把判断变成结构性事实**：

- `pullmem` 推送"开放地址列表"，把"意识到缺了什么"变成"读一个列表"；
- GPM 用 harness 自己观测到的 `stall / attempts / attractor` 行为信号门控；
- AOM 用 obligation graph 的可达性决定"此刻哪个原语合法"。

**③ 答案键 vs 指令的错位。**

`redact`（"不要说出容器身份"）在**遮挡**任务里是对的——身份就是被 withheld 的答案；被无差别继承到**搬运**任务上就反了——那里身份**就是指令**。产物是 `Place the object into the open container.`——一个没有宾语的命令。**同一个措辞在不同 memory_type 上语义相反**，这是靠调参发现不了的。

**④ 分数不动的地方，可能是能力边界，也可能是物理边界。**

- t19 的剩余差距是**第二段搬运的物理执行方差**，文本通道推不动（§6.5）；
- 反过来，**机制指标动了而分数没动**（帧/调用 0.20 → 0.88）时，正确的结论是"我修改了机制"，不是"我改进了系统"。

**⑤ 一个任务可以吞掉整个宏平均。**

PMH 的记忆层在剔除 t22 后 3 个 seed 全正（+5.3 pp），但 26 任务宏平均只显示 −1.09。**宏平均会把"一个任务的归零"和"其余任务的稳定正收益"混在一起。**

### 7.3 三条必须遵守的测量纪律

**纪律一：单种子单元永远不能与多种子宏平均比较。** 违反它直接产生过一个**假结论**（"AOM 低于 `nomem`"，而正确的同种子对照下 AOM 是高于的）。

**纪律二：任何早停门限都必须用同种子的对照轨迹去校准。** 门限要严格低于对照组**在门控能看到的每一个前缀上的**最小滑动均值，否则它杀的是**对照组**。违反它产生过一个把**截断**写成**测量值**的例子（t19 v4 的 14.8），进而把一个运行错误写成了架构退化趋势。

**纪律三：单次读数不是定论，尤其在单任务漂移大于待测效应时。** 这条纪律是 PMH 线用两次误判换来的：

- **t22 的 −50 pp 曾被归因为"过度检索"**，据此设计了"封锁"分支；事后撤回——封锁/放行两种条件下 t22 **都是 0.0**，真正病因是软提交打断动作流；
- **"封锁是 t4 的病根"未确立**：同代码同 seed，t4 测得 75.0（567876）与 25.0（567948），逐 attempt 分布 `[75,0,25,75]` vs `[25,25,25,0]`。

同类错误在这条线上一共出现**三次**：PMH 的 t22 归因、PMH 的 t4 诊断、AOM 的 t19 v4 门限。

前两条纪律现在都被自动化：`selftest_futility.py` 从 `run_26x1.sbatch` **本身**提取门控 awk 程序（拷贝无法漂移），并回放归档单元验证门控不会误杀任何 `nomem` 对照；T10 用**runner 自己的参数**回放同种子对照，使那个错误门限无法再次被提交。

### 7.4 如果要继续做，证据支持的方向

1. **不要在 t19 上继续调 prompt。** 剩余差距在执行侧。
2. **先把 n 提上去再谈 t19。** 3 个种子（100/110/120）的 v6 + 对照是让 t19 那张表变得可解释的最便宜做法；三个种子的 `nomem` 对照已经存在（`hard3_t19_nomem_h0_3x10`）。
3. **离线诊断第二段搬运。** 读失败 episode 的回放/状态流，看是碰撞、掉落还是对准失败，而不是再加记忆。
4. **要一个可辩护的胜利，就用 t5 与 t8。** 两者都在门控 OFF（图与归档完全一致）的情况下击败 GPM，这是干净的陈述。
5. **`DERIVED` 是唯一一条结构性新增能力**，且它对应的 `extra_pour_detected` 在全部历史 run 里都是 0——这是这条线上**最有价值的未测量量**，值得专门下一次测量。
6. **PMH 的信念化方向（`PMH_new.md`）尚未被实测。** Claim 失效条件、VOI 读取决策、独立动作验证都是设计而非读数；如果要做，第一件事是给它们加计数器。

---

## 8. 文件与复现入口

| 内容 | 路径 |
|---|---|
| PMH 设计（起点 / 信念化版） | `PMH.md`、`PMH_new.md` |
| PMH 实现说明与缺陷普查 | `PMH_IMPL_587161.md`、`PMH_V2_GRAPH_EVIDENCE.md`、`PMH_V3_ARCH.md`、`PMH_OPTIMAL_ARCH.md` |
| PMH 实现 | `evaluation_benchmark/harness/pmh_memory.py`、`scripts/run_harness_variant.sh` |
| 基线与 push/pull 系列 | `experiments/mem_efficacy/arms/{nomem,pushmem,pullmem,pullmem_er}.sh` |
| push/pull 报告与代码快照 | `docs/experiments/mem_efficacy_pull_push_2026-09-18/` |
| GPM 实现 | `experiments/mem_efficacy/memexp_evmem.py`、`memexp_evmem_bind.py`、`arms/evmem.sh` |
| GPM 报告与打包 | `docs/experiments/evmem_gpm_hard4_2026-09-27/` |
| AOM 实现 | `experiments/mem_efficacy/memexp_aom.py`、`memexp_aom_bind.py`、`arms/evmem_aom.sh` |
| AOM 报告与打包 | `docs/experiments/aom_2026-09-28/` |
| AOM 交接说明 | `AOM_HANDOFF.md` |
| 评测循环 / 早停 | `evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py`、`experiments/mem_efficacy/run_26x1.sbatch` |
| 任务与阶段定义 | `evaluation_benchmark/scripts/task2_26_reference_stage.py`、`evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json`（执行划分在 `async_vlm26_det/fullvlm_v2_26_memory_tasks.json` 的 `primitive_order`） |
| 臂对齐预检 / 事后 census | `experiments/mem_efficacy/validate_arm.py`、`census_channels.py` |

```bash
cd /project/peilab/why/RoboMemArena

# 臂对齐预检（在花 GPU 之前）
ROOT="$PWD" .venv/bin/python experiments/mem_efficacy/validate_arm.py

# AOM 自检与早停门控自检（无 GPU）
cd experiments/mem_efficacy
PYTHONPATH=/project/peilab/why/RoboMemArena/evaluation_benchmark:. python selftest_aom.py
FUTILITY_WATCH_UNDER_TEST=3 FUTILITY_FLOOR_UNDER_TEST=8 FUTILITY_CONSEC_UNDER_TEST=3 \
  FUTILITY_TASK_UNDER_TEST=19 FUTILITY_SEED_UNDER_TEST=100 \
  PYTHONPATH=/project/peilab/why/RoboMemArena/evaluation_benchmark:. python selftest_futility.py
```

---

## 9. 一句话结论

这条线的核心发现不是"哪种记忆更好"，而是：

> **在这套异步快慢系统上，"把历史交给规划器"的收益一直被两件事限制——一是通道静默失效（启用了但从不发射），二是瓶颈其实在物理执行侧。PMH 最早证明了"请模型自己意识到缺什么"不可靠（85% / 55.9% 的时间它选择不看），此后每一个设计都在减少对模型自觉性的依赖；而真正可复现、可解释、可归因的进展，来自先把通道的"是否在工作"变成可计数的量，而不是来自更长的 prompt。**
