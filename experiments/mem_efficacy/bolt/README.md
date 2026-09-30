# BOLT-Sync

Belief–Obligation–Ledger–Telemetry with Clock Synchronization.

实现文档：仓库根目录 [`BOLT.md`](../../../BOLT.md)。本目录是 Phase-1 确定性核心
（可关闭的 Phase-2/3/4 模块骨架已就位，默认 OFF）。

## 布局

```
bolt/
├── __init__.py
├── types.py       ActionContract / DecisionNeed / VerifyResult / 状态常量
├── claim.py       Claim + BeliefState（可失效）
├── clock.py       SyncState（cognition/action/evidence age + commitment_gap）
├── graph.py       不变量层：ObligationGraph（始终从 BDDL primitive_order 建图）
├── serve.py       指令策略层：每类义务的措辞阶梯（纯函数，不可门控）
├── ledger.py      VerifiedLedger（count:* 只从账本派生）
├── need.py        DecisionNeedCompiler（结构化缺口，不靠模型自省）
├── router.py      规则 EvidenceRouter + 证据饥饿地板
├── arbiter.py     纪律层：UnifiedArbiter（六层检查；拒绝时改写为 ACTIVE）
├── verify.py      IndependentVerifier（VLA 停止 ≠ 成功）
├── board.py       prompt board 渲染
├── harness.py     BoltHarness — S_t 控制器（§10 接口）
└── bind.py        规划器挂钩

旁路文件：
  ../memexp_bolt_bind.py     sitecustomize 入口
  ../arms/bolt.sh            臂定义
  ../selftest_bolt.py        无 GPU 自检（T1–T24）
```

## 三层架构（v6 重构）

v1 把三件不同的事混在同一批文件里，这是所有测量到的失败的共同根因。
现在它们严格分层，层与层之间只有单向数据流：

```
   graph.py       不变量层   —— 什么可以门控什么（可被证明、可被断言）
      │  唯一出口：active() / deps_ready() / serve_text()
      ▼
   serve.py       策略层     —— 对 VLA 说哪句话（纯函数，绝不门控）
      │  唯一出口：一句完整祈使句
      ▼
   arbiter.py     纪律层     —— 拒绝违反义务顺序的提案（永不生成命令）
```

### 不变量层（`graph.py`）

每次建图后 `_assert_invariants()` 断言以下四条，违反即写入
`coverage()['invariant_violations']` 并在自检中失败：

| 编号 | 不变量 | 修复的历史故障 |
|---|---|---|
| I1 | **闭合**：hard 边不得指向永远无法 SETTLE 的节点 | t8 v1 全零：硬门永不开 |
| I2 | **禁止门控**：可验证节点不得被不可验证节点门控（hard 或 soft） | t8 v5：Lift 完成后 ACTIVE 仍卡在 chocolate 备料上 |
| I3 | **可达**：每个计分 stage 必须由某个节点承载 | 对齐后漏掉的 stage 会被静默容忍 |
| I4 | **序数**：`pour_two` hard 依赖 `pour_one` | VLA 会在第一次倾倒之前要求"再倒一次" |
| I5 | **阶段可满足性**：每一档指令都必须能完成该节点被验证的阶段 | t8 v7/v8：抓取档只要求"抓"，而该义务由环境的 **Lift** 阶段计分 → 抓起但没抬 → 永不计分 → v8 的 seed 100/102/104 全部 0.0 |

I2 是最关键的一条。t8 里"把巧克力放进煎锅"是**不可验证**的备料，被接成了
pour 的一条 *soft* 边——看起来无害（soft 边一次服务即释放），实际效果是
Lift 结算后 pour_one 仍不 ready，`earliest_open()` 的层级回退到工具型节点，
于是 board 在 Lift 与 Pour_One 之间指挥机器人去摆巧克力，整集预算花在无分
动作上。现在不可验证的前置永不作为门控存在。

### 策略层（`serve.py`）

措辞阶梯是**纯函数**，且**循环**而非饱和。v1 把索引夹在 `len(ladder)-1`，
5 次指令之后同一句话永远回来；归档的 v5 t8 里，四个从未抬起瓶子的 trial
分别输出了 14/17、15/17、16/18、17/19 次**逐字节相同**的 `pick tomato sauce.`。
对照组 GPM 每步换一种说法，在同样的场景里成功抬起并拿到 46.7。

关键约束：
- 阶梯**循环**，且 `_pick_serve` 保证不连续返回同一句。
- 原始 BDDL 标签**不再是阶梯的一档**——它是图标识符（`pick tomato sauce.`），
  不是机器人指令。所有档位都是带落地名词短语的完整祈使句。
- 倾倒物质自动命名为容器（"tomato sauce bottle"），与对照组的措辞一致。

### 纪律层（`arbiter.py`）

- **绑定优先于解析**：`graph.recognize()` 先用"我们发过的指令"做精确绑定，
  再做 phase/object 模糊重解析。这一层之前缺失，导致**我们自己发出的轮换措辞
  反而绑定不上**（`Re-localize the …` 被重解析成 `close` 相位），落进"图外
  放行"分支，从而完全绕过依赖门与序数门。
- **禁止逐字重复**：与上一步刚发出的指令逐字相同的提案被拒绝并改写；
  任何其他合法措辞都接受（变化本身就是目标）。
- **运行期强制 I5**：任何**无法完成**当前义务所对应阶段的提案一律拒绝并改写回阶梯。
  这是对"标识符泄漏"的兜底：无论泄漏来自板子的哪一行、哪个来源，都不可能再
  作为一条必然失败的死指令进入环境。
- **序数兜底**：即使措辞未被绑定，只要文本明确请求"第二次倾倒"，
  仍会被绑定到对应序数节点，交由 I4 拒绝。

### 提示词卫生（跨层规则）

**内部标识符绝不进入动作通道。** 这条规则踩过两次坑，都是同一类：

| 泄漏源 | 表现 | 后果 |
|---|---|---|
| BDDL 标签（v5） | 阶梯末档是 `pick tomato sauce.` | 67.5% 输出逐字重复，饱和 |
| board 的 `OPEN` 行（v7/v8） | 打印 `[n3] grasp tomato sauce [OPEN]` | planner 抄成 `grasp tomato sauce bottle` → 纯抓取 → Lift 永不计分 → 0.0 |

现在 `OPEN` 行只命名**计分阶段**（`stage=01_Lift_Tomato_Sauce`），ledger 行改成
非祈使措辞（`grasp stages verified = 0`），并且有 `selftest` 的 T26 与全 26 任务
扫描两道网同时把守。

## 与 AOM / GPM 的关键差异

| | AOM | BOLT |
|---|---|---|
| 建图来源 | 门控（部分任务走 scoring） | **始终** execution partition |
| 合法性 | 图可达性 | 图 + Claim + 双时钟 +（可选）几何/安全 |
| 进度 | stage settle | **仅 VERIFIED** 才降 Φ |
| 计数 | DERIVED 义务 | ledger `count:*`（永不读 VLM 自述） |
| 缺口 | 模型/板子提示 | Decision Need 由状态编译 |
| 同步 | 无 | commitment–evidence window |

吸收了 AOM 的教训：grasp→place 用 **soft** 依赖（尝试过即可），pour 链用 **hard**。

v6 新增（针对 t8 输给 GPM 的根因）：措辞阶梯循环不饱和、BDDL 标签退出指令通道、
不可验证备料不再门控计分阶段、倾倒序数硬门控、绑定优先于解析。

## 快速自检

```bash
cd /project/peilab/why/RoboMemArena/experiments/mem_efficacy
PYTHONPATH=/project/peilab/why/RoboMemArena/evaluation_benchmark:\
/project/peilab/why/RoboMemArena/evaluation_benchmark/scripts:. \
  python selftest_bolt.py
```

## 跑一个单元（示例）

```bash
cd /project/peilab/why/RoboMemArena
ROOT="$PWD" source experiments/mem_efficacy/arms/bolt.sh
# 然后走既有 mem_efficacy runner（run_26x1.sbatch），ARM_OVERRIDE=bolt
```

## 环境旋钮

| 变量 | 默认 | 含义 |
|---|---|---|
| `MEMEXP_BOLT` | — | 总开关（臂文件置 1） |
| `MEMEXP_BOLT_GAP_MIN/MAX` | 0 / 30 | commitment_gap 支持窗口（秒） |
| `MEMEXP_BOLT_ROUTER` | 1 | 规则 Evidence Router |
| `MEMEXP_BOLT_FLOOR` | 1 | 证据饥饿地板 |
| `MEMEXP_BOLT_STARVE_LIMIT` | 4 | 连续无证据步数触发地板 |
| `MEMEXP_BOLT_STALL_LIMIT` | 2 | 同一义务被服务多少次后判定停滞（用于 board 的 `[RECOVER]` 标记；措辞阶梯无论此值都会循环） |
| `MEMEXP_BOLT_GEOMETRY` | 0 | Phase-3 几何层 |
| `MEMEXP_BOLT_SAFETY` | 0 | Phase-4 STL 安全层 |
| `MEMEXP_BOLT_VISION_VERIFY` | 0 | Phase-3 视觉验证 |
| `MEMEXP_BOLT_STM` | 0 | Phase-2 PRISM 短期 token（骨架） |
| `MEMEXP_BOLT_WORLD_MODEL` | 0 | Phase-4 世界模型（骨架） |

## 公共 API（BOLT.md §10）

```python
from bolt import BoltHarness

h = BoltHarness()
h.seed(exec_labels, scored_names, label_to_scored, name_objects=True)

state    = h.observe(obs, action_trace)
need     = h.compile_decision_need()
board    = h.context(need)
decision = h.arbitrate(proposal)       # proposal: str | ActionContract
result   = h.verify(action_trace, decision.contract,
                    verified_stages=[...])
h.commit(result)
print(h.report())
```
