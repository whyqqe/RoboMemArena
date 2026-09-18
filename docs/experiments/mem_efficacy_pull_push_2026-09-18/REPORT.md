# 记忆机制有效性实验报告：`pullmem` / `pushmem` / `nomem`

- 日期：2026-09-18
- 作业：`595132`（三臂同批）、`595262`（`pullmem` 工具重构后重测）
- 结论一句话：本轮定位并修复了 5 处机制缺陷；修复后 `pullmem` 从同批最后一名变为第一名，且机制指标（每次工具调用换到的观测帧数）提升 4.4 倍。**分数层面的差异仍在噪声内，尚不能声称统计显著。**

## 本目录内容

| 路径 | 内容 |
|---|---|
| `REPORT.md` | 本报告 |
| `CODE.md` | 代码快照清单、改动 diff、新增闸门说明 |
| `code/` | 产出本报告全部结果的代码快照（29 个文件，与 job 595262 的 SHA-256 逐一相符） |
| `results/mem_efficacy_595132/` | 同批三臂：`nomem` / `pushmem` / `pullmem`（工具重构前） |
| `results/mem_efficacy_595262/` | `pullmem` 工具重构后重测 |

每个结果目录内含：`summary.json` / `summary.tsv` / `aggregate.json`（分数）、
`task5/ep*/attempt*/api_vlm_trace.jsonl`（每个 plan step 的 `K_indices_abs`、`n_store`、`out_text` 等）、
`memexp_pull_report.json.*`（工具调用与检索计数）、`harness_memory/`、`logs/`、
`videos/`（失败 episode 的主视角与腕部录像）、`code_provenance.json`（代码指纹）。


---

## 1. 实验设计与协议

| 项 | 值 |
|---|---|
| 任务 | `task5`（单一任务子集） |
| 臂 | `nomem`（基线）、`pushmem`（主动推送关键帧）、`pullmem`（Planner 主动检索） |
| `SEED` | 100 |
| `NUM_TRIALS` | 10（覆盖 seed 100..109，单次评估过程内完成） |
| 评分 | `stage_score_pct`（宏平均，10 个 trial 的均值） |
| 规划后端 | `gemini-3.8-flash`（API planner） |

三臂的设计差异（`MEMEXP_EXPECTED_DIFF` 声明并由预检强制）：

| 开关 | `nomem` | `pushmem` | `pullmem` |
|---|---|---|---|
| `VLM_USE_KEYFRAME_MEMORY` | 0 | 1 | 1 |
| `MEM_STAGE_ANCHOR` | 0 | 1 | 1 |
| `K_MAX` | 0 | 8 | 8 |
| `MEM_KF_SPREAD` | — | 8 | 8 |
| `MEM_KF_STORE_INTERVAL` | 0 | 5 | 5 |
| `MEMEXP_PULL_ENABLE` | 0 | 0 | 1 |
| `HARNESS_VLM_CONTEXT`（通道 A 文本） | 0 | 0 | 0 |

`HARNESS_VLM_CONTEXT=0` 是三臂共同的**受控常量**，不是缺陷：本地双轨研究测得文本通道对视觉通道是净损失（加入后增益减半、5/5 阳性降至 3/5）。三臂都关，因此该结论不构成本实验的混淆。

---

## 2. 本轮修复的 5 处缺陷

前 4 处是机制缺陷，第 5 处是测量缺陷。

### D1 —— 关键帧库的候选池太小（根因）

关键帧库的候选帧来自 `planner.frame_store_main`，而该 store 原先**只从成功抵达模型的 planner 调用的上下文窗口**写入。官方配置 `VLM_INTERVAL=5` + `VLM_QUEUE_SIZE=1`，且队列满时新提交**驱逐**旧的（`eval_fullvlm26_async_vlm_vla.py`：`queue.Full → get_nowait → put_nowait`），而 env 循环远快于推理规划器，因此绝大多数提交从未抵达模型。

后果：约 2470 步的 episode 只写入约 35 帧（1.4%），且偏向开局。一个前提是"记录过去"的通道几乎没有过去可记录。

**修复**：新增 `MEM_KF_STORE_INTERVAL`，在 env 循环中按固定间隔直接写帧，**把"记录"与"调用"解耦**。该开关默认 0（官方路径不变），仅两个处理臂置 5。

**证据**（库容量）：

| `MEM_KF_STORE_INTERVAL` | frame store 中位 | 最大 |
|---|---|---|
| 0（`nomem`，未启用） | 10 | 35 |
| 5（处理臂） | 197 ~ 227 | 515 ~ 525 |

### D2 —— `_kf_spread_union` 的端点吸附

`_kf_spread_union` 用等距取样把时间上分散的帧补进库，原实现为 `older[int(k * step)]`。其 `k=0` 项**恒等于 `older[0]`**，即候选池中**最旧**的帧。

而官方 `merge_keyframe_bank` 本身已把 `frame 0` 选为锚点（step-0 的 subtask 变化是显著事件），因此这一格实际落在 `frame 1` —— `frame 0` 的近似重复帧。实测（job 594675）：

| | `pushmem` | `pullmem` |
|---|---|---|
| `frame 1` 出现在库中的比例 | 70.1% | 44.8% |
| 其余低位帧（2..4） | 3~7% | 2~5% |

**8 个库位里有 1 个几乎每步都花在 episode 开局上**，正是这套机制本该避免的组成。

**修复**：改为分层取样，每段取**中点**而非端点 —— 保留原实现要的"均匀覆盖"性质，去掉端点黏附。`frame 0` 是官方锚点，**未改动**。

**证据**（`frame 1` 占比）：

| | 修前（594675） | 修后（595132） | 修后（595262） |
|---|---|---|---|
| `pullmem` | 44.8% | **2.6%** | 5.6% |
| `pushmem` | 70.1% | **4.1%** | — |

已加闸门 `CHECK 17: the spread picks NO frame from the episode's opening`，并**实测证伪**：退回旧写法立即 `[FAIL] early picks = [0]`。

### D3 —— 工具说明在招揽"枚举类"工具

`search_memory` / `query_world` 返回观测帧；`list_facts` / `list_unbound` 是**纯枚举** —— 返回地址清单与解析状态，**按构造不可能返回任何观测帧**。而每个 plan step 的系统推送块（`render_push`）**已经把这批清单渲染进 prompt**了。

因此调用 `list_facts` 等于花一次完整的模型往返，去要一份刚递给它的东西。实测（595132）：95 次工具调用中 49%（47 次）是这两个枚举工具，且它们贡献了 19 帧中的 **0 帧**。

**修复**：从工具说明中**移除这两个工具的推荐**。没有移除能力 —— 两者仍注册在册、被调用仍会应答 —— 移除的是"招揽"。

### D4 —— 说明承诺的预算 ≠ 循环实际执行的预算

工具说明写 "You may do this as many times as you need"（无限次），而 `MEMEXP_PULL_MAX_ROUNDS=3`。模型被告知可以自由调用，然后被掐断。实测 `cap_hits=2`。

**修复**：让说明从**同一个变量**读取并写明真实预算（`MEMEXP_PULL_MAX_ROUNDS` 由 3 调整为 2）；同时让说明与 fuse 成为两个独立可比的产物，新增闸门比对二者。

### D5 —— `search_memory` 的 miss 是死胡同

`_rank` 要求字面 token 重叠（`if not overlap: continue`），因此模型用自己的措辞提问时，只要措辞没字面出现在地址或值里就返回 miss；而 miss 分支回复 `render_addresses()` —— **又是一份清单，不是证据**。实测 25 次检索中 12 次 miss（48%），即该臂价值最高的路径有一半空手而归，且那一半往返还花在列表上。

**修复**：新增 `_recent_readable()` 回退 —— 按时间倒序、并**过滤出真正有可读帧的记录**，让 miss 也变成可读的证据。`how` 字段回显 `recency_fallback` 与 `search`，使模型与 census 都能区分，无需信任任何自由文本。

> **测量口径警告**：D5 使 `n_search_miss` 结构性趋近 0。因此下表"检索命中率 13/12 → 48/0"是**定义改变的结果，不是排序质量的改善** —— `_rank` 的排序逻辑未作任何修改。要分离两者需要新增一个 `recency_fallback` 计数器（本次未加，以免推送的代码与产出结果的代码不一致）。

---

## 3. 结果

### 3.1 同批三臂（job 595132）

| 臂 | `stage_score_pct` | `goal_success_rate` | `stage_success_rate` |
|---|---|---|---|
| `pushmem` | **28.8** | 0.2875 | 0.0 |
| `nomem` | **26.2** | 0.2625 | 0.0 |
| `pullmem`（工具重构**前**） | **22.5** | 0.2250 | 0.0 |

配对差值：`pushmem − nomem` = **+2.6 pp**；`pullmem − nomem` = **−3.7 pp**；`pullmem − pushmem` = **−6.3 pp**。

方向与预期（`pullmem > pushmem > nomem`）相反。

### 3.2 `pullmem` 工具重构后重测（job 595262）

| 臂 | `stage_score_pct` | `goal_success_rate` | 退出码 |
|---|---|---|---|
| `pullmem`（重构**后**） | **31.2** | 0.3125 | `RC=0` |

### 3.3 汇总

| 臂 | 分数 | 来源 |
|---|---|---|
| **`pullmem`（重构后）** | **31.2** | 595262 |
| `pushmem` | 28.8 | 595132 |
| `nomem` | 26.2 | 595132 |
| `pullmem`（重构前） | 22.5 | 595132 |

- `pullmem(后) − pushmem` = **+2.4 pp**（重构前为 −6.3 pp）
- `pullmem(后) − nomem` = **+5.0 pp**（重构前为 −3.7 pp）
- `pullmem(后) − pullmem(前)` = **+8.7 pp**

重构后顺序成为 `pullmem > pushmem > nomem`，是本轮唯一符合设计预期的顺序。

---

## 4. 机制证据（`pullmem` 重构前 / 后）

这是本报告最有说服力的部分：**分子与分母同时向预期方向移动**。

| 指标 | 595132（前） | 595262（后） | 方向 |
|---|---|---|---|
| plan steps | 68 | 59 | — |
| 工具说明展示次数 | 68 | 59 | —（= plan step 数，100% 覆盖） |
| 工具调用总数 | 95 | **67** | ↓ |
| 有工具调用的 step | 50（74%） | 48（81%） | ↑ |
| `by_tool` | `list_facts` 41、`search_memory` 25、`query_world` 23、`list_unbound` 6 | **`search_memory` 48、`query_world` 19** | 枚举工具归零 |
| 枚举类调用占比 | **49%**（47/95） | **0%**（0/67） | ↓ |
| 送回的上下文外观测帧 | 19 | **59** | ↑ 3.1× |
| **每次工具调用换到的帧数** | **0.20** | **0.88** | **↑ 4.4×** |
| `cap_hits` | 2 | **0** | ↓ |
| `api_errors` | 9 | 4 | ↓ |
| 已完成阶段数 | 20 | 27 | ↑ |
| census 结论 | FAIL（1 项：提名提示开启但 0 次提名） | **PASSED** | — |

关键帧库组成（`frame 0` 为官方锚点，占有残留在预期内）：

| 指标 | 594675（修前） | 595132 | 595262 |
|---|---|---|---|
| `frame 1` 占比（`pullmem`） | 44.8% | 2.6% | 5.6% |
| `frame 0` 占比（`pullmem`） | 100% | 66.7% | 55.6% |
| 库位 ≤ `frame 8`（`pullmem`） | 20.5% of 767 | 11.2% of 312 | 11.5% of 288 |
| 库位 ≤ `frame 8`（`pushmem`） | 23.7% of 1175 | 16.3% of 584 | — |

> 注：594675 的任务范围是 `[1,4,5,11,14]`，595132/595262 是 `[5]`。选取规则本身与任务无关，但该对比并非同范围。

---

## 5. 统计限制：本报告**不能**声称的结论

必须明确，否则上表会被过度解读。

1. **样本量与噪声**。`n=10` / 臂，单 trial 的 `stage_score_pct` 标准差实测约 **32 pp**（非饱和任务上 36.8 pp），因此差值标准误约 **14 pp**。上表所有差值（2.4 / 5.0 / 8.7 pp）**都落在噪声内**。
2. **不是配对设计**。`pushmem`/`nomem` 来自 595132，`pullmem` 来自 595262 —— 不同作业、不同墙钟时段、不同 API 负载。跨作业比较存在系统性差异风险。
3. **单任务单种子**。结论不外推到其他任务或完整 26 任务协议。
4. **检索命中率的改善是定义性的**（见 D5 警告），不是排序质量改善。
5. 因此：
   - **不能**说 `pullmem` 显著优于 `pushmem`；
   - **不能**说 31.2 是稳定值；
   - **可以**说的是：机制指标（帧/调用 0.20 → 0.88、枚举占比 49% → 0%、`cap_hits` 2 → 0）变化幅度大、方向一致，且与分数变化同向 —— 这构成"改动确实改变了机制"的强证据，而不是"改动提升了分数"的统计证据。

要获得统计结论，需要：把三臂放进**同一作业**、提高 `NUM_TRIALS`（按 SD=32 pp 估算，分辨 15 pp 需约 35 trial/臂，分辨 25 pp 需约 13 trial/臂），并在多个种子上重复。

---

## 6. 复现

```bash
cd /project/peilab/why/RoboMemArena

# 预检：确认基线对齐、每个处理臂只在本臂声明的键上不同
ROOT="$PWD" .venv/bin/python experiments/mem_efficacy/validate_arm.py

# 同批三臂（本报告的 595132 配置）
ARM_OVERRIDE="nomem pushmem pullmem" SEED_LIST=100 \
TASKS_JSON_SCOPE='[5]' NUM_TRIALS_SCOPE=10 \
sbatch experiments/mem_efficacy/run_26x1.sbatch

# pullmem 单臂重测（本报告的 595262 配置）
ARM_OVERRIDE="pullmem" SEED_LIST=100 \
TASKS_JSON_SCOPE='[5]' NUM_TRIALS_SCOPE=10 \
sbatch experiments/mem_efficacy/run_26x1.sbatch

# 事后 census：通道健康度 + 分数
ROOT="$PWD" .venv/bin/python experiments/mem_efficacy/census_channels.py \
  --run-root experiments/mem_efficacy/results/mem_efficacy_595262
```

`TASKS_JSON_SCOPE` 是**声明的子集**，census 会区分"声明的子集"与"被截断的运行"，并仍然标注该分数不是协议数字。

---

## 7. 代码 provenance

由 `code_provenance.json` 记录的 29 个输入文件的 SHA-256 与当前磁盘状态核对：

| 作业 | 一致 | 不一致 |
|---|---|---|
| **595262** | **29 / 29** | **0** —— 推送的代码精确复现该作业 |
| 595132 | 24 / 29 | 5 |

595132 中不一致的 5 个文件，及其对**该作业结果**的实际影响：

| 文件 | 是否影响 eval 行为 |
|---|---|
| `experiments/mem_efficacy/census_channels.py` | 否（仅事后分析） |
| `experiments/mem_efficacy/validate_arm.py` | 否（仅预检） |
| `experiments/mem_efficacy/memexp_bind.py` | 否（改动位于 `verify_binding()`，仅被预检调用，不在 eval 路径） |
| `experiments/mem_efficacy/arms/pullmem.sh` | **是**，仅影响 `pullmem` |
| `experiments/mem_efficacy/memexp_tools_search.py` | **是**，仅影响 `pullmem` |

结论：**595132 的 `nomem` 与 `pushmem` 结果与当前代码完全一致，可复现**；595132 的 `pullmem` 是本文档明确记录的"重构前"状态，其差异即 D3/D4/D5。

结果目录中不含密钥（`api_key.txt` 已被 `.gitignore` 排除；两个结果目录经明文密钥与通用密钥模式扫描，均为干净）。
