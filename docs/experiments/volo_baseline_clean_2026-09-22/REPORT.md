# VoLo 对齐基线（未被污染子集）：RoboMemArena 任务 20 / 23

- 日期：2026-09-22
- 来源作业：`596348`（`volo_aligned_selffamily_10x`）
- 一句话结论：在 **任务 20 / 23** 上给出 VoLo 架构对齐的 `nomem` 基线，10 trials × seed 100，宏平均 **CSR 53.3% / TSR 25.0%**。该子集同时通过**映射器等价性**与**运行完整性**两道检验，可作为干净对照使用。

> **什么是"VoLo 对齐基线"**：把 VoLo 的**编排层**（VLM Planner + 监视-中止-恢复 harness）接到本仓库已有的 **PrediMem 冻结执行器**上，并关闭全部记忆通道。因此它测的是"Planner+harness 本身能拿到多少分"，是所有记忆机制（`pushmem`/`pullmem`/RSI 经验记忆等）的对照起点。

---

## 本目录内容

| 路径 | 内容 |
|---|---|
| `REPORT.md` | 本报告 |
| `CODE.md` | 代码快照清单与审计脚本说明 |
| `AUDIT.md` | 污染审计全文（为何剔除 12/13/17） |
| `code/` | 产出这些结果的代码快照 + 4 个审计脚本 + 修复后的单元测试 |
| `results/volo_aligned_selffamily_10x_clean_subset/nomem_s100/` | **本报告数据**：仅 `task20` / `task23` |
| `results/audit/` | 三道检验的原始输出（映射器等价性、override 归因、运行完整性） |

---

## 1. 协议与配置

| 项 | 值 |
|---|---|
| Planner | `claude-opus-4-6`（API），`https://api.closeai-asia.com/v1` |
| 执行器 VLA | `pi05_robomemarena` ← `checkpoints/PrediMem/vla_alltask`（**冻结**，所有 arm 共用） |
| PrediMem VLM ckpt | `checkpoints/PrediMem/vlm_tasks1to26_ckpt74500` |
| PrediMem revision | `e645741c2f34f27e1596bfa89e856d6f3560ed90` |
| 任务 | 20, 23（自指族·微波） |
| trials / seed | 10 / 100（`NUM_TRIALS=10` → seeds 100–109） |
| `MAX_STEPS` | 2500 |
| harness | `max_retries=2 stall_steps=80 vlm_context=False vla_hints=False smart_retry=True skip_score=95.0` |
| 记忆通道 | **全关**：`VLM_USE_KEYFRAME_MEMORY=0`、`K_MAX=0`、`N_RECENT=5`、`D_MERGE=6` |
| 基线对齐 | `arms/nomem.sh` → `source arms/official_protocol.sh`，只做已声明的偏离 |

`nomem` 的定义即"官方协议 + harness，且每一条记忆通道都关闭"，由 `validate_arm.py` 在作业内断言（该作业 GATE 1 `preflight PASSED`）。

**注意**：本 run 的 `TASKS_JSON=[12,13,17,20,23]`（5 个任务同批跑完）。本目录**只发布其中 2 个**，原因见第 3 节。原始 5 任务记录完整保留为 `*_all5tasks_original.*`。

---

## 2. 结果

| task | 场景 | CSR | TSR | stage_score |
|---|---|---|---|---|
| 20 | 先把 cookies 放进微波炉，再把 chocolate 放到 cookies 所在位置 | 50.0% | 10.0% | 50.0% |
| 23 | 先把 cream 放进微波炉，再把 popcorn 放到 cream 所在位置 | 56.7% | 40.0% | 56.7% |
| **宏平均** | | **53.3%** | **25.0%** | **53.4%** |

`aggregate_clean_subset.json` 为按 task20/23 重算的结果（`num_tasks: 2`）。原始 5 任务宏平均为 52.7%，保留在 `aggregate_all5tasks_original.json`，**不要**与上表混用。

### 与同 VLA 的无 Planner 对照

| 配置 | 任务 20 | 任务 23 | 两者均值 |
|---|---|---|---|
| `pi05_reactive`（同一 VLA，**无** Planner） | 0.0% | 100.0% | 50.0% |
| **本基线**（同 VLA + Planner + harness，n=10） | 50.0% | 56.7% | **53.3%** |

`pi05_reactive` 每任务仅 1 trial，噪声极大，**不构成统计比较**，仅用于说明这两任务对执行器本身并非不可达。

---

## 3. 为什么只发布任务 20 / 23

被剔除的两个成因**互不相同**，必须分开陈述。

### 3.1 任务 12 / 13 / 17：`stage_mapper` 平局缺陷污染（详细见 `AUDIT.md`）

运行时 `stage_mapper.py` 的 SHA-256 为 `40cabe7d…`，其模糊匹配在平局时取了先出现的标签。后果：在 `02_Place_*` / `03_Place_*` 阶段，恢复候选变成 `open middle drawer` —— 正是机器人已经卡住的动作，于是被 `controller.py` 的 `candidate != subtask` 卫语句拦下，阶梯**静默失效**；一旦 subtask 漂移，override 反而把机器人推回**已确认完成**的开抽屉阶段。

逐事件归因（按 `episodic` 流中的活跃 `stage_name` 判定，而非 episode 结束时卡住的 stage）：

| 族 | 可归因 override | 正确 | **误导向** | 误导向率 |
|---|---|---|---|---|
| 抽屉 12/13/17 | 112 | 54 | **58** | **51.8%** |
| **微波 20/23** | 49 | 47 | **0** | **0.0%** |
| 4, 5 | 97 | 55 | 1 | 1.0% |

### 3.2 任务 4 / 5：运行完整性失败（作业碰撞，**与上述缺陷无关**）

`volo_aligned_t4t5_10x` 由两个作业（`596262` 与 `596284`）写入**同一输出目录**，各自以追加模式打开 `sync_vlm.log`，导致：

```
task4/ep1  line 618:  12:54:01  Episode 1 seed=101  stage_score=12.5
           line 639:  12:51:06  Episode 1 seed=101  stage_score=50.0   ← 时间更早却在文件末尾
```

即**同一 episode 目录里有两份同 seed、不同分的记录**，且时间戳倒序。episode 级的 `harness_memory.json` / `api_vlm_trace.jsonl` / 视频只属于**其中一个**写入者，`summary.json` 也随之出现 task4 重复行。

| run | 异常 ep 目录 | `summary.json` | 判定 |
|---|---|---|---|
| `volo_aligned_t4t5_10x` | task4: 3 个、task5: 3 个 | rows=3 / unique=2（task4 重复） | ❌ 不可发布 |
| `volo_aligned_selffamily_10x` | **0 个** | 正常 | ✅ 内部自洽 |

**因此任务 4 / 5 需重跑**，且当前被 API 余额阻塞（见第 5 节）。原始数据保留在 `experiments/mem_efficacy/results/volo_aligned_t4t5_10x/`，未删除。

### 3.3 为什么任务 20 / 23 是干净的 —— 机械论证，非修辞

对 20/23 的"干净"不能停留在"实测 0 次误导向"，而应证明**缺陷对它们是恒等变换**：

| task | stages / labels | 走的代码路径 | 比对次数 | 缺陷版 vs 修复版不一致 |
|---|---|---|---|---|
| 4, 5 | 9 / 9 | 规则 (1) 索引对齐 → **缺陷不可达** | 1400 each | **0** |
| **20, 23** | 3 / 6 | 规则 (2) 模糊路径**可达但无平局** | 176 each | **0** |
| 12, 13, 17 | 4 / 6 | 规则 (2) **平局 → 缺陷命中** | 275 each | **110** |

20/23 无平局的原因：唯一竞争标签 `open microwave` 仅命中 `microwave` 一词（`hit=1`），低于 `hit >= min(2, len(tokens))` = 2 的门槛而被拒绝；而抽屉的 `open middle drawer` 会命中 `middle` + `drawer`（`hit=2`）与 `place cookies` 打平。

**结论**：20/23 在缺陷版与修复版下**逐输入完全一致**（176/176）。它们不是"碰巧跑对"，也**无需重跑**；发布它们不引入任何已知偏差。检验脚本与输出见 `code/experiments/mem_efficacy/mapper_equivalence_for_clean_tasks.py` 与 `results/audit/mapper_equivalence.txt`。

---

## 4. 使用本基线时的边界

1. **只在任务 20 / 23 上可作对照。** 宏平均 53.3% 是这两个任务的宏平均，**不可**与 26 任务宏平均或 5 任务宏平均相比 —— 任务集是事后按"未被污染"选出的。
2. **同批的 12/13/17 数据不可用**（`AUDIT.md`），**任务 4/5 数据亦不可用**（3.2 节），两者都在各自 run 的目录内，不要误取。
3. `prompt_trace.tsv`、`run.log`、`code_provenance.json` 是**整个 5 任务 run** 的记录（含 12/13/17），保留是为了溯源完整性，不是本子集的数据。
4. Planner 为 `claude-opus-4-6`。换 Planner 即为不同实验。

---

## 5. 已知阻塞与后续

| 事项 | 状态 |
|---|---|
| 任务 12/13/17 重跑 | 修复已就绪并验证（`AUDIT.md` 第 4 节），作业 `605408` 已提交但因**余额不足**作废（24/24 episode 因 Planner 连续空响应中断） |
| 任务 4/5 重跑 | 需在**独立输出目录**重跑，避免再次碰撞 |
| API 余额 | **为负**：`您的可用余额为 -0.20998753 元`（403 `insufficient_balance`）。需充值后方可继续 |
| 看门狗 | `PLANNER_MAX_CONSECUTIVE_FAILURES` 已生效，其本身工作正常（正是它中止了 `605408`，避免产出假分数） |

> 关于 `605408`：该作业的**分数下降是假象**。其 Planner 死亡后 episode 在 ~470 步即中止（正常 2500 步），导致 stage 完成数从 21 降到 12。对其做归因可见修复本身是成功的：**63/63 override 全部正确，0% 误导向**（对比污染版 54/112、51.8% 误导向）。该 run 的任何分数都不可引用。
