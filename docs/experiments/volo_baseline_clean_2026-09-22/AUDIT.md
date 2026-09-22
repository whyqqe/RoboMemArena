# VoLo 对齐基线审计：`stage_mapper` 平局缺陷如何污染了抽屉族的恢复通道

- 日期：2026-09-22
- 被审计对象：`volo_aligned_selffamily_10x`（作业 `596348`）、`volo_aligned_t4t5_10x`（作业 `596262`）
- 结论一句话：**这两个基线在抽屉族（任务 12/13/17）不可用**。运行时 `stage_mapper.py` 的模糊匹配在平局时取了先出现的标签，导致 stall 恢复阶梯在 place 阶段把机器人推回 `open middle drawer`。逐事件归因测得抽屉族 **58/112（51.8%）** 的 override 是反方向的，而微波族为 **0/49（0.0%）**。修复后已重跑（作业 `605408`）。

---

## 1. 为什么会有这份审计

在复盘 `rsiguard` 实验时发现：抽屉族 9/9 episode 的 `subtask_override` 计数为 0，而微波族每 episode 约 2 次。两个族共用同一个 VLA、同一个 Planner、同一批 seed，唯一差别是 **stage 的命名**。这条线索指向 harness 内部的阶段→基元映射，而不是模型。

---

## 2. 证据链

### 2.1 运行时确实是缺陷版本（哈希级）

`code_provenance.json` 记录了每个 source 文件的 SHA-256：

```
volo_aligned_selffamily_10x : stage_mapper.py = 40cabe7d76fd21ba145201a8aca18b27a4b699612cef240ce0ffd71a8d33a3c2
volo_aligned_t4t5_10x       : stage_mapper.py = 40cabe7d76fd21ba145201a8aca18b27a4b699612cef240ce0ffd71a8d33a3c2
```

该版本**不在** `git HEAD`（`1f02faca…`）也不在当时的磁盘状态中，但完整保存在代码归档里：

| 位置 | 哈希 |
|---|---|
| `code_archive/api_pmh_md__587007/src/harness/stage_mapper.py` | `40cabe7d…` ✅ |
| `code_archive/api_seam_m{,2,3,4,5}__*/src/harness/stage_mapper.py` | `40cabe7d…` ✅ |
| `docs/experiments/mem_efficacy_pull_push_2026-09-18/code/…/stage_mapper.py` | `40cabe7d…` ✅ |

即缺陷版本可被完整复原与复现。

### 2.2 缺陷本身

`expected_primitive_for_stage` 的规则 (2)（模糊匹配）以「标签中长度 >2 的词元有多少出现在 stage 名里」计分，并用**严格大于**取最优：

```python
hit = sum(1 for tok in tokens if tok in stage_name)
if hit >= min(2, len(tokens)) and hit > best_score:   # 严格 >：平局时先出现的赢
    best, best_score = label, hit
```

对 `stage_idx=1`（活跃 stage = `02_Place_Cookies_Middle_Drawer`）：

| 候选标签 | 命中词元 | 得分 |
|---|---|---|
| `open middle drawer` | `middle`, `drawer` | **2** ← 平局，且先出现 |
| `place cookies` | `place`, `cookies` | **2** |
| `pick cookies` | `cookies` | 1（被阈值拒绝） |

严格 `>` 使 `open middle drawer` 获胜。

### 2.3 为什么缺陷表现为「静默」

`controller.py:670` 只在**与当前 subtask 不同**时才写出 override：

```python
if self.config.subtask_override_on_stall and candidate and candidate != subtask:
    self.subtask_override = candidate
```

而机器人卡住时执行的正是 `open middle drawer`，与候选相同 → 卫语句拦下 → **阶梯静默失效**。

只有当 subtask 漂移成别的字符串（如 `reach_to_middle_drawer_handle`）时，卫语句才放行，此时 override 反而把机器人推回已完成的开抽屉阶段。

### 2.4 为什么只有两个族受害（族间不对称的结构性解释）

规则 (1)（索引对齐）在 `len(primitive_labels) == len(stage_specs)` 时**优先返回**，根本不进入模糊路径：

| 任务 | stages / labels | 走哪条规则 | 结果 |
|---|---|---|---|
| 4, 5 | 9 / 9 | 规则 (1) 索引对齐 | **正确**，未受害 |
| 12, 13, 17 | 4 / 6 | 规则 (2) 模糊 → 平局 | **受害** |
| 20, 23 | 3 / 6 | 规则 (2) 模糊，但无词元重叠 | 恰好正确 |

微波族的 stage 名 `02_Place_Cookies_Microwave` 与 `open microwave` 无共享词元，`open microwave` 得分为 1，被 `hit >= 2` 阈值拒绝 —— 因此**偶然躲过**。

### 2.5 实测：override 取值分布

```
task12   40 × 'open middle drawer'                      ← 无一次 place*
task13   31 × 'open middle drawer'                      ← 无一次 place*
task17   42 × 'open middle drawer'                      ← 无一次 place*
task20   12 × open microwave + 11 × place cookies + 5 × place chocolate   ← 正常
task23   10 × open microwave +  8 × place cream   + 5 × place popcorn     ← 正常
```

### 2.6 逐事件归因（决定性证据）

`harness_memory.json` 的 `episodic` 流记录了每次 `stall` 事件**当时的活跃 `stage_name`**，因此可以把每次 override 归因到它发生时真正在跑的 stage，而不是 episode 结束时卡住的 stage。

脚本：`code/experiments/mem_efficacy/attribute_overrides_to_stage.py`（本目录内，随本提交跟踪；
工作树中的等价路径为 `experiments/mem_efficacy/`，该目录按仓库惯例不纳入版本控制）

| 族 | 可归因 override | 正确 | **误导向** | 误导向率 |
|---|---|---|---|---|
| 抽屉 12/13/17 | 112 | 54 | **58** | **51.8%** |
| 微波 20/23 | 49 | 47 | **0** | **0.0%** |
| 4, 5 | 97 | 55 | 1 | 1.0% |

典型实例：

```
task12 @t=720 :  活跃 stage = 02_Place_Cookies_Middle_Drawer
                 ladder 给出 'open middle drawer'，正确应为 'place cookies'
                 （01_Open_Middle_Drawer 已于 t=639 确认完成 —— 这是回退到已完成阶段）

task17 @t=1890:  活跃 stage = 03_Place_Chocolate_Middle_Drawer
                 ladder 给出 'open middle drawer'，正确应为 'place chocolate'
```

---

## 3. 这不是执行器（VLA）的问题

**该猜测已排除，理由有三，且第一条是决定性的。**

### 3.1 同一次运行内的族间不对称

抽屉族 51.8% 误导向 vs 微波族 0.0%，这两组数据来自**同一作业、同一 VLA 权重、同一批 seed（100–109）、同一 harness**。执行器不可能因为 harness 如何给 stage 命名而改变行为。唯一的自变量是词元重叠。

### 3.2 基线用的就是 PrediMem 的 VLA，不存在「换」这个动作

| # | 证据 | 内容 |
|---|---|---|
| 1 | `arms/official_protocol.sh` | `export VLA_CONFIG=pi05_robomemarena`，注释：*served from `checkpoints/PrediMem/vla_alltask` … Frozen for every arm* |
| 2 | `arms/nomem.sh` | `source official_protocol.sh`，逐字继承 |
| 3 | 两个基线的 `run.log` | `VLA_CONFIG=pi05_robomemarena`、`VLA_CKPT=…/vla_alltask` |
| 4 | `logs/serve_policy.log` | 实际加载 `checkpoints/PrediMem/vla_alltask/params`（6.2 GiB orbax），资产路径含 `robomemarena_assets` |
| 5 | `outputs/paper_baselines_rma_535214/pi05_reactive` manifest | `"mode": "reactive VLA (PrediMem vla_alltask, no VLM)"` |

三者关系：

```
VoLo 对齐基线 = PrediMem VLA + API Planner + harness
PrediMem      = PrediMem VLA + PrediMem 自己的记忆
pi05_reactive = PrediMem VLA 单独
```

**VLA 是三者共用的常量，无法充当差异来源。**

### 3.3 本地没有第二个具备 RMA 能力的执行器

| 候选 | 结果 |
|---|---|
| `MemoryVLA/memvla-libero-100` | 零样本 LIBERO，论文基线实测 **0.0% CSR / 0.0% TSR**（26 任务全零） |
| `EventVLA/RoboTwin-MeM` | 训练域 `rmbench_hard8`（`RMBench/lerobot_data`），**另一个 benchmark**；本地 RMA 数据在 `data/RoboMemArena` |
| `MemER` | 公开版是 dusting 任务的 high-level Qwen3-VL，即 planner，非执行器 |
| `HiF-VLA` | 无公开 RMA checkpoint |

### 3.4 VLA 确实弱，但 harness 在帮它

| 配置 | 同 5 任务（12/13/17/20/23）平均 CSR |
|---|---|
| `pi05_reactive`（同一 VLA，无 Planner） | 26.7%（n=1 每任务，噪声极大） |
| VoLo 对齐基线（同 VLA + Planner + harness） | 52.7%（n=10） |

同一份权重下 Planner 层把成绩翻了一倍以上。执行器对这类记忆任务确实弱（这是 RMA 论文的前提），但本 harness 是在补偿它。

---

## 4. 修复

`evaluation_benchmark/harness/stage_mapper.py` 的规则 (2) 增加**动词一致性平局裁决**：平局时优先选择动词与 stage 动词一致的标签。

```python
stage_verb = next((v for v in _STAGE_VERBS if v in stage_name), None)
for label in primitive_labels:
    label_verb = next((tok for tok in tokens if tok in _STAGE_VERBS), None)
    verb_ok = stage_verb is not None and label_verb == stage_verb
    # 主键：词元命中数。平局裁决：动词一致。
    if hit > best_score or (hit == best_score and verb_ok and not best_verb_ok):
        best, best_score, best_verb_ok = label, hit, verb_ok
```

**修复是外科式的**：只改变平局结果。任务 4/5 走索引对齐路径、微波族无平局，两者均不受影响（并有测试断言）。

| 版本 | SHA-256 |
|---|---|
| 污染运行时 | `40cabe7d76fd21ba145201a8aca18b27a4b699612cef240ce0ffd71a8d33a3c2` |
| 修复后 | `705c516c545ef1f45283d5713cfa0c0144420e4663b033423f5c681c2a95cc4c` |

### 测试具备判别力

新增 `evaluation_benchmark/tests/test_stage_mapper_tie.py`。关键的**负对照**通过 `-p _buggy_matcher_plugin` 在内存中重装缺陷版匹配器（不触碰磁盘）：

| 版本 | 结果 |
|---|---|
| 缺陷版（原始 `hit > best_score`） | **9 failed, 26 passed** |
| 修复版 | **45 passed** |

失败的 9 项包含 3 个**端到端**测试：它们直接驱动真实 `HarnessController`，断言 `subtask_override` 真的被写出 `place cookies` —— 正是实测为 0 的那个量。

### 实盘验证（重跑作业 `605408`）

第一集落盘后即刻核验（缺陷版此处恒为 0 次 place*）：

```
task12/ep0 : 2 × 'open middle drawer' + 1 × 'place cookies'
task12/ep1 : 2 × 'open middle drawer' + 2 × 'place cookies'
```

`place cookies` override 首次出现，修复在实际运行路径上生效。

---

## 5. 已知的数据不一致（非本次缺陷）

task 11 的 `05_Place_Butter_Middle_Drawer` 在 `primitive_order` 中**不含任何 butter 标签**，任何匹配器都无法为该 stage 找到候选。已用独立测试 `test_known_task_definition_inconsistencies` 显式钉住。任务 11 不在本实验集内。

---

## 6. 结论与影响范围

1. **本缺陷按任务选择性污染，因此两个 run 都不能整体发布，但都含可用的干净子集。**

   | task | 本缺陷 | 另一独立问题 | 可用性 |
   |---|---|---|---|
   | 12, 13, 17 | **污染**（51.8% 误导向） | — | ❌ 剔除 |
   | 20, 23 | 干净（0.0%） | 无 | ✅ **已发布** |
   | 4, 5 | 干净（1.0%，且结构上缺陷不可达） | **作业碰撞**（ep 目录双写） | ❌ 剔除 |

   发布子集为任务 **20 / 23**，见同目录 `REPORT.md`。任务 4/5 的剔除与本缺陷无关，
   成因见 `REPORT.md` 第 3.2 节。

2. **任何在污染数据上做的「记忆 vs 无记忆」对照都会产生错误归因**，因为处理效应在族间是不对称的（抽屉族恢复通道 51.8% 反向，微波族 0%）。

3. 污染与执行器无关，与 Planner 模型无关，也与记忆通道无关 —— 它是一个纯粹的基础设施缺陷。

4. **"干净"不止是"实测 0 次误导向"**：对 20/23 已证明缺陷是**恒等变换**（逐输入 176/176 一致），因此无需重跑；而 4/5 虽同样干净，却因运行完整性问题必须重跑。

5. 修复已完成并通过判别性测试（缺陷版 9 failed / 修复版 45 passed），但**重跑被 API 余额为负阻塞**（见 `REPORT.md` 第 5 节）。
