# 代码快照与审计脚本说明

本目录记录产出 `results/` 中基线的代码身份，以及为判定"哪些任务可发布"而写的四个审计脚本。

---

## 1. 代码身份

基线的每个 run 都带 `code_provenance.json`，逐一记录 29 个 source 文件的 SHA-256。其中与本审计相关的关键项：

| 文件 | 运行时 SHA-256 | 说明 |
|---|---|---|
| `evaluation_benchmark/harness/stage_mapper.py` | `40cabe7d76fd21ba…` | **缺陷版本**，即真正参与运行的版本 |
| `evaluation_benchmark/harness/controller.py` | `33ecc3a824757235…` | 含 `_handle_stall` 与 `candidate != subtask` 卫语句 |
| `evaluation_benchmark/harness/api_vlm_planner.py` | `b1e80427b0224482…` | Planner 客户端 |
| `experiments/mem_efficacy/arms/nomem.sh` | `542fb65f042a5971…` | 基线 arm 定义 |

其它 provenance 字段：`job=596348`、`git_head=bc6d088`、`git_describe=bc6d088-dirty`（工作区有 230 个未提交改动）、`missing_files=[]`。

### 快照内容与哈希

```
40cabe7d76fd21ba  code/evaluation_benchmark/harness/stage_mapper.runtime-40cabe7d.py
```

**这个文件名里的哈希不是标注，是校验值**：它与 `code_provenance.json` 记录一致（`40cabe7d76fd21ba…`），因此就是当时真正运行的文件。它被保存为**证据**而非"推荐使用的代码"—— 缺陷被完整保留以便复现。

当前修复版为 `705c516c545ef1f4…`，**不在本目录**（它在 `evaluation_benchmark/harness/stage_mapper.py`），差异见 `AUDIT.md` 第 4 节。

```
9da9168cc299ee4e  code/evaluation_benchmark/tests/_buggy_matcher_plugin.py
de549ac98ef2ed28  code/evaluation_benchmark/tests/test_stage_mapper_tie.py
```

`test_stage_mapper_tie.py` 是钉住该缺陷的单元测试。`_buggy_matcher_plugin.py` 是配套的**负对照**：通过 `-p _buggy_matcher_plugin` 在**内存中**重装缺陷版匹配器（不触碰磁盘），用以证明这些测试真的具备判别力 —— 否则会出现"修复前后都通过"的无效测试。

| 版本 | 测试结果 |
|---|---|
| 缺陷版（内存补丁） | **9 failed, 26 passed** |
| 修复版 | **45 passed** |

---

## 2. 审计脚本

四个脚本只读，不写任何基线目录。

### `mapper_equivalence_for_clean_tasks.py` — 决定"哪些任务可发布"

把缺陷版（从 `code_archive/` 按哈希加载）与修复版对同一批输入逐一对拍，扫遍每个 `stage_idx` × 典型 `current_subtask` × 每个 `ndone`。

```
task 4, 5 : 1400 次比对，0 不一致   ← 索引对齐路径，缺陷不可达
task 20,23:  176 次比对，0 不一致   ← 模糊路径可达，但无平局
task 12/13/17: 275 次比对，110 不一致 ← 平局命中缺陷
```

输出：`results/audit/mapper_equivalence.txt`

**这是任务 20/23 可发布的直接依据**：缺陷对它们是恒等变换，所以"在缺陷版下跑的"与"在修复版下跑的"是同一个结果，无需重跑。

### `attribute_overrides_to_stage.py` — 量化污染

按 `harness_memory.json` 的 `episodic` 流定位**每次 override 发生时真正在跑的 stage**，再判定它是否正确。

关键点：不能用 `attempts[].stalled_stage` 来评分 —— 那是 episode **结束时**卡住的 stage。早期 override 发生在机器人合法地卡在 `01_Open_*` 时，给出 `open …` 是**正确**的；只有在该 stage 已确认完成之后发出的 override 才算回退。

输出：`results/audit/override_attribution.{txt,json}`

### `run_integrity_check.py` — 检出作业碰撞

检测三类症状：同一 ep 目录出现多条 `Episode` 记录、单个日志内时间戳倒序（两个追加偏移的指纹）、`summary.json` 出现重复 `task_id`。

正是它把任务 4/5 判为不可发布（`REPORT.md` 3.2 节）。

输出：`results/audit/run_integrity.{txt,json}`

### `audit_volo_baseline.py` — 污染面概览

按任务列出 override 取值分布与 `expected_but_absent`，用于快速定位受害族。

---

## 3. 证据链小结

```
                   ┌─ mapper_equivalence ──► 20/23 恒等 ──┐
任务 20/23 ────────┤                                       ├──► 可发布
                   └─ run_integrity ──────► run 自洽 ─────┘

任务 12/13/17 ─── mapper_equivalence ──► 110 处不一致 ──┐
               └ attribute_overrides ──► 51.8% 误导向 ──┴──► 剔除（待重跑）

任务 4/5 ──────── run_integrity ──► 6 个 ep 目录含双写 ────► 剔除（待重跑）
```
