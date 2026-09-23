# Stage 0 — 运行台账

> 日期：2026-09-23。用途：把已有报告数字指回作业、配置和原始输出。找不到原始输出的标为「仅报告级证据」。

## 1. 主线资产（API Planner + 冻结 PrediMem π0.5）

| 作业 / 报告 | 臂 | Planner | VLA | 任务 | 种子 / trials | retry / stage-anchor | 结果目录 | 证据等级 |
|---|---|---|---|---|---|---|---|---|
| **595132** | nomem / pushmem / pullmem（修前） | gemini-3.8-flash | PrediMem pi05 冻结 | task 5 | seed 100，NUM_TRIALS=10 | ON（Hlegacy 类）/ MEM_STAGE_ANCHOR=1 on push/pull | `docs/experiments/mem_efficacy_pull_push_2026-09-18/results/mem_efficacy_595132/` | **原始可追溯**：`aggregate.json` + traces |
| **595262** | pullmem（工具重构后） | 同上 | 同上 | task 5 | 同上 | 同上 | `.../mem_efficacy_595262/` | **原始可追溯**；与 595132 对照臂**跨作业**，不可直接当同批配对 |
| dual_track_report | memory_kf / harness_exec / harness_v21 等 | **本地 PrediMem Qwen**（非 API） | PrediMem | 8 任务 × 5 seed | 1 trial/cell | memory track: retry OFF；harness track: retry ON | 报告级；见 `docs/experiments/dual_track_report.md` + `REPRODUCIBILITY.md` | **仅报告级**（与 API 线不可直接相减） |

### 595132 / 595262 冻结分数（`macro_stage_score_pct`）

| 臂 | 作业 | stage_score_pct | 报告中的 goal_success_rate |
|---|---|---:|---:|
| nomem | 595132 | 26.2 | 0.2625 |
| pushmem | 595132 | 28.8 | 0.2875 |
| pullmem（修前） | 595132 | 22.5 | 0.2250 |
| pullmem（修后） | 595262 | 31.2 | 0.3125 |

代码指纹：报告称 595262 与快照 `code/` 29 文件 SHA-256 一致。快照路径：`docs/experiments/mem_efficacy_pull_push_2026-09-18/code/`。

### 配置要点（595132/595262，Hlegacy 类）

- `HARNESS_MAX_RETRIES=2`，subtask override / stage checkpoint / smart retry ON
- push/pull：`VLM_USE_KEYFRAME_MEMORY=1`，`MEM_STAGE_ANCHOR=1`，`MEM_KF_STORE_INTERVAL=5`，`K_MAX=8`
- 三臂共同：`HARNESS_VLM_CONTEXT=0`
- pull 修复后：工具说明去掉枚举招揽；`MEMEXP_PULL_MAX_ROUNDS=2`；miss 回退 `recency_fallback`

## 2. 本地 PrediMem 双轨（不可与上表混算）

| 声明 | 来源 | 原始输出 | 状态 |
|---|---|---|---|
| 组合 `harness_v21` ~+13.4 pp CSR | `dual_track_report.md` | 报告未在本仓库附带完整 `aggregate.json` 树 | **仅报告级**；与 API pull/push **不是同一底座** |
| 主实验 ~+5.3 pp CSR | `REPRODUCIBILITY.md` Tier II | 指向 `harness_main_p0.sbatch` 协议 | **与 dual_track +13.4 不同来源**；核清前不得合并引用 |

## 3. VoLo 对齐基线（非本阶段主线）

| 作业 / 文档 | 任务 | 状态 |
|---|---|---|
| `docs/experiments/volo_baseline_clean_2026-09-22/` | 仅 **20 / 23** | 已发布干净结果；12/13/17（mapper）与 4/5（OUT_ROOT 碰撞）**未**纳入 |
| 阶段 1 **不**重跑 VoLo | — | 按学长指示：VoLo / Harness VLA 视为外部 baseline，Proactive Memory 加在 VLM+VLA harness 之上 |

## 4. 已知污染 / 不可发表作业（备查）

| 问题 | 作业 / 现象 | 处理 |
|---|---|---|
| OUT_ROOT 碰撞 | 596262 / 596284 同目录双写 | 分数作废；`run_26x1.sbatch` 已加 `.run.lock` |
| Planner 余额耗尽仍重试 | 多作业 `insufficient_balance` 403 重试 | `api_planner` 永久 abort + `API_QUOTA_EXHAUSTED`；Stage-1 早停 |
| 死 planner 继续跑 | 598082/83/84 | `PLANNER_MAX_CONSECUTIVE_FAILURES` + watchdog 自检 |

## 5. 阶段 1 新跑（本目录后续填写）

| 作业 | MEMPROF | 模型 | 臂 | 任务 | trials | OUT_BASE | 状态 |
|---|---|---|---|---|---|---|---|
| 609328 | h0 | gpt-6-luna | pullmem | [5] | 1 | `.../stage1_h0_gpt-6-luna_609328/` | **FAILED GATE 0**：继承交互式 `PLANNER_API_MAX_TOKENS=256` + GATE0 原始 `max_tokens`；已修 |
| **609345** | h0 | gpt-6-luna | pullmem | [5] | 1 | `experiments/mem_efficacy/results/stage1_h0_gpt-6-luna_609345/` | **流水线通过**：eval RC=0；H0 开关正确；pull 7 次工具调用 / 4 帧；task5 单 trial `stage_score=0.0`（分数不作结论）。原 job 因 census 读裸 arm 的 `MEM_STAGE_ANCHOR=1` 误报 nomination FAIL（RC=4）；修 census 后本地复检 **PASSED** |
