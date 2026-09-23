# Stage 0+1 — 共同底座基线包（2026-09-23）

Proactive Memory 是主创新；当前 VLM+VLA 与 VoLo 只作不同 baseline。本包完成 **阶段 0 审计** 与 **阶段 1 可重复底座**（先不重跑 VoLo）。

## 本目录

| 文件 | 内容 |
|---|---|
| `LEDGER.md` | 运行台账（作业 → 配置 → 原始输出） |
| `METRICS.md` | 指标口径（含 `goal_success_rate = stage_pct/100` 说明） |
| `CLAIMS.md` | 旧结论证据等级与能否复用 |
| `REPORT.md` | 本文件：底座定义与如何跑 Stage-1 |

## 共同底座（不是 Harness VLA / VoLo）

固定：RoboMemArena 评估路径、相机/动作接口、冻结 PrediMem π0.5 VLA、API Planner、任务初始状态协议。

两个 profile（`experiments/mem_efficacy/profiles/`）：

| Profile | 用途 | 关键开关 |
|---|---|---|
| **Hlegacy** | 复核 09-18；解释旧结果 | retry / override / checkpoint ON；push/pull 可 `MEM_STAGE_ANCHOR=1` |
| **H0** | 记忆因果主线 | 上述全 OFF；`MEM_STAGE_ANCHOR=0`；purge LTM/ledger；trial 记忆从空白开始 |

臂：`nomem` / `pushmem` / `pullmem`（写入程序在 push/pull 上相同；差异只在 Planner 读到什么）。

默认 Planner：**`gpt-6-luna`**（CloseAI，成本低于 gemini-3.8-flash）。`api_planner.py` 已处理 `max_completion_tokens`、禁 `temperature`、余额不足不重试。

## 安全闸门（避免旧事故）

| 闸门 | 位置 |
|---|---|
| 臂×profile 环境契约 | `profile_preflight.py` |
| 模型可达（廉价 1×1 图） | `planner_model_probe.py` |
| OUT_ROOT 防碰撞 | `run_26x1.sbatch` `.run.lock` |
| 配额早停 | 日志出现 `API_QUOTA_EXHAUSTED` → 杀评估进程（`EARLY_STOP_ON_QUOTA=1`） |
| 死 planner | `PLANNER_MAX_CONSECUTIVE_FAILURES`（Stage-1 默认 3） |

## 如何跑

```bash
cd /project/peilab/why/RoboMemArena

# 本地预检（不占 GPU）
.venv/bin/python experiments/mem_efficacy/profile_preflight.py
PLANNER_API_MODEL=gpt-6-luna .venv/bin/python experiments/mem_efficacy/planner_model_probe.py

# Stage-1 默认 smoke：H0 + pullmem + task5 × 1 trial
sbatch experiments/mem_efficacy/run_stage1.sbatch

# 显式三臂（仍建议先 NUM_TRIALS=1）
ARM_OVERRIDE="nomem pushmem pullmem" sbatch experiments/mem_efficacy/run_stage1.sbatch
```

结果目录：`experiments/mem_efficacy/results/stage1_<profile>_<model>_<jobid>/`（每作业唯一）。

## 完成标准对照

| 标准 | 状态 |
|---|---|
| 运行台账 + 指标定义 + 旧结论清单 | 本目录三文件 |
| Hlegacy / H0 可复用配置 | `profiles/*.sh` |
| 预检 | `profile_preflight.py`（本地 PASSED） |
| task5 smoke | job **609345**：H0+pullmem+gpt-6-luna；机制（工具拉取）OK；分数单 trial 不作结论 |
| 不重跑 VoLo | 遵守 |

阶段 2（同版三臂复核 + H0 四臂）在额度确认后再开。
