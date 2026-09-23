# Stage 0 — 指标口径

> 日期：2026-09-23。新报告必须按本文件分列；旧字段仅作兼容，不再单独称为「CSR」而不加说明。

## 1. 评估器实际在算什么

在 `evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py` 中：

```text
goal_cnt += stage_pct / 100.0
goal_success_rate = goal_cnt / n
```

因此：

| 字段名（JSON / TSV） | 实际含义 | 能否当作「任务成功」 |
|---|---|---|
| `stage_score_pct` / `macro_stage_score_pct` | 每个 trial 的阶段进度百分比；任务内平均后再跨任务宏平均 | 否；是**平均阶段进度** |
| `goal_success_rate` / `macro_goal_success_rate` | 就是 `stage_pct/100` 的均值 | **否**；与阶段进度同构，不是严格 0/1 目标成功 |
| `stage_success_rate` | 评估器诊断的「全部必需阶段完成」比例（若写入） | 接近「全阶段完成率」；须核对 `diagnostics.stage_success` |
| 真值最终目标成功 | 依赖 BDDL / `goal_success` 诊断是否在该任务上可靠 | 有则单列；无则标「评估器未提供」 |

旧 09-18 报告把 `stage_score_pct` 当主分、并把 `goal_success_rate` 并列表出是一致的（二者只差 ×100），但**不能**再把后者口头叫成 CSR 而不解释。

## 2. 新报告必须分列的指标

1. **平均阶段进度（主指标，兼容旧表）**  
   每个 trial 的 `stage_pct` → 任务内均值 → 跨任务等权宏平均。单位：百分比点（pp）。

2. **全部必需阶段完成率**  
   `#{stage_success=true 的有效 trial} / #{有效 trial}`。失败/中断 trial 的计入规则见下。

3. **经核实的最终任务目标成功率**（若有）  
   仅当 episode 诊断明确给出最终 goal 布尔值且与离线 BDDL 复核一致时使用。

4. **机制指标**（记忆臂）  
   工具调用次数、每次调用返回的观测帧数、空返回 / `recency_fallback`、关键帧库容量、`frame 0/1` 占比等。机制指标与分数分开写。

## 3. 失败 / 中断 trial 的计入（跑前冻结）

| 情况 | 阶段进度 | 完成率分母 | 备注 |
|---|---|---|---|
| 正常结束 | 用评估器 `stage_pct` | 计入 | — |
| 早停（配额耗尽 / 死 planner） | **该 trial 记 0**，并在报告「异常」列标原因 | 计入分母 | 不得从均值中静默丢弃 |
| 作业崩溃且无任何 episode 产物 | 整臂标 `FAILED`；不进主表均值 | — | 保留 `run.log` |
| H0 下 retry | **不应发生**（`HARNESS_MAX_RETRIES=0`） | — | 若出现 attempt>0，预检/事后审计失败 |

Hlegacy 下允许 retry：主表报「首轮」与「best-of-retries」须分列；旧 09-18 数字是 best-of-retries 口径，与 H0 不可直接比。

## 4. 与旧文档数字的关系

| 旧说法 | 应对 |
|---|---|
| dual_track「CSR +13.4 / +8.2 / +8.6」 | 本地 PrediMem 线；指标名沿用当时 CSR=阶段进度均值；**不与 API 线混算** |
| REPRODUCIBILITY「+5.3 pp CSR」 | Tier II 另一配置；与 +13.4 **不同来源** |
| 09-18「goal_success_rate」 | = `stage_pct/100`；新文写作「归一化阶段进度（旧字段名 goal_success_rate）」 |
