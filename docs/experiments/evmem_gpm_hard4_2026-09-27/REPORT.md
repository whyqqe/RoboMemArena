# EvMem-GPM hard4 实验报告：相对 `nomem` 基线

- 日期：2026-09-27
- 模型：`EvMem-GPM`（Goal-Predicate Memory；臂名 `evmem`）
- GPM 作业：`615169`（t8）、`615189`（t5）、`615190`（t19）、`615191`（t22）
- nomem 作业：`614243`（t5）、`614797`（t8）、`614799`（t19）、`614801`（t22）
- 结论一句话：在 hard 子集 `{5,8,19,22}` 上，GPM 相对同协议 `nomem`（seed100 / 1×10）四任务宏平均 `stage_score_pct` 从 **31.8 → 41.7**（+9.9）；t8/t19/t5 均抬升，t22 持平。

## 本目录内容

| 路径 | 内容 |
|---|---|
| `REPORT.md` | 本报告 |
| `CODE.md` | 代码快照清单与架构要点 |
| `PROVENANCE.txt` / `SHA256SUMS.txt` / `scoreboard.json` | 打包来源、指纹、分数汇总 |
| `code/` | 产出本报告 GPM 结果的代码快照（9 个文件） |
| `results/evmem_gpm_t{5,8,19,22}_evmem_h0_1x10/` | GPM 1×10（h0 / seed100）完整产物 |
| `results/nomem_baselines/task{5,8,19,22}_nomem_s100/` | 对照 `nomem` seed100 完整产物 |

每个结果目录内含：`summary.tsv` / `summary.json`、`videos/`（失败 episode 主视角与腕部录像）、`code_provenance.json`、轨迹与 harness 日志等。

---

## 1. 实验设计与协议

| 项 | 值 |
|---|---|
| 任务 | hard4：`task5` / `task8` / `task19` / `task22` |
| 臂 | `nomem`（基线）、`evmem`（GPM：在 nomem 上叠加谓词脚手架 / 门控证据 / 硬 C 门禁） |
| `SEED` | 100（`NUM_TRIALS=10`，覆盖 seed 100..109） |
| 评分 | `stage_score_pct`（宏平均，10 trials） |
| 规划后端 | API planner（与 mem_efficacy h0 协议一致） |
| KF bank | 关闭（与 nomem 一致；GPM 不走 keyframe 推送通道） |

对比约定：只比 seed100 / 1×10 单元，避免把 3×10 其他 seed 混入 GPM 对照。

nomem 来源：

| 任务 | 源目录 |
|---|---|
| 5 | `experiments/mem_efficacy/results/task5_h0_controls/h0/nomem_s100` |
| 8 / 19 / 22 | `experiments/mem_efficacy/results/hard3_t{8,19,22}_nomem_h0_3x10/h0/nomem_s100` |

---

## 2. GPM 架构（相对 nomem 的增量）

默认行为 ≈ nomem（fluent）。增量分三路：

| 通道 | 作用 |
|---|---|
| **G** | 谓词脚手架：按 scorer 对齐的 NEED 谓词推进；对 identity / 容器答案键做 redact-safe 处理 |
| **E** | 门控证据：仅在 stall / attempts / attractor 时注入；失败帧打标签并衰减；模板受 admissible 约束 |
| **C** | 硬原语控制器：拦截 off-stage、过早 pour、stage-label / ordinal 粘贴等 |

关键设计约束：`SEEN ≠ VERIFIED`；不写 notes；不打开 KF bank。实现见 `code/memexp_evmem.py` + `code/memexp_evmem_bind.py`，由 `code/arms/evmem.sh` 在 `nomem.sh` 之上启用。

---

## 3. 主结果（`stage_score_pct`，seed100 / 1×10）

| 任务 | nomem | EvMem-GPM | Δ |
|---|---:|---:|---:|
| 5 | 13.8 | 20.0 | +6.2 |
| 8 | 26.7 | 46.7 | +20.0 |
| 19 | 13.3 | 26.7 | +13.4 |
| 22 | 73.3 | 73.3 | 0.0 |
| **宏平均** | **31.8** | **41.7** | **+9.9** |

机器可读汇总：`scoreboard.json`。

相对更早 EvMem 迭代（非本包对照）：GPM t22 持平 nomem 73.3，高于 v1 的 56.7（+16.6）。

---

## 4. 读数与残留失败模式

- **t8**：抬升最大（+20）。计数倾倒族（Lift → Pour_One → Pour_Two）上，C 门禁与 attractor 负反馈抑制了过早 pour / 简报吸引子漂移。
- **t19 / t5**：中等抬升；仍有 lift stall、物体定位失败。
- **t22**：与 nomem 持平（已高基线）；未继续抬升，残留含 `pour_two` / frypan 表述类问题。

本包不做统计显著性声明：每任务仅 1×10，分数差可能含噪声；报告重点是可复现快照与同协议对照。

---

## 5. 复现入口（包内快照）

```bash
# 工作区路径需指向仓库 ROOT；包内脚本默认 ROOT=/project/peilab/why/RoboMemArena
bash code/run_evmem_gpm_t8_1x10.sh            # t8 早停 1×10
bash code/run_evmem_gpm_remaining_1x10.sh      # t5/t19/t22 早停 1×10
# nomem 对照：arms/nomem.sh + mem_efficacy stage1 协议（见 CODE.md）
python code/selftest_evmem.py
```
