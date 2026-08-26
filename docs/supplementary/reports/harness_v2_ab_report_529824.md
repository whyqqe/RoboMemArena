# Harness v2 A/B 实验报告（Job 529824）

**日期**：2026-08-24  
**环境**：dgx-35，PrediMem（VLM task1 / tasks1–26 ckpt74500 + π0.5 VLA）  
**协议**：RoboMemArena async VLM26，1 trial / task，seed=100  

---

## 1. 背景与动机

### 1.1 PrediMem 基线

PrediMem 采用 **双系统架构**：Qwen3-VL 高层 planner 输出 primitive subtask，π0.5 低层 VLA 执行。VLM 侧已有 keyframe memory（`K_MAX`、`D_MERGE`），但在长程、多 stage 任务上仍易出现 stall 与 stage 遗漏。

### 1.2 相关工作映射

| 工作 | 核心思想 | 本实验 Harness v2 对应 |
|------|----------|-------------------------|
| **MemER** (Pan et al., ICLR 2026) | 高层 VLM 从 recent frames + 检索 keyframes 生成 subtask；1D 聚类压缩 episodic memory | `memory_reason` + VLM context 注入；PrediMem 原生 keyframe buffer |
| **MemoryVLA** (Shi et al., ICLR 2026) | Working memory + PCMB 长期库；检索-融合-巩固 | 外接 `ExternalMemory`（working + episodic + global rules） |
| **HarnessVLA** (why/HarnessVLA) | Planner 不变，外接 memory + stall recovery + analytic primitives | `HarnessController`、retry、subtask override、`release_gripper` |
| **PrediMem / RoboMemArena** | 评测 26 个 memory-heavy LIBERO 变体 | 本报告 8-task 子集 A/B |

**设计原则（v2）**：Memory 回灌 **VLM Planner**，**不污染 VLA prompt**（π0.5 未在 harness prose 上训练）。

---

## 2. 方法：Harness v2

### 2.1 模块

```
VLM (PrediMem)  ←── harness.get_vlm_context()  [MemoryVLA/MemER 式 read-time context]
       ↓ subtask
VLA (π0.5)      ←── 干净 prompt（HARNESS_VLA_HINTS=0）
       ↓
HarnessController: stall 检测 → subtask override / force replan
                   attempt 结束 → cross-attempt memory（stage checkpoint）
                   retry 前 → release_gripper()
```

### 2.2 关键环境变量

| 变量 | v2 值 | 含义 |
|------|-------|------|
| `HARNESS_VLM_CONTEXT` | 1 | episodic + global rules 注入 VLM messages |
| `HARNESS_VLA_HINTS` | 0 | 不修改 VLA prompt |
| `HARNESS_SUBTASK_OVERRIDE` | 1 | stall 时 stage→primitive 映射 override |
| `HARNESS_STAGE_CHECKPOINT` | 1 | retry 保留 partial progress memory |
| `HARNESS_MAX_RETRIES` | 2 | 最多 3 次 attempt |
| `HARNESS_STALL_STEPS` | 120 | stall 阈值 |
| `HARNESS_API_PLANNER` | 0 | 集群无外网 API，使用本地 heuristics |

### 2.3 评测子集

8 个任务：`[1, 4, 5, 11, 14, 16, 18, 22]`  
覆盖 basket、multi-drawer、counting pour、cabinet 等 memory-heavy 类型。

---

## 3. 实验结果

### 3.1 宏观指标

| 指标 | Baseline | Harness v2 | Δ |
|------|----------|------------|---|
| **TSR**（stage 全完成率） | 12.5% (1/8) | **37.5% (3/8)** | **+25.0 pp** |
| **CSR**（goal 完成率） | 37.9% | **71.1%** | **+33.2 pp** |
| **Stage Score** | 37.9% | **71.2%** | **+33.3 pp** |

输出目录：
- Baseline：`outputs/harness_ab_baseline_529824/`
- Harness：`outputs/harness_ab_harness_529824/`

### 3.2 逐任务 CSR / TSR

| Task | 类型 | Base CSR | Harness CSR | Δ CSR | Base TSR | Harness TSR |
|------|------|----------|-------------|-------|----------|-------------|
| 1 | basket | 0% | **100%** | +100% | 0 | **1** |
| 4 | multi-drawer | 0% | 50% | +50% | 0 | 0 |
| 5 | multi-drawer | 50% | 12.5% | **-37.5%** | 0 | 0 |
| 11 | multi-drawer | 0% | **80%** | +80% | 0 | 0 |
| 14 | multi-drawer | 20% | **60%** | +40% | 0 | 0 |
| 16 | pour | 66.7% | 66.7% | 0 | 0 | 0 |
| 18 | cabinet | 100% | 100% | 0 | 1 | 1 |
| 22 | pour | 66.7% | **100%** | +33.3% | 0 | **1** |

### 3.3 与 v1 Harness（529751）对比

| 配置 | TSR | CSR |
|------|-----|-----|
| v1 Harness（VLA hint 污染） | 37.5% | 58.3% |
| **v2 Harness** | 37.5% | **71.1%** |

TSR 相同，CSR 提升 **+12.8 pp**，说明 v2「只改 planner、不改 VLA」更有效。

---

## 4. 机制分析

### 4.1 成功案例

**Task 1（0% → 100%）**  
- attempt0/1 失败后，attempt2 在 stall 时被 override 为 `place cookies into basket`  
- 完成 `01_Place_Cookies_Basket` → `02_Place_Tomato_Basket`  
- 验证：**retry + subtask override + VLM context** 组合有效

**Task 11（0% → 80%）**  
- 3 次 attempt，attempt2 通过 override（place cookies / close drawer / open middle drawer）推进至 5/6 stage

**Task 22（66.7% → 100%）**  
- stall 时 forced VLM replan，三 stage 全完成

### 4.2 失败与退步

**Task 5（50% → 12.5%）**  
- Baseline 偶然打开 top drawer；Harness 3 次 retry 全量 reset 后丢失进度  
- best-partial 选取保留 attempt1 的 12.5%，说明 **retry 策略需「有进展才 retry」**

**Task 16（无改善）**  
- 3 次 attempt 均卡在 `03_Pour_Two`；末 stage 瓶颈非 memory  alone 可解

### 4.3 运行时开销

Harness 单 task 耗时约为 baseline 的 **2–3×**（retry + 更长 episode），例如 task4：102s → 280s。

---

## 5. 结论

1. **Harness v2 在 8-task 子集上显著优于 Baseline**（CSR +33 pp，TSR +25 pp）。
2. **架构方向正确**：Memory → VLM、VLA prompt 保持干净，优于 v1 污染 VLA。
3. **主要收益来自 memory-heavy 多 stage 任务**（1、11、14、22）。
4. **未解决问题**：retry 有害案例（task5）、末 stage 瓶颈（16）、外部 API planner（集群网络）。

---

## 6. 后续实验（过夜流水线）

见 `slurm/harness_overnight_ablations.sbatch`，对 12 个 memory-heavy 任务做机制消融：

| Variant | 对应研究 | 测试假设 |
|---------|----------|----------|
| `baseline` | — | 下界 |
| `v2_full` | HarnessVLA + MemER | 完整 v2 |
| `memer_kf` | MemER | 加强 keyframe（K_MAX=8） |
| `ctx_only` | MemER read-time | 仅 VLM memory，无 retry/override |
| `retry_only` | HarnessVLA exec | 仅 retry/checkpoint，无 memory context |
| `override_only` | analytic recovery | 仅 stall override |
| `stall_fast` | sensitivity | stall=80 |
| `v1_hints` | negative control | VLA hint 污染 |

---

## 附录：复现命令

```bash
cd /project/peilab/why/RoboMemArena
sbatch slurm/harness_abtest_8tasks.sbatch
bash scripts/compare_harness_abtest.sh \
  outputs/harness_ab_baseline_529824 \
  outputs/harness_ab_harness_529824
```
