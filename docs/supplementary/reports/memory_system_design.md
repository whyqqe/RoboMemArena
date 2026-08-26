# 可优化 Memory 系统设计（实践向）

## 目标

在不立刻微调模型权重的前提下，构建 **可搜索超参、可消融、可积累** 的长期记忆系统，并与外接 Harness 执行层解耦评测。

## 三层记忆（对应 MemoryVLA / MemER）

| 层 | 内容 | 实现 | 可优化超参 |
|----|------|------|------------|
| **Working** | recent N 帧 + 当前 subtask | PrediMem `N_RECENT` | `N_RECENT` |
| **Episodic visual** | VLM 提名 keyframe → 聚类 | `build_visual_memory` / `merge_keyframe_bank` | `MEM_CLUSTER_D`, `MEM_BANK_MAX`, `K_MAX` |
| **Episodic semantic** | stage 完成、stall、subtask 轨迹 | Harness `ExternalMemory` → VLM context | global rules、检索 top-k |

## 本次实现：`memory_plus`

- **Stage anchor**：stage 完成时 `pin_keyframe(step)`，聚类时优先保留（跨时段地标）
- **Salience**：subtask 切换时标记 salient step（MemER 提名扩展）
- **Merge bank**：提名 + pin + salient 再聚类，cap `MEM_BANK_MAX`

## 实践中如何「训练/优化」（无需梯度）

1. **超参搜索**：`MEM_CLUSTER_D`, `MEM_BANK_MAX`, `N_RECENT`, `HARNESS_STALL_STEPS` on 8-task 子集
2. **规则优化**：`global_rules.json`、stage→primitive 映射
3. **结构消融**：working only / +episodic / +anchor（本实验）
4. **后续**：用成功 episode 的 pin/salient 轨迹做 VLM LoRA（学提名），或蒸馏 API planner

## 与外接 Harness 分工

- **Memory 轨**：只改 planner 输入（`memory_kf`, `memory_ctx`, `memory_plus`），`MAX_RETRIES=0`
- **Harness 轨**：`harness_exec`（无 VLM context）、`harness_v21`（完整）
