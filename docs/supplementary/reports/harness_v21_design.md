# Harness v2.1 设计说明

基于 529824 / 529932 消融，组合各机制优势：

| 来源 | 机制 | v2.1 实现 |
|------|------|-----------|
| **MemER** | 高层检索 keyframe，短窗 + 压缩 episodic | `K_MAX=8`, `D_MERGE=4`, `N_RECENT=7` |
| **MemoryVLA** | working + episodic 读回 planner | `HARNESS_VLM_CONTEXT=1`（外接 episodic + rules） |
| **HarnessVLA** | retry / checkpoint / release | `MAX_RETRIES=2`, checkpoint, release_gripper |
| **529932** | 更早 stall 恢复 | `HARNESS_STALL_STEPS=80` |
| **529932 教训** | retry 破坏已高分 episode | `HARNESS_SMART_RETRY=1`, `RETRY_SKIP_SCORE=95` |
| **v2 原则** | 不污染 VLA | `HARNESS_VLA_HINTS=0` |

**不采用**：单独 subtask override（override_only）、单独 ctx_only、VLA hint 污染。

# P0 main: `v21_best` = v21_combo + skip retry on task 18/22 + require progress to retry
