# Harness and long-horizon memory system design

This document describes the inference-time extension layered on fixed PrediMem weights (Qwen3-VL planner + π0.5 VLA). **Long-horizon memory modules** primarily modify VLM planner inputs; the **external execution harness** handles recovery, episodic semantic memory, and multi-attempt scheduling. Both can be enabled independently or combined.

---

## 1. Design principles

| Principle | Description |
|-----------|-------------|
| **Planner / executor separation** | Semantic memory is injected into the **VLM planner** by default; the VLA receives a clean subtask prompt without harness prose |
| **Inference-time enhancement** | No weight updates; gains come from hyperparameters, rules, episode control, and read-time context |
| **Decoupled evaluation** | Memory configurations disable retry; harness configurations can disable VLM context and denser keyframes for attribution |
| **Practical optimization** | Keyframe density, clustering distance, stall thresholds, retry policy, and global rules are searchable without gradients |

**Pipeline:** observations → VLM (historical keyframes + recent window + optional text memory) → primitive subtask → VLA execution → harness monitors stages / stall → optional subtask override or episode retry.

---

## 2. Related work and baseline stack

### 2.1 Baseline

- **PrediMem:** dual-system stack with built-in keyframe memory (`J_hist`, sparse clustering, `K_MAX` / `D_MERGE`).
- **RoboMemArena:** LIBERO-based evaluation with CSR/TSR and multi-stage predicates.

### 2.2 Mapping to prior systems

| Component | Primary reference | Implementation in this repo |
|-----------|-----------------|------------------------------|
| Visual episodic memory | MemER | VLM keyframe nomination + 1D temporal clustering |
| Long-term bank consolidation | MemoryVLA (simplified) | Stage anchor, salience tags, `merge_keyframe_bank` |
| Semantic episodic + read-time reasoning | MemER / MemoryVLA | `ExternalMemory` + `memory_reason` text for VLM |
| Execution recovery | HarnessVLA | Stall detection, retry, checkpoint, `release_gripper`, subtask override |

**Differences from full paper systems:** no trainable memory bank (no MemoryVLA PCMB); minimal analytic primitive library; API planner disabled by default.

---

## 3. Memory system architecture

Three layers target **VLM planner inputs**: working → episodic visual → episodic semantic.

### 3.1 Working memory

- **Content:** `N_RECENT` consecutive frames (default 7), main and optional wrist cameras; last frame is current observation.
- **Role:** infer the active primitive; presented separately from historical keyframes in the prompt.
- **Tunable:** `N_RECENT`.

### 3.2 Episodic visual memory

Most effective memory lever in evaluation: **nominate → compress → inject at read time**.

#### Nomination

VLM outputs JSON with `current_primitive` and `keyframe_positions` (1-indexed within the recent window). Positions are converted to absolute step indices and appended to `J_hist`.

#### Temporal clustering (PrediMem / MemER)

On all nominations in `J_hist`:

1. Sort and deduplicate indices
2. Cluster with gap ≤ `D_MERGE`
3. Select median index per cluster
4. Drop indices inside the recent window

Baseline: sparse settings (`K_MAX=0`, `D_MERGE=6`).

#### Denser keyframes (`memory_kf`)

| Parameter | Typical | Effect |
|-----------|---------|--------|
| `K_MAX` | 8 | Cap on stored keyframes |
| `D_MERGE` | 4 | Tighter clustering |
| `N_RECENT` | 7 | Recent window length |

When exceeding `K_MAX`, pinned (stage-anchored) frames are retained first.

#### Stage anchor and merge bank (`memory_plus`)

| Mechanism | Trigger | Action |
|-----------|---------|--------|
| Stage anchor | Stage completes | Pin stage-start step in `pinned_keyframe_steps` |
| Salience | Subtask changes | Mark current step in `salient_keyframe_steps` |
| Merge bank | After each VLM call | Merge nominations + pins + salient steps; re-cluster |

Merge rules: prefer earliest pinned step per cluster; cap at `MEM_BANK_MAX`; tunables `MEM_STAGE_ANCHOR`, `MEM_SALIENCE_SUBTASK`, `MEM_CLUSTER_D`.

#### Read-time injection

Compressed keyframe images are loaded from `frame_store` into the VLM message as “Historical keyframes” before the recent window.

### 3.3 Episodic semantic memory

When `HARNESS_VLM_CONTEXT=1`, harness `ExternalMemory` produces text in `harness_extra_context`, inserted as a separate VLM message block. No images; structured events and evidence (see Section 4). Used by `memory_ctx` and combined configs.

---

## 4. External harness architecture

Episode-level controller around PrediMem: monitoring, memory, recovery, multi-attempt scheduling—no weight changes.

### 4.1 Modules

| Module | Responsibility |
|--------|----------------|
| `HarnessConfig` | Environment-driven flags and thresholds |
| `HarnessController` | Attempt lifecycle orchestration |
| `ExternalMemory` | Structured episodic store across steps and attempts |
| `memory_reason` | Format memory for VLM; token-based evidence retrieval |
| `stage_mapper` | Heuristic stage → primitive mapping on stall |
| `api_planner` (optional) | OpenAI-compatible recovery planner |
| `primitives` | Analytic actions (currently `release_gripper` before retry) |

### 4.2 ExternalMemory schema

| Field | Description |
|-------|-------------|
| `global_rules` | Task-level long-horizon rules |
| `working` | Recent subtask deque (~12 entries) |
| `episodic` | Events: subtask change, stage complete, stall, retry |
| `episode_evidence` | Structured evidence records |
| `attempts` / `best_partial` | Per-attempt scores and best partial progress |

Optional JSON persistence per task/attempt for analysis.

### 4.3 Read-time reasoning (`memory_reason_for_planner`)

Assembles global rules, cross-attempt progress, retrieved evidence, stall context, recovery suggestions, and retry reminders—for **VLM only**.

### 4.4 Stall detection and recovery

`on_stage_progress` tracks stage index. If unchanged for ≥ `HARNESS_STALL_STEPS`, trigger stall handling: stage mapping, optional API planner, refresh VLM context, optional `subtask_override`, optional forced VLM replan.

### 4.5 Episode retry and checkpoint

Up to `1 + HARNESS_MAX_RETRIES` attempts per trial. Non-first attempts may release the gripper and inject resume context from `best_partial`. `should_retry` respects smart retry thresholds, progress requirements, and per-task retry skip lists (e.g. tasks 18, 22). Trial score uses the best attempt by stage score.

### 4.6 VLM / VLA boundary

| Flag | Default (combined) | Effect |
|------|-------------------|--------|
| `HARNESS_VLM_CONTEXT` | 1 | Inject semantic memory into VLM |
| `HARNESS_VLA_HINTS` | 0 | Do not modify VLA prompt |
| `HARNESS_SUBTASK_OVERRIDE` | 1 | Direct subtask override on stall |

---

## 5. Evaluation loop integration

Per simulation step: refresh harness VLM context → submit async VLM job → apply subtask override if set → query VLA → update stages (pin keyframe if anchor enabled) → harness stall check. After episode: record attempt, optionally retry with `env.reset`.

---

## 6. Standard configuration presets

| Preset | Harness | Visual memory | VLM semantic | Retry |
|--------|---------|---------------|--------------|-------|
| baseline | off | sparse | — | — |
| `memory_kf` | off | denser KF | — | no |
| `memory_ctx` | memory only | denser KF | yes | no |
| `memory_plus` | memory only | denser KF + anchor | yes | no |
| `harness_exec` | execution only | sparse | no | yes |
| `harness_v21` | full | denser KF | yes | yes (smart) |

**Recommended:** `harness_v21` — ~+13.4 pp CSR vs. baseline (8 tasks, 5 seeds, paired comparison).

---

## 7. Environment variable reference

### Memory / keyframes

`VLM_USE_KEYFRAME_MEMORY`, `K_MAX`, `D_MERGE`, `N_RECENT`, `MEM_STAGE_ANCHOR`, `MEM_SALIENCE_SUBTASK`, `MEM_BANK_MAX`, `MEM_CLUSTER_D`

### Harness

`HARNESS_ENABLE`, `HARNESS_MAX_RETRIES`, `HARNESS_STALL_STEPS`, `HARNESS_VLM_CONTEXT`, `HARNESS_VLA_HINTS`, `HARNESS_SUBTASK_OVERRIDE`, `HARNESS_STAGE_CHECKPOINT`, `HARNESS_RELEASE_ON_RETRY`, `HARNESS_SMART_RETRY`, `HARNESS_RETRY_SKIP_SCORE`, `HARNESS_RETRY_REQUIRE_PROGRESS`, `HARNESS_SKIP_RETRY_TASKS`, `HARNESS_FORCE_VLM_REPLAN`, `HARNESS_API_PLANNER`, `HARNESS_GLOBAL_RULES`

Apply via `scripts/run_harness_variant.sh <preset>`.

---

## 8. Empirical design validation

From the decoupled eight-task, five-seed study:

| Design choice | Observation | Implication |
|---------------|-------------|-------------|
| Denser visual memory | +8.2 pp; 5/5 seeds positive | Primary memory lever; tune `K_MAX` / `D_MERGE` |
| Episodic text only | +4.0 pp; below visual-only | Semantic layer needs tighter, task-aware content |
| Stage anchor bank | +5.9 pp; below `memory_kf` | Current anchor implementation needs revision |
| Execution harness only | +8.6 pp without VLM memory | Orthogonal, comparable gain |
| Combined `harness_v21` | +13.4 pp CSR, +10.0 pp TSR | Planner and executor mechanisms complement |
| Retry on task 22 without skip | 100% → 53% mean | Per-task retry policy required |
| `HARNESS_VLA_HINTS=0` | Stable gains vs. polluted VLA prompt | Keep memory on planner side |

---

## 9. Limitations and future work

- Rule-based visual bank; no learned consolidation (vs. full MemoryVLA).
- Minimal analytic primitive set (vs. full HarnessVLA).
- Token-match semantic retrieval; no embedding index.
- API planner optional and often disabled on air-gapped clusters.
- Results on eight-task subset with meaningful variance.

Future: hyperparameter search, global rule optimization, improved semantic summaries, task-adaptive retry, optional lightweight trainable memory or VLM nomination distillation.

---

## 10. Summary

The memory extension adds working, visual episodic, and semantic episodic inputs to the VLM planner; **denser visual keyframes** deliver the largest isolated gain. The harness provides episodic storage, stall recovery, and multi-attempt scheduling in the HarnessVLA tradition. **`harness_v21`** combines both under the principle that memory enriches the planner while the VLA prompt remains clean.
