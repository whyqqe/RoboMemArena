# Decoupled evaluation: external harness and long-horizon memory

Under fixed PrediMem weights (Qwen3-VL + π0.5 VLA, no fine-tuning), a decoupled experiment suite validates:

1. Whether long-horizon **memory modules** improve performance independently
2. Whether an external **execution harness** improves performance independently
3. Whether the **combined** configuration outperforms either alone

---

## 1. Research questions

| ID | Question |
|----|----------|
| RQ1 | Do memory modules alone improve CSR when episode retry is disabled? |
| RQ2 | Does the execution harness alone improve CSR without VLM memory injection or denser keyframes? |
| RQ3 | Does the combined configuration exceed the best single-track result? |

Prior single-seed comparisons reported large CSR gains with unknown variance; a five-seed paired study provides attributable evidence.

---

## 2. Experimental design

### 2.1 Decoupling protocol

From a shared PrediMem baseline, two tracks isolate planner-side vs. executor-side changes:

- **Memory track:** Modifies VLM planner inputs (keyframe density, episodic text, stage anchors). Episode retry is disabled (`MAX_RETRIES=0`).
- **Harness track:** Modifies execution recovery (stall detection, retry, checkpoint, gripper release). VLM episodic injection is disabled (`HARNESS_VLM_CONTEXT=0`); default sparse keyframes (`K_MAX=0`).
- **Combined:** Enables recommended settings from both tracks (`harness_v21`).

### 2.2 Paired comparison

Each LIBERO seed runs baseline first, then variants. Primary metric:

**ΔCSR = variant_CSR − baseline_CSR** (same seed)

Secondary metrics: TSR, count of seeds with positive ΔCSR.

### 2.3 Protocol

| Parameter | Value |
|-----------|-------|
| Tasks | 1, 4, 5, 11, 14, 16, 18, 22 |
| Seeds | 100, 101, 102, 103, 104 |
| Trials per task | 1 |
| Model | Fixed PrediMem checkpoints; inference-only changes |
| VLA prompt | `HARNESS_VLA_HINTS=0` throughout |
| API planner | Disabled; local heuristics |

---

## 3. Configurations

### Baseline

PrediMem default sparse keyframes (`K_MAX=0`, `D_MERGE=6`); harness disabled.

### Memory track (no retry)

| Config | Key settings | Intent |
|--------|--------------|--------|
| `memory_kf` | Harness off; `K_MAX=8`, `D_MERGE=4` | Denser visual episodic memory only |
| `memory_ctx` | Denser KF + VLM text context; `MAX_RETRIES=0` | Adds semantic episodic injection |
| `memory_plus` | Above + stage anchor, salience, merge bank | Structured long-term visual bank |

### Harness track

| Config | Key settings | Intent |
|--------|--------------|--------|
| `harness_exec` | No VLM context; retry + checkpoint; default KF | Execution recovery only |
| `harness_v21` | Full harness + denser KF + smart retry | Recommended combination |

---

## 4. Results (5-seed mean ± std)

| Configuration | Track | CSR | ΔCSR | TSR | ΔTSR | Seeds ΔCSR > 0 |
|---------------|-------|-----|------|-----|------|----------------|
| baseline | — | 44.1% ± 9.6% | — | 22.5% ± 5.6% | — | — |
| memory_kf | Memory | 52.4% ± 10.4% | **+8.2 pp** | 27.5% | +5.0 pp | **5/5** |
| memory_ctx | Memory | 48.1% ± 10.7% | +4.0 pp | 25.0% | +2.5 pp | 3/5 |
| memory_plus | Memory | 50.0% ± 7.6% | +5.9 pp | 27.5% | +5.0 pp | 4/5 |
| harness_exec | Harness | 52.7% ± 7.0% | **+8.6 pp** | 27.5% | +5.0 pp | 4/5 |
| harness_v21 | Combined | **57.5% ± 8.2%** | **+13.4 pp** | **32.5%** | **+10.0 pp** | 4/5 |

Per-seed baseline CSR: 34.6%, 59.3%, 38.3%, 41.5%, 47.1%.

### Per-seed ΔCSR

| Seed | baseline | memory_kf | memory_ctx | memory_plus | harness_exec | harness_v21 |
|------|----------|-----------|------------|-------------|--------------|-------------|
| 100 | 34.6% | +5.2 | +3.8 | +8.8 | +10.0 | +13.8 |
| 101 | 59.3% | +5.6 | −5.1 | −10.9 | −10.3 | 0.0 |
| 102 | 38.3% | +15.9 | +13.6 | +6.3 | +21.5 | +29.7 |
| 103 | 41.5% | +2.5 | −5.8 | +10.0 | +8.4 | +8.8 |
| 104 | 47.1% | +11.9 | +13.4 | +15.3 | +13.2 | +14.8 |

### Per-task success rate (5-seed mean)

| Task | baseline | memory_kf | memory_ctx | memory_plus | harness_exec | harness_v21 |
|------|----------|-----------|------------|-------------|--------------|-------------|
| 1 | 20% | 40% | 40% | 50% | 60% | 60% |
| 4 | 8% | 25% | 15% | 5% | 15% | 12% |
| 5 | 5% | 10% | 10% | 2% | 20% | 22% |
| 11 | 32% | 48% | 44% | 52% | 44% | 60% |
| 14 | 32% | 36% | 36% | 44% | 56% | 52% |
| 16 | 67% | 67% | 60% | 67% | 73% | 73% |
| 18 | 90% | 100% | 100% | 100% | 100% | 100% |
| 22 | 100% | 93% | 80% | 80% | 53% | 80% |

Task 22: baseline is already 100%; `harness_exec` without per-task retry skip drops to 53% mean success.

---

## 5. Conclusions

### RQ1 — Memory modules

`memory_kf` provides the cleanest evidence: **+8.2 pp CSR**, positive on all five seeds, with harness and retry disabled. The primary lever is **denser visual episodic memory** (MemER-style keyframe densification), not episodic text or the current stage-anchor implementation (`memory_plus` did not exceed `memory_kf`).

### RQ2 — Execution harness

`harness_exec` yields **+8.6 pp CSR** without VLM memory or denser keyframes, comparable to the memory track. Gains are driven by episode-level recovery (stall, retry, checkpoint). Retry can harm near-saturated pour tasks; per-task retry policy is required.

### RQ3 — Combined configuration

`harness_v21` at **+13.4 pp CSR** and **+10.0 pp TSR** exceeds either single track (~+8 pp), indicating complementary planner-side and executor-side mechanisms.

---

## 6. Recommended deployment

| Priority | Configuration | ΔCSR (reference) | Use case |
|----------|---------------|------------------|----------|
| 1 | `harness_v21` | +13.4 pp | Default production setting |
| 2 | `harness_exec` | +8.6 pp | Execution recovery without memory injection |
| 3 | `memory_kf` | +8.2 pp | Memory-only when retry is unavailable |

```bash
source scripts/run_harness_variant.sh harness_v21
```

---

## 7. Limitations

- High per-seed variance; conclusions require multi-seed paired comparison.
- Eight-task subset; not all 26 benchmark tasks.
- `memory_plus` underperformed `memory_kf` in this study.
- Extension results are inference-time only; no weight updates.
