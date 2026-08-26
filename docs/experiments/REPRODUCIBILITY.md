# Reproducibility guide

This document defines the **recommended experiment suite** for reproducing official benchmark numbers and validating the PrediMem harness/memory extension.

Architecture and configuration details: [design/harness_memory_system_design.md](../design/harness_memory_system_design.md).

Extended ablations and development jobs: [supplementary/README.md](../supplementary/README.md).

---

## Experiment tiers

| Tier | Name | Purpose |
|------|------|---------|
| **I** | Benchmark reproduction | Align with official PrediMem CSR/TSR on tasks 1–26 |
| **II** | Extension primary comparison | PrediMem baseline vs. recommended harness configuration |
| **III** | Extension ablation study | Isolate memory modules vs. execution harness vs. combined |

---

## Tier I — Benchmark reproduction

**Protocol:** Fixed PrediMem weights, async VLM26 reference runner, CSR and TSR metrics.

| Setting | Value |
|---------|-------|
| Tasks | 1–26 |
| Trials | 1 per task (fast) or 51 per task (paper protocol) |
| Harness | Disabled |

```bash
sbatch slurm/benchmark/reproduce_all26_1seed.sbatch
```

Single-task debug: `slurm/benchmark/reproduce_task1.sbatch` or local `TASKS_JSON='[1]'`.

---

## Tier II — Extension primary comparison

**Protocol:** Eight memory-intensive tasks, five LIBERO seeds, paired comparison against PrediMem baseline.

| Task IDs | 1, 4, 5, 11, 14, 16, 18, 22 |
| Seeds | 100, 101, 102, 103, 104 |
| Variants | `baseline`, `v21_best` (alias: `harness_v21`) |

`v21_best` / `harness_v21` is the **recommended production configuration**: denser keyframe memory, full harness, smart retry, skip retry on tasks 18 and 22.

```bash
sbatch slurm/benchmark/harness_main_p0.sbatch
```

Aggregate multi-seed results:

```bash
bash scripts/aggregate_multiseed_main.sh <OUTPUT_DIRECTORY>
```

**Reference result:** ~+5.3 pp CSR (5-seed mean, paired vs. baseline).

---

## Tier III — Decoupled ablation study

**Protocol:** Same eight tasks and five seeds; each seed runs baseline plus five extension variants to attribute gains to memory vs. execution recovery.

### Memory track (no episode retry)

| Variant | Isolates |
|---------|----------|
| `memory_kf` | Denser visual keyframe memory only |
| `memory_ctx` | Visual memory + VLM episodic text context |
| `memory_plus` | Above + stage anchor and merge bank |

### Execution harness track

| Variant | Isolates |
|---------|----------|
| `harness_exec` | Stall detection, retry, checkpoint (no VLM memory injection) |
| `harness_v21` | Combined memory + harness (recommended) |

```bash
sbatch slurm/benchmark/harness_dual_track.sbatch
```

Aggregate:

```bash
bash scripts/aggregate_split_tracks.sh outputs/harness_dual_track_<JOB_ID>/
```

Full analysis: [dual_track_report.md](dual_track_report.md).

### Local single-variant run

```bash
source scripts/run_harness_variant.sh harness_v21
export TASKS_JSON='[1,4,5,11,14,16,18,22]'
export SEED=100
export OUT_ROOT=outputs/debug_harness_v21_s100
cd evaluation_benchmark/async_vlm26_reference
bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
```

---

## Recommended configuration: `harness_v21`

```bash
source scripts/run_harness_variant.sh harness_v21
```

| Category | Settings |
|----------|----------|
| Visual memory | `K_MAX=8`, `D_MERGE=4`, `N_RECENT=7` |
| Harness | VLM context, subtask override, `HARNESS_STALL_STEPS=80` |
| Retry | Smart retry, progress required, tasks 18/22 excluded |
| VLA | `HARNESS_VLA_HINTS=0` (clean low-level prompt) |

Memory-only lightweight alternative (no retry): `memory_kf`.

---

## Metrics and outputs

| Metric | Definition |
|--------|------------|
| **CSR** | Macro goal success rate across evaluated tasks |
| **TSR** | Macro stage success rate (all stages completed) |
| **ΔCSR** | Per-seed difference vs. baseline (percentage points) |

Each run produces `summary.tsv` and `aggregate.json` under `OUT_ROOT`. Merge multiple task outputs with `scripts/merge_eval_outputs.sh`.

---

## Task subset

Extension experiments use tasks **1, 4, 5, 11, 14, 16, 18, 22** (basket, multi-drawer, counting pour, cabinet, and related memory-intensive variants).
