# Supplementary experiments

Material in this directory supports **development and extended ablations**. It is not part of the primary reproducibility suite documented in [experiments/REPRODUCIBILITY.md](../experiments/REPRODUCIBILITY.md).

---

## Slurm jobs (`slurm/supplementary/`)

| Script | Description |
|--------|-------------|
| `harness_abtest_8tasks.sbatch` | Early single-seed baseline vs. harness v2 comparison |
| `harness_overnight_ablations.sbatch` | Multi-variant mechanism ablation (retry-only, ctx-only, etc.) |
| `harness_v21_eval.sbatch` | Single-configuration v21 evaluation |
| `smoke_task1.sbatch` | Task 1 smoke test after environment setup |
| `test_api_proxy.sbatch` | External API planner connectivity (optional; disabled by default) |

---

## Reports (`reports/`)

| Report | Description |
|--------|-------------|
| [harness_v2_ab_report_529824.md](reports/harness_v2_ab_report_529824.md) | Single-seed harness v2 A/B (large CSR gain; high variance) |
| [harness_v21_design.md](reports/harness_v21_design.md) | Composition notes for v21 configuration |
| [memory_system_design.md](reports/memory_system_design.md) | Early memory_plus design notes (superseded by main design doc) |

---

## Additional harness variants

`scripts/run_harness_variant.sh` also defines variants used in supplementary ablations:

| Variant | Role |
|---------|------|
| `v2_full` | Full harness v2 baseline |
| `memer_kf` | Denser keyframes + full harness |
| `ctx_only` | VLM context without retry |
| `retry_only` | Retry without VLM context |
| `override_only` | Stall override only |
| `stall_fast` | Reduced stall threshold |
| `v21_combo` | v21 without per-task retry skip list |
| `v1_hints` | Negative control: harness text in VLA prompt |

---

## Comparison scripts

| Script | Purpose |
|--------|---------|
| `scripts/compare_harness_abtest.sh` | Compare early A/B output directories |
| `scripts/compare_harness_matrix.sh` | Multi-variant results matrix |
