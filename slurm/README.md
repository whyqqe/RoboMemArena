# Slurm job scripts

Batch scripts for cluster submission. Adjust `#SBATCH --account`, `--partition`, and resource limits for your site.

The repository root is resolved automatically from each script’s location (`ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"`).

Logs: `outputs/slurm/%x-%j.out` (not tracked in git).

---

## `benchmark/`

Official reproduction and **recommended** extension experiments. Documented in [docs/experiments/REPRODUCIBILITY.md](../docs/experiments/REPRODUCIBILITY.md).

| Script | Tier | Description |
|--------|------|-------------|
| `reproduce_all26_1seed.sbatch` | I | PrediMem on all 26 tasks, 1 trial per task |
| `reproduce_task1.sbatch` | I | Single-task reproduction |
| `harness_main_p0.sbatch` | II | Baseline vs. `v21_best` / `harness_v21` |
| `harness_dual_track.sbatch` | III | Decoupled memory vs. harness ablation |

Typical resources: 2 GPUs, 160 GB RAM, 6–12 hours wall time.

---

## `supplementary/`

Development smoke tests and extended ablations. See [docs/supplementary/README.md](../docs/supplementary/README.md).
