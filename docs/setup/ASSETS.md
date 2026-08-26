# Assets: checkpoints and dataset

## Overview

| Asset | Required for simulation eval | Default location | Source |
|-------|------------------------------|------------------|--------|
| **PrediMem weights** | Yes (reference stack) | `checkpoints/PrediMem/` | [Hugging Face: huashuolei/PrediMem](https://huggingface.co/huashuolei/PrediMem) |
| **RoboMemArena HDF5 dataset** | No (training / RLDS only) | `data/RoboMemArena/` | [HF dataset](https://huggingface.co/datasets/RoboMemArenaBenchmark/RoboMemArena) |
| **BDDL task specs** | Yes | `bddl/`, `evaluation_benchmark/bddl/` | Bundled in repository |
| **LIBERO fork** | Yes | `evaluation_benchmark/libero_fork/` | Bundled in repository |

Large directories are listed in `.gitignore`. Clone users must download weights locally.

---

## One-command download

```bash
source scripts/activate_eval.sh
bash scripts/download_assets.sh
```

The script:

1. Downloads the RoboMemArena dataset to `data/RoboMemArena/` (falls back to ModelScope if Hugging Face fails)
2. Downloads PrediMem at revision `e645741c2f34f27e1596bfa89e856d6f3560ed90` to `checkpoints/PrediMem/`

Override paths:

```bash
export DATA_DIR=/path/to/data/RoboMemArena
export CKPT_DIR=/path/to/checkpoints/PrediMem
export PREDIMEM_REVISION=e645741c2f34f27e1596bfa89e856d6f3560ed90
bash scripts/download_assets.sh
```

### Hugging Face authentication

For gated repositories:

```bash
huggingface-cli login
```

Cache directory: `HF_HOME` (default under `CACHE_ROOT` or `.cache`).

---

## PrediMem checkpoint layout

After download, `checkpoints/PrediMem/` should contain:

| Path | Role |
|------|------|
| `vla_alltask/` | π0.5 VLA policy (shared across tasks) |
| `vla_alltask/assets/policy_assets/norm_stats.json` | Action normalization |
| `vlm_task1/` | VLM checkpoint for task 1 |
| `vlm_tasks1to26_ckpt74500/` | VLM checkpoint for tasks 2–26 |

Reference evaluation selects VLM weights automatically: task 1 → `vlm_task1`; tasks 2–26 → `vlm_tasks1to26_ckpt74500`.

```bash
export PREDIMEM_HF_SNAPSHOT="${ROBOMEMARENA_ROOT}/checkpoints/PrediMem"
```

---

## RoboMemArena dataset

### Release version

Use the dataset release **on or after June 20, 2026 (v2)**. Earlier snapshots differ in annotations, task 6 logic, and scenes for tasks 1–3. Replace old downloads entirely rather than merging.

### Layout (after download)

```
data/RoboMemArena/
├── <category>/
│   └── <task_folder>/
│       ├── full_trajectory/      # Full-episode HDF5
│       └── subtask_data/         # Subtask HDF5 (common for training)
```

Filename pattern: `<primitive>_<subtask_order>_seed<seed>_task<task_id>.hdf5`.

### When to download

| Use case | Download HDF5? |
|----------|----------------|
| PrediMem / LIBERO CSR/TSR evaluation | **No** |
| RLDS conversion or training | **Yes** |
| Single-seed training subset | `scripts/download_dataset_one_seed.sh` |

---

## Bundled repository assets (no download)

- Task BDDL definitions (`bddl/`)
- LIBERO-compatible simulator (`evaluation_benchmark/libero_fork/`)
- OpenPI minimal runtime (`third_party/openpi_minimal/`)
- Harness and memory modules (`evaluation_benchmark/harness/`, `memory_system/`)

---

## Disk usage (approximate)

| Asset | Size |
|-------|------|
| PrediMem weights | Tens of GB |
| Full RoboMemArena dataset | Large (plan 100 GB+ if storing everything) |
| Per-evaluation `outputs/` | Several GB when videos are enabled |

---

## Verify downloads

```bash
test -d checkpoints/PrediMem/vla_alltask && echo "VLA OK"
test -d checkpoints/PrediMem/vlm_task1 && echo "VLM task1 OK"
test -d checkpoints/PrediMem/vlm_tasks1to26_ckpt74500 && echo "VLM tasks 2-26 OK"
```

Smoke test: `sbatch slurm/supplementary/smoke_task1.sbatch` or local run with `TASKS_JSON='[1]'`.
