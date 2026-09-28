# Documentation

This index covers the RoboMemArena benchmark, the PrediMem reference evaluator, and the inference-time harness/memory extension.

## Getting started

| Guide | Description |
|-------|-------------|
| [Installation](setup/INSTALLATION.md) | Dependencies, dual virtualenv setup, GPU and EGL |
| [Assets](setup/ASSETS.md) | PrediMem checkpoints and dataset download |

## Experiments and design

| Document | Description |
|----------|-------------|
| [Reproducibility](experiments/REPRODUCIBILITY.md) | Official benchmark and recommended extension runs |
| [Harness & memory design](design/harness_memory_system_design.md) | Architecture, configuration, and module reference |
| [Dual-track report](experiments/dual_track_report.md) | Decoupled evaluation of memory vs. execution harness |
| [PrediMem error patterns](experiments/predimem_error_pattern_534999.md) | Trajectory-level failure modes from job 534999 |
| [EvMem-GPM hard4 package](experiments/evmem_gpm_hard4_2026-09-27/REPORT.md) | nomem vs GPM on tasks {5,8,19,22} (code + results snapshot) |
| [AOM series package](experiments/aom_2026-09-28/REPORT.md) | Arbitrated Obligation Memory: scoring-vs-execution partition defect, fix, and seed-100 results on tasks {5,8,19,22} (code + results snapshot) |

## Supplementary material

| Document | Description |
|----------|-------------|
| [Supplementary experiments](supplementary/README.md) | Optional ablations and development jobs |

## Official benchmark API

| Document | Description |
|----------|-------------|
| [evaluation_benchmark/README.md](../evaluation_benchmark/README.md) | Adapter contract and 26-task sweep |
| [Evaluate your model](../evaluation_benchmark/docs/evaluate_your_model.md) | Custom checkpoint integration |
| [async_vlm26_reference](../evaluation_benchmark/async_vlm26_reference/README.md) | PrediMem VLM+VLA reference runner |

## Cluster jobs

| Directory | Description |
|-----------|-------------|
| [slurm/benchmark](../slurm/benchmark/) | Official reproduction and recommended experiments |
| [slurm/supplementary](../slurm/supplementary/) | Extended ablations and smoke tests |
