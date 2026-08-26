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
