# RoboMemArena

RoboMemArena is a robotic memory benchmark with 26 manipulation tasks, demonstration data, and evaluation BDDL specifications.

This repository provides:

- The **official evaluation benchmark** (adapter-based CSR/TSR for tasks 1–26)
- A **PrediMem reference stack** (Qwen3-VL planner + π0.5 VLA)
- An **inference-time extension** with an external execution harness and long-horizon memory modules (no model fine-tuning)

[![arXiv](https://img.shields.io/badge/arXiv-2605.10921-b31b1b)](https://arxiv.org/html/2605.10921v1)
[![Project Page](https://img.shields.io/badge/Project-Page-76b900)](https://robomemarena.github.io/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Results-007ec6)](https://robomemarena.github.io/leaderboard.html)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-f3b900)](https://huggingface.co/datasets/RoboMemArenaBenchmark/RoboMemArena)

## Table of contents

- [Documentation](#documentation)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Reproducibility](#reproducibility)
- [Dataset](#dataset)
- [Evaluate your own model](#evaluate-your-own-model)
- [Citation](#citation)
- [Contact](#contact)

## Documentation

| Document | Description |
|----------|-------------|
| [docs/](docs/README.md) | Documentation index |
| [Installation & dependencies](docs/setup/INSTALLATION.md) | Python environments, GPU, EGL rendering |
| [Assets download](docs/setup/ASSETS.md) | PrediMem checkpoints and dataset |
| [Reproducibility guide](docs/experiments/REPRODUCIBILITY.md) | Benchmark and extension experiments |
| [System design](docs/design/harness_memory_system_design.md) | Harness and memory architecture |
| [Dual-track evaluation report](docs/experiments/dual_track_report.md) | Decoupled memory vs. harness study |
| [Supplementary experiments](docs/supplementary/README.md) | Extended ablations (optional) |

## Installation

**Requirements:** Linux, NVIDIA GPU (2 GPUs recommended for PrediMem reference eval), Python 3.11, CUDA 12.x–compatible driver.

```bash
git clone https://github.com/OpenHelix-Team/RoboMemArena.git
cd RoboMemArena
bash scripts/setup_env.sh
source scripts/activate_eval.sh
bash scripts/download_assets.sh   # PrediMem weights (required for reference eval)
```

See [docs/setup/INSTALLATION.md](docs/setup/INSTALLATION.md) for details (dual virtualenv layout, headless EGL, troubleshooting).

## Quick start

**Single-task smoke test (PrediMem baseline):**

```bash
source scripts/activate_eval.sh
cd evaluation_benchmark/async_vlm26_reference
export TASKS_JSON='[1]' SEED=100 NUM_TRIALS=1
bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
```

**Recommended harness configuration (`harness_v21`):**

```bash
cd /path/to/RoboMemArena
source scripts/run_harness_variant.sh harness_v21
export TASKS_JSON='[1,4,5,11,14,16,18,22]' SEED=100
cd evaluation_benchmark/async_vlm26_reference
bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
```

**Slurm (cluster):**

```bash
sbatch slurm/benchmark/reproduce_all26_1seed.sbatch
sbatch slurm/benchmark/harness_dual_track.sbatch
```

## Repository layout

```
RoboMemArena/
├── docs/                          # Documentation
│   ├── setup/                     # Installation and assets
│   ├── experiments/               # Reproducibility and reports
│   ├── design/                    # Architecture reference
│   └── supplementary/             # Optional ablation notes
├── scripts/                       # Setup, download, harness variants, aggregation
├── slurm/
│   ├── benchmark/                 # Official and recommended experiments
│   └── supplementary/             # Development and ablation jobs
├── evaluation_benchmark/
│   ├── harness/                   # External execution harness
│   ├── memory_system/             # Long-horizon memory extensions
│   ├── async_vlm26_reference/     # PrediMem reference evaluator
│   ├── libero_fork/               # LIBERO-compatible simulation
│   └── scripts/                   # Generic adapter-based evaluation
├── third_party/openpi_minimal/    # OpenPI VLA server runtime
├── bddl/                          # Task BDDL definitions (bundled)
├── checkpoints/PrediMem/          # Downloaded weights (not in git)
├── data/RoboMemArena/             # Downloaded dataset (not in git)
└── outputs/                       # Evaluation logs and videos (not in git)
```

## Reproducibility

We organize experiments into three tiers. Full protocols are in [docs/experiments/REPRODUCIBILITY.md](docs/experiments/REPRODUCIBILITY.md).

| Tier | Experiment | Slurm job |
|------|------------|-----------|
| **Benchmark** | PrediMem on all 26 tasks (official protocol) | `slurm/benchmark/reproduce_all26_1seed.sbatch` |
| **Extension (primary)** | PrediMem vs. recommended harness (`harness_v21`) | `slurm/benchmark/harness_main_p0.sbatch` |
| **Extension (ablation)** | Decoupled memory vs. harness vs. combined | `slurm/benchmark/harness_dual_track.sbatch` |

**Extension summary** (8 memory-intensive tasks, 5 seeds, paired comparison):

| Configuration | ΔCSR vs. baseline |
|---------------|-------------------|
| `memory_kf` (visual memory only) | +8.2 pp |
| `harness_exec` (execution recovery only) | +8.6 pp |
| `harness_v21` (combined, recommended) | +13.4 pp |

Details: [docs/experiments/dual_track_report.md](docs/experiments/dual_track_report.md).

## Dataset

The dataset is hosted on [Hugging Face](https://huggingface.co/datasets/RoboMemArenaBenchmark/RoboMemArena) with a ModelScope mirror (see project page).

**Use the dataset release on or after June 20, 2026 (v2).** If you downloaded an older snapshot, replace it entirely.

Simulation-based evaluation of PrediMem **does not require** the HDF5 training dataset. Download is only needed for RLDS conversion or model training. See [docs/setup/ASSETS.md](docs/setup/ASSETS.md).

### RLDS conversion

```bash
conda create -n robomemarena-rlds python=3.9 -y
pip install tensorflow==2.13.0 tensorflow-datasets==4.9.2 h5py==3.9.0 numpy==1.24.3
export ROBOMEMARENA_DATA_ROOT=/path/to/dataset
python -c "
import RoboMemArena_dataset_builder as b
ds_builder = b.RoboMemArenaDataset(data_dir='/path/to/output')
ds_builder.download_and_prepare()
"
```

## Evaluate your own model

- [Evaluation benchmark overview](evaluation_benchmark/README.md)
- [Adapter integration guide](evaluation_benchmark/docs/evaluate_your_model.md)
- [PrediMem reference runner](evaluation_benchmark/async_vlm26_reference/README.md)

## PrediMem training add-ons

- [predictive_coding_head/](predictive_coding_head/) — Predictive Coding Head integration for PrediMem S2
- Low-level policy (S1): [OpenPI](https://github.com/Physical-Intelligence/openpi)
- VLM data format: [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

## Citation

```bibtex
@article{robomemarena2025,
  title   = {RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark},
  author  = {Huashuo Lei and Wenxuan Song and Huarui Zhang and Jieyuan Pei and Jiayi Chen and Haodong Yan and Han Zhao and Pengxiang Ding and Zhipeng Zhang and Lida Huang and Donglin Wang and Yan Wang and Haoang Li},
  journal = {arXiv preprint arXiv:2605.10921},
  year    = {2026}
}
```

## Contact

WeChat: `leshuaigeye` · Email: `leihuashuohit@gmail.com`
