# Installation

This guide describes how to install dependencies for **PrediMem reference evaluation** and the **harness/memory extension** on a Linux GPU server, including headless Slurm nodes.

The reference stack uses **two separate Python virtualenvs** so that the VLM planner (`transformers>=4.57`) and the OpenPI VLA server can coexist without dependency conflicts.

---

## System requirements

| Component | Requirement |
|-----------|-------------|
| GPU | **2 NVIDIA GPUs** recommended (VLA server on GPU 0, VLM + simulation on GPU 1) |
| GPU memory | ≥ 40 GB per GPU suggested |
| Driver | CUDA 12.x–compatible NVIDIA driver |
| CPU / RAM | ≥ 28 cores / 120 GB RAM for full 26-task evaluation |
| OS | Linux with EGL offscreen rendering for MuJoCo |

### Headless rendering

Set before running simulation:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If OpenGL errors mention `libGLU.so.1`, add the system GLU library directory to `LD_LIBRARY_PATH` (e.g. `/usr/lib/x86_64-linux-gnu`).

---

## Python environments

| Virtualenv | Default path | Role | Key packages |
|------------|--------------|------|--------------|
| **Eval** | `.venv/` | Qwen3-VL, LIBERO/MuJoCo, harness evaluation | `transformers>=4.57`, `mujoco==3.1.6`, `robosuite==1.4.1` |
| **VLA** | `third_party/openpi_minimal/.venv/` | OpenPI π0.5 WebSocket policy server | `jax`, OpenPI via `PYTHONPATH` |

- **Python version:** 3.11 recommended (3.10 usually works).
- **NumPy:** pinned to `<2.0` in eval requirements.

OpenPI packages live under `third_party/openpi_minimal/packages/` and are loaded via `PYTHONPATH` (no `pip install -e`).

---

## Install

From the repository root:

```bash
bash scripts/setup_env.sh
```

This creates both virtualenvs, installs `requirements-eval.txt` and `requirements-vla.txt`, applies a robosuite compatibility patch, and runs import smoke tests.

### Optional: custom cache location

```bash
export ROBOMEMARENA_ROOT=/path/to/RoboMemArena
export CACHE_ROOT=/path/to/large_disk/cache
bash scripts/setup_env.sh
```

Default cache: `$ROBOMEMARENA_ROOT/.cache` when `CACHE_ROOT` is unset.

---

## Activation

```bash
# Terminal A — VLM planner + simulation
source scripts/activate_eval.sh

# Terminal B — VLA policy server (requires GPU)
source scripts/activate_vla.sh
```

Activation scripts configure `OPENPI_ROOT`, `TARGET_LIBERO_PATH`, `HF_HOME`, `PYTHONPATH`, and LIBERO config paths.

---

## Verification

On a **GPU node**:

```bash
source scripts/activate_eval.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

source scripts/activate_vla.sh
python -c "import torch, jax; print('cuda', torch.cuda.is_available()); print('jax', jax.devices())"

source scripts/activate_eval.sh
python -c "import mujoco, robosuite; print('simulation OK')"
```

On login nodes without GPUs, `torch.cuda.is_available()` may be `False`; import checks can still run on CPU.

---

## Running evaluation

### Environment variables

| Variable | Purpose |
|----------|---------|
| `PREDIMEM_HF_SNAPSHOT` | Root directory of downloaded PrediMem weights |
| `TASKS_JSON` | Task ID list, e.g. `'[1,4,5]'` |
| `SEED` | LIBERO simulation seed |
| `NUM_TRIALS` | Trials per task |
| `OUT_ROOT` | Output directory |

### Single-task debug

```bash
source scripts/activate_eval.sh
cd evaluation_benchmark/async_vlm26_reference
export TASKS_JSON='[1]' SEED=100 NUM_TRIALS=1
bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
```

### Harness configurations

```bash
source scripts/run_harness_variant.sh harness_v21   # recommended
# Alternatives: baseline, memory_kf, harness_exec, ...
```

See [experiments/REPRODUCIBILITY.md](experiments/REPRODUCIBILITY.md) and [design/harness_memory_system_design.md](../design/harness_memory_system_design.md).

### Slurm

Edit `#SBATCH` account/partition lines for your cluster, then:

```bash
sbatch slurm/benchmark/reproduce_all26_1seed.sbatch
sbatch slurm/benchmark/harness_dual_track.sbatch
```

Job scripts resolve the repository root relative to their location.

---

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| `Qwen3VLForConditionalGeneration` not found | Upgrade eval venv: `pip install 'transformers>=4.57'` |
| MuJoCo / GL errors | Set `MUJOCO_GL=egl`; verify `libGLU` on `LD_LIBRARY_PATH` |
| mujoco 3.12 + robosuite incompatibility | Keep `mujoco==3.1.6` (pinned in requirements) |
| LIBERO interactive config prompt | Run via `activate_eval.sh` (writes `LIBERO_CONFIG_PATH`) |
| Gated Hugging Face assets | `huggingface-cli login` or use ModelScope fallback in download script |
| Disk quota on `$HOME` | Set `CACHE_ROOT` to a large filesystem |
| VLA server not ready | Check `OUT_ROOT/logs/serve_policy.log` |

### Cluster modules (example)

```bash
module load slurm nvhpc-hpcx-cuda12/23.11   # adjust for your site
```

PyTorch `cu126` wheels bundle a CUDA runtime; the node driver should still support CUDA 12.x.

### Resource estimates (8-task harness study)

- GPUs: 2 × 40GB+ class
- Wall time: ~1.5 h per seed (full variant matrix); ~7 h for 5-seed dual-track study
- Disk: tens of GB for checkpoints; outputs grow quickly if videos are saved

---

## Requirement files

| File | Contents |
|------|----------|
| `requirements-cu126.txt` | PyTorch 2.6 + CUDA 12.6 wheels |
| `requirements-eval.txt` | VLM, simulation, evaluation |
| `requirements-vla.txt` | OpenPI VLA server |

RLDS dataset conversion uses a separate Python 3.9 + TensorFlow 2.13 environment (see root README).
