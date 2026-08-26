#!/usr/bin/env bash
# Create pip virtualenvs for RoboMemArena on peilab (NO conda).
#
# Cluster: DGX nodes, NVIDIA H800 x8, account=peilab, partition=normal|preempt
# Module:  nvhpc-hpcx-cuda12/23.11
# Torch:   2.6.0+cu126 (bundled CUDA 12.6 runtime; works on H800)
#
# Prefer a GPU allocation to verify CUDA after install:
#   module load slurm nvhpc-hpcx-cuda12/23.11
#   srun --account=peilab --partition=normal --gpus=1 --cpus-per-gpu=28 \
#        --mem=120G --time=04:00:00 --pty bash
#   bash scripts/setup_env.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_common.sh"
robomem_setup_cache

EVAL_VENV="${EVAL_VENV:-${ROOT}/.venv}"
VLA_VENV="${VLA_VENV:-${OPENPI_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x /usr/bin/python3.11 ]]; then
    PYTHON_BIN=/usr/bin/python3.11
  elif [[ -x /cm/shared/apps/Anaconda3/2023.09-0/bin/python3.11 ]]; then
    PYTHON_BIN=/cm/shared/apps/Anaconda3/2023.09-0/bin/python3.11
  elif [[ -x /usr/bin/python3.10 ]]; then
    PYTHON_BIN=/usr/bin/python3.10
  else
    PYTHON_BIN=python3
  fi
fi

echo "[INFO] Bootstrap Python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version))"
echo "[INFO] Eval venv  -> ${EVAL_VENV}"
echo "[INFO] VLA venv   -> ${VLA_VENV}"
echo "[INFO] Cache root -> ${CACHE_ROOT}"

create_venv() {
  local venv_dir="$1"
  if [[ ! -f "${venv_dir}/bin/activate" ]]; then
    "${PYTHON_BIN}" -m venv "${venv_dir}"
  fi
  # shellcheck disable=SC1091
  source "${venv_dir}/bin/activate"
  python -m pip install --upgrade pip wheel setuptools
}

# third_party/openpi_minimal is a source-only bundle (no pyproject.toml).
# openpi / openpi_client are imported via PYTHONPATH (see env_common.sh).

echo "========== [1/2] Eval venv (VLM + MuJoCo/LIBERO) =========="
create_venv "${EVAL_VENV}"
python -m pip install -r "${ROOT}/requirements-eval.txt"
python "${ROOT}/scripts/patch_robosuite_binding.py"

echo "========== [2/2] VLA venv (OpenPI policy server) =========="
create_venv "${VLA_VENV}"
python -m pip install -r "${ROOT}/requirements-vla.txt"

echo "========== Smoke checks (CPU ok; CUDA check needs GPU node) =========="
# shellcheck disable=SC1091
source "${EVAL_VENV}/bin/activate"
robomem_setup_cache
python - <<'PY'
import importlib
mods = ["torch", "transformers", "peft", "openpi_client"]
for m in mods:
    importlib.import_module(m)
from transformers import Qwen3VLForConditionalGeneration
try:
    importlib.import_module("mujoco")
    importlib.import_module("robosuite")
    mods.extend(["mujoco", "robosuite"])
except Exception as exc:
    print(f"[WARN] mujoco/robosuite skipped on this node (need GPU+EGL): {exc}")
print("[OK] eval imports:", mods, "+ Qwen3VLForConditionalGeneration")
import torch
print(f"[eval] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
PY

# shellcheck disable=SC1091
source "${VLA_VENV}/bin/activate"
robomem_setup_cache
python - <<'PY'
import importlib
mods = ["torch", "transformers", "jax", "flax", "tyro", "websockets", "openpi_client", "openpi"]
for m in mods:
    importlib.import_module(m)
print("[OK] vla imports:", mods)
import torch
print(f"[vla] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
PY

cat <<EOF

[OK] RoboMemArena environments ready.

Activate eval (VLM + sim):
  source ${ROOT}/scripts/activate_eval.sh

Activate VLA server:
  source ${ROOT}/scripts/activate_vla.sh

Download assets (after HF token if needed):
  bash ${ROOT}/scripts/download_assets.sh

Verify CUDA inside a GPU shell:
  source ${ROOT}/scripts/activate_eval.sh
  python -c "import torch; print(torch.cuda.get_device_name(0))"
EOF
