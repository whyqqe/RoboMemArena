#!/usr/bin/env bash
# Shared cache / path helpers for RoboMemArena on peilab (pip venv, NO conda).
# All writable caches stay under /project/peilab/why/cache — never /home.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_DEFAULT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
ROBOMEMARENA_ROOT="${ROBOMEMARENA_ROOT:-${_DEFAULT_ROOT}}"
CACHE_ROOT="${CACHE_ROOT:-${ROBOMEMARENA_ROOT}/.cache}"
PROJ_CACHE="${ROBOMEMARENA_CACHE:-${CACHE_ROOT}/robomemarena}"

robomem_setup_modules() {
  if command -v module >/dev/null 2>&1; then
    module load slurm "nvhpc-hpcx-cuda12/23.11" 2>/dev/null || \
      module load "nvhpc-hpcx-cuda12/23.11" 2>/dev/null || \
      module load "cuda12.2/toolkit/12.2.2" 2>/dev/null || \
      echo "[WARN] Could not load CUDA module; PyTorch bundled CUDA runtime may still work."
  fi
}

robomem_setup_cache() {
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export ROBOMEMARENA_ROOT
  export CACHE_ROOT
  export ROBOMEMARENA_CACHE="${PROJ_CACHE}"

  export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
  export HF_HOME="${CACHE_ROOT}/huggingface"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export TRANSFORMERS_CACHE="${HF_HOME}/hub"
  export TORCH_HOME="${CACHE_ROOT}/torch"
  export UV_CACHE_DIR="${CACHE_ROOT}/uv"
  export XDG_CACHE_HOME="${PROJ_CACHE}/xdg"
  export XDG_CONFIG_HOME="${PROJ_CACHE}/xdg-config"
  export TMPDIR="${PROJ_CACHE}/tmp"
  export TMP="${TMPDIR}"
  export TEMP="${TMPDIR}"

  # Redirect any accidental home writes into the project tree.
  export HOME="${ROBOMEMARENA_ROOT}/outputs/download_scratch/fake_home"

  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"

  export OPENPI_ROOT="${OPENPI_ROOT:-${ROBOMEMARENA_ROOT}/third_party/openpi_minimal}"
  export OPENPI_INFERENCE_ROOT="${OPENPI_INFERENCE_ROOT:-${ROBOMEMARENA_ROOT}}"
  export LIBERO_FORK_ROOT="${LIBERO_FORK_ROOT:-${ROBOMEMARENA_ROOT}/evaluation_benchmark/libero_fork}"
  export TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH:-${LIBERO_FORK_ROOT}/libero}"
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PROJ_CACHE}/libero_config}"
  export LIBERO_BDDL_PATH="${LIBERO_BDDL_PATH:-${ROBOMEMARENA_ROOT}/evaluation_benchmark/bddl}"

  # libero.libero.* imports need libero_fork on PYTHONPATH (not libero_fork/libero).
  export PYTHONPATH="${LIBERO_FORK_ROOT}:${ROBOMEMARENA_ROOT}/evaluation_benchmark/openpi_minimal_runtime:${ROBOMEMARENA_ROOT}/evaluation_benchmark/scripts:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/packages/openpi/src:${PYTHONPATH:-}"

  if command -v gcc >/dev/null 2>&1; then
    export CC="$(command -v gcc)"
  fi
  if command -v g++ >/dev/null 2>&1; then
    export CXX="$(command -v g++)"
  fi
  unset CFLAGS CXXFLAGS 2>/dev/null || true

  for v in PIP_CACHE_DIR HF_HOME HUGGINGFACE_HUB_CACHE TORCH_HOME TMPDIR XDG_CACHE_HOME; do
    val="${!v}"
    case "${val}" in
      /home/*)
        echo "[ERROR] ${v}=${val} is under /home. Abort." >&2
        return 1 2>/dev/null || exit 1
        ;;
    esac
  done

  mkdir -p \
    "${HOME}" \
    "${HF_HOME}/hub" \
    "${TORCH_HOME}" \
    "${PIP_CACHE_DIR}" \
    "${UV_CACHE_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${TMPDIR}" \
    "${LIBERO_CONFIG_PATH}" \
    "${ROBOMEMARENA_ROOT}/outputs/logs" \
    "${ROBOMEMARENA_ROOT}/outputs/slurm" \
    "${ROBOMEMARENA_ROOT}/checkpoints" \
    "${ROBOMEMARENA_ROOT}/data" \
    "${PROJ_CACHE}"

  # Non-interactive LIBERO config (avoids ~/.libero prompt on import).
  if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
    cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${TARGET_LIBERO_PATH}
bddl_files: ${TARGET_LIBERO_PATH}/bddl_files
init_states: ${TARGET_LIBERO_PATH}/init_files
datasets: ${ROBOMEMARENA_ROOT}/data/libero_datasets
assets: ${TARGET_LIBERO_PATH}/assets
EOF
  fi
}
