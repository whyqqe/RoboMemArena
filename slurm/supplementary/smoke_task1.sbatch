#!/usr/bin/env bash
#SBATCH --job-name=rma-smoke-t1
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Smoke / initial PrediMem reproduction on RoboMemArena Task 1 (1 episode).
# VLA server on GPU0, VLM+sim on GPU1.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${ROOT}/outputs/slurm" "${ROOT}/outputs/logs"
cd "${ROOT}"

module load slurm "nvhpc-hpcx-cuda12/23.11" 2>/dev/null || true

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# EGL offscreen rendering may need libGLU on GPU nodes.
for _glu_dir in \
  /usr/lib/x86_64-linux-gnu \
  /usr/lib64 \
  /cm/shared/apps/mesa/*/lib; do
  if [[ -f "${_glu_dir}/libGLU.so.1" || -f "${_glu_dir}/libGLU.so" ]]; then
    export LD_LIBRARY_PATH="${_glu_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    break
  fi
done
unset _glu_dir

# shellcheck disable=SC1091
source "${ROOT}/scripts/env_common.sh"
robomem_setup_modules
robomem_setup_cache

export OPENPI_ROOT="${ROOT}/third_party/openpi_minimal"
export OPENPI_INFERENCE_ROOT="${ROOT}"
export LIBERO_FORK_ROOT="${ROOT}/evaluation_benchmark/libero_fork"
export TARGET_LIBERO_PATH="${LIBERO_FORK_ROOT}/libero"
export PREDIMEM_HF_SNAPSHOT="${ROOT}/checkpoints/PrediMem"
export SERVER_PY="${OPENPI_ROOT}/.venv/bin/python"
export EVAL_PY="${ROOT}/.venv/bin/python"
export TASKS_JSON='[1]'
export NUM_TRIALS=1
export SEED=100
export PORT=8026
# Auto map GPUs: prefer split (VLA=0, VLM=1); fall back to shared GPU 0.
NGPU=$("${EVAL_PY}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "${NGPU}" -lt 2 ]]; then
  echo "[ERROR] need >=2 GPUs on one node (VLA+VLM); got ${NGPU}. Check #SBATCH --nodes=1 --gpus=2" >&2
  exit 2
fi
export SERVER_CUDA_VISIBLE_DEVICES=0
export EVAL_CUDA_VISIBLE_DEVICES=1
export OUT_ROOT="${ROOT}/outputs/eval_smoke_task1_${SLURM_JOB_ID:-manual}"
export PREDIMEM_ASSET_GROUP=task1

# Prefer system gcc for any JIT (avoid nvhpc nvc flags).
if command -v gcc >/dev/null 2>&1; then export CC="$(command -v gcc)"; fi
if command -v g++ >/dev/null 2>&1; then export CXX="$(command -v g++)"; fi

echo "[INFO] hostname=$(hostname) job=${SLURM_JOB_ID:-n/a}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
"${EVAL_PY}" - <<'PY'
import torch
print("eval torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("n_gpus", torch.cuda.device_count())
PY

cd "${ROOT}/evaluation_benchmark/async_vlm26_reference"
bash run_fullvlm26_async_vlm_vla_csr_tsr.sh

echo "[OK] smoke finished. OUT_ROOT=${OUT_ROOT}"
ls -la "${OUT_ROOT}/" || true
cat "${OUT_ROOT}/summary.tsv" 2>/dev/null || true
cat "${OUT_ROOT}/aggregate.json" 2>/dev/null || true
