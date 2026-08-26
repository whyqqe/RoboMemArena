#!/usr/bin/env bash
#SBATCH --job-name=rma-all26-1s
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Official PrediMem async_vlm26_reference reproduction on all 26 tasks.
# Same protocol as paper/runner defaults, except NUM_TRIALS=1 (one seed per task).
#
# Checkpoints (auto-selected by runner):
#   Task 1     -> vlm_task1 + vla_alltask
#   Task 2-26  -> vlm_tasks1to26_ckpt74500 + vla_alltask
# Seeds: task_id trial0 uses SEED+0 (=100 for all tasks in single-trial mode).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${ROOT}/outputs/slurm" "${ROOT}/outputs/logs"
cd "${ROOT}"

module load slurm "nvhpc-hpcx-cuda12/23.11" 2>/dev/null || true

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
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

# --- Official PrediMem eval knobs (see run_fullvlm26_async_vlm_vla_csr_tsr.sh) ---
export TASKS_JSON='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]'
export NUM_TRIALS=1
export SEED=100
export PORT=8026
export RESIZE_SIZE=256
export REPLAN_STEPS=10
export NUM_STEPS_WAIT=10
export MAX_STEPS=2500
export POST_GOAL_STEPS=200
export POST_STAGE_STEPS=30
export FAIL_ON_EXTRA_POUR=1
export ASYNC_VLM=1
export VLM_INTERVAL=5
export VLM_QUEUE_SIZE=1
export N_RECENT=5
export K_MAX=0
export D_MERGE=6
export VLM_USE_WRIST=1
export VLM_USE_KEYFRAME_MEMORY=1
export VLM_INPUT_PROFILE=fullvlm_256
export VLM_MATCH_TRAINING_JPEG_ROUNDTRIP=0
export VLA_CONFIG=pi05_robomemarena

export OUT_ROOT="${ROOT}/outputs/eval_repro_all26_1seed_${SLURM_JOB_ID:-manual}"
# Let runner auto-split Task1 vs Tasks2-26 and pick VLM ckpt per group.
unset PREDIMEM_ASSET_GROUP

NGPU=$("${EVAL_PY}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "${NGPU}" -lt 2 ]]; then
  echo "[ERROR] need >=2 GPUs on one node (VLA+VLM); got ${NGPU}" >&2
  exit 2
fi
export SERVER_CUDA_VISIBLE_DEVICES=0
export EVAL_CUDA_VISIBLE_DEVICES=1

if command -v gcc >/dev/null 2>&1; then export CC="$(command -v gcc)"; fi
if command -v g++ >/dev/null 2>&1; then export CXX="$(command -v g++)"; fi

echo "[INFO] hostname=$(hostname) job=${SLURM_JOB_ID:-n/a}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] TASKS_JSON=${TASKS_JSON} NUM_TRIALS=${NUM_TRIALS} SEED=${SEED}"
"${EVAL_PY}" - <<'PY'
import torch
print("eval torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("n_gpus", torch.cuda.device_count())
PY

cd "${ROOT}/evaluation_benchmark/async_vlm26_reference"
bash run_fullvlm26_async_vlm_vla_csr_tsr.sh

bash "${ROOT}/scripts/merge_eval_outputs.sh" "${OUT_ROOT}"

echo "[OK] all26 1-seed reproduce finished. OUT_ROOT=${OUT_ROOT}"
cat "${OUT_ROOT}/summary.tsv" 2>/dev/null || true
cat "${OUT_ROOT}/aggregate.json" 2>/dev/null || true
