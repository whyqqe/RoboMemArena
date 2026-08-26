#!/usr/bin/env bash
#SBATCH --job-name=rma-harness-ab
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

# A/B test: baseline PrediMem vs Harness-enhanced (same models, same seeds).
# 8 memory-heavy / mixed tasks, 1 trial each, seed=100.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_TAG="${SLURM_JOB_ID:-manual}"
BASELINE_OUT="${ROOT}/outputs/harness_ab_baseline_${JOB_TAG}"
HARNESS_OUT="${ROOT}/outputs/harness_ab_harness_${JOB_TAG}"
# Mix of previously easy/hard/memory-dependent tasks.
TASKS='[1,4,5,11,14,16,18,22]'

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

export TASKS_JSON="${TASKS}"
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
unset PREDIMEM_ASSET_GROUP

NGPU=$("${EVAL_PY}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "${NGPU}" -lt 2 ]]; then
  echo "[ERROR] need >=2 GPUs on one node; got ${NGPU}" >&2
  exit 2
fi
export SERVER_CUDA_VISIBLE_DEVICES=0
export EVAL_CUDA_VISIBLE_DEVICES=1

run_eval() {
  local label="$1"
  local out_root="$2"
  local harness_enable="$3"
  export OUT_ROOT="${out_root}"
  export HARNESS_ENABLE="${harness_enable}"
  export HARNESS_MAX_RETRIES=2
  export HARNESS_STALL_STEPS=120
  export HARNESS_FORCE_VLM_REPLAN=1
  export HARNESS_VLM_CONTEXT=1
  export HARNESS_VLA_HINTS=0
  export HARNESS_SUBTASK_OVERRIDE=1
  export HARNESS_STAGE_CHECKPOINT=1
  export HARNESS_RELEASE_ON_RETRY=1
  export HARNESS_PERSIST_MEMORY=1
  export HARNESS_GLOBAL_RULES="${ROOT}/evaluation_benchmark/harness/global_rules.json"
  if [[ "${harness_enable}" == "1" ]]; then
    export HARNESS_API_PLANNER=0
  fi

  echo "[INFO] === ${label} === OUT_ROOT=${OUT_ROOT} HARNESS_ENABLE=${HARNESS_ENABLE}"
  cd "${ROOT}/evaluation_benchmark/async_vlm26_reference"
  bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
}

echo "[INFO] hostname=$(hostname) job=${SLURM_JOB_ID:-n/a}"
echo "[INFO] TASKS_JSON=${TASKS_JSON} NUM_TRIALS=${NUM_TRIALS} SEED=${SEED}"

run_eval "baseline" "${BASELINE_OUT}" 0
run_eval "harness" "${HARNESS_OUT}" 1

echo "[INFO] Merging outputs..."
bash "${ROOT}/scripts/merge_eval_outputs.sh" "${BASELINE_OUT}"
bash "${ROOT}/scripts/merge_eval_outputs.sh" "${HARNESS_OUT}"

echo "[INFO] Running comparison..."
bash "${ROOT}/scripts/compare_harness_abtest.sh" "${BASELINE_OUT}" "${HARNESS_OUT}"

echo "[OK] A/B complete."
echo "  Baseline: ${BASELINE_OUT}"
echo "  Harness:  ${HARNESS_OUT}"
