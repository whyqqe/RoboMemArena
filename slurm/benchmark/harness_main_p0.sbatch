#!/usr/bin/env bash
#SBATCH --job-name=rma-main-p0
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# P0 main experiment: baseline vs v21_best, 8 memory-heavy tasks, 5 seeds.
# v21_best = memer_kf + stall_fast + smart retry + skip retry on task 18/22.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_BASE="${ROOT}/outputs/harness_main_p0_${JOB_TAG}"

TASKS='[1,4,5,11,14,16,18,22]'
SEEDS=(100 101 102 103 104)

mkdir -p "${ROOT}/outputs/slurm" "${OUT_BASE}"
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
export VLM_USE_WRIST=1
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
  echo "[ERROR] need >=2 GPUs; got ${NGPU}" >&2
  exit 2
fi
export SERVER_CUDA_VISIBLE_DEVICES=0
export EVAL_CUDA_VISIBLE_DEVICES=1

run_variant() {
  local variant="$1"
  local out_root="$2"
  local seed="$3"
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/run_harness_variant.sh" "${variant}"
  export OUT_ROOT="${out_root}"
  export SEED="${seed}"
  echo "[INFO] === variant=${variant} seed=${seed} OUT_ROOT=${OUT_ROOT} ==="
  cd "${ROOT}/evaluation_benchmark/async_vlm26_reference"
  bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
  bash "${ROOT}/scripts/merge_eval_outputs.sh" "${OUT_ROOT}"
}

echo "[INFO] hostname=$(hostname) job=${JOB_TAG}"
echo "[INFO] OUT_BASE=${OUT_BASE} TASKS_JSON=${TASKS_JSON}"
echo "[INFO] seeds=${SEEDS[*]} harness=v21_best"

for s in "${SEEDS[@]}"; do
  run_variant baseline "${OUT_BASE}/baseline_s${s}" "${s}"
  run_variant v21_best "${OUT_BASE}/v21_best_s${s}" "${s}"
done

REPORT="${OUT_BASE}/main_multiseed_report.md"
bash "${ROOT}/scripts/aggregate_multiseed_main.sh" "${OUT_BASE}" "baseline_s" "v21_best_s" | tee "${REPORT}"

echo "[OK] P0 main experiment complete."
echo "  Report: ${REPORT}"
echo "  Summary: ${OUT_BASE}/main_multiseed_summary.json"
