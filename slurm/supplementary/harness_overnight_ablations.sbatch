#!/usr/bin/env bash
#SBATCH --job-name=rma-harn-ovn
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Overnight harness mechanism ablations (PrediMem + Harness v2).
# 1 shared baseline + 7 harness variants on 12 memory-heavy tasks.
# Inspired by MemER (keyframe retrieval), MemoryVLA (read-time memory),
# HarnessVLA (retry/override/primitives).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_BASE="${ROOT}/outputs/harness_overnight_${JOB_TAG}"
BASELINE_OUT="${OUT_BASE}/baseline"

# 12 memory-heavy / multi-stage tasks (~6–8h estimated with retries)
TASKS='[1,4,5,6,11,12,14,15,16,17,18,22]'

VARIANTS=(
  v2_full
  memer_kf
  ctx_only
  retry_only
  override_only
  stall_fast
  v1_hints
)

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
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/run_harness_variant.sh" "${variant}"
  export OUT_ROOT="${out_root}"
  echo "[INFO] === variant=${variant} OUT_ROOT=${OUT_ROOT} ==="
  echo "[INFO]   HARNESS_ENABLE=${HARNESS_ENABLE:-0} K_MAX=${K_MAX:-0} STALL=${HARNESS_STALL_STEPS:-n/a}"
  cd "${ROOT}/evaluation_benchmark/async_vlm26_reference"
  bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
  bash "${ROOT}/scripts/merge_eval_outputs.sh" "${OUT_ROOT}"
}

echo "[INFO] hostname=$(hostname) job=${JOB_TAG}"
echo "[INFO] OUT_BASE=${OUT_BASE}"
echo "[INFO] TASKS_JSON=${TASKS_JSON} NUM_TRIALS=${NUM_TRIALS} SEED=${SEED}"
echo "[INFO] variants: baseline ${VARIANTS[*]}"

# --- Phase 1: shared baseline ---
run_variant baseline "${BASELINE_OUT}"

# --- Phase 2: harness ablations ---
for v in "${VARIANTS[@]}"; do
  run_variant "${v}" "${OUT_BASE}/variant_${v}"
done

# --- Phase 3: matrix comparison ---
REPORT="${OUT_BASE}/ablation_matrix.md"
VARIANT_PATHS=()
for v in "${VARIANTS[@]}"; do
  VARIANT_PATHS+=("${OUT_BASE}/variant_${v}")
done
bash "${ROOT}/scripts/compare_harness_matrix.sh" "${BASELINE_OUT}" "${VARIANT_PATHS[@]}" | tee "${REPORT}"

# Save machine-readable summary
python3 - "${OUT_BASE}" "${BASELINE_OUT}" "${VARIANTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_base = Path(sys.argv[1])
baseline = Path(sys.argv[2])
variants = sys.argv[3:]

def agg(p):
    f = p / "aggregate.json"
    return json.loads(f.read_text()) if f.is_file() else {}

summary = {
    "baseline": {"path": str(baseline), **agg(baseline)},
    "variants": {},
}
for v in variants:
    p = out_base / f"variant_{v}"
    summary["variants"][v] = {"path": str(p), **agg(p)}

(out_base / "ablation_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("[OK] wrote", out_base / "ablation_summary.json")
PY

echo "[OK] overnight ablations complete."
echo "  Report: ${REPORT}"
echo "  Summary: ${OUT_BASE}/ablation_summary.json"
cat "${OUT_BASE}/ablation_summary.json"
