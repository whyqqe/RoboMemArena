#!/usr/bin/env bash
# Download RoboMemArena dataset from ModelScope (4 category splits).
# HF dataset is gated; ModelScope mirror is under haodong123 namespace.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_eval.sh"

python -m pip install -q modelscope

DATA_ROOT="${DATA_ROOT:-${ROOT}/data/RoboMemArena}"
LOG_DIR="${ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}" "${DATA_ROOT}"

download_one() {
  local ms_id="$1"
  local local_name="$2"
  local dest="${DATA_ROOT}/${local_name}"
  local log="${LOG_DIR}/download_ms_${local_name}.log"
  echo "[START] ${ms_id} -> ${dest} $(date)" | tee "${log}"
  modelscope download --dataset "${ms_id}" --local_dir "${dest}" 2>&1 | tee -a "${log}"
  echo "[DONE] ${ms_id} size=$(du -sh "${dest}" | cut -f1) $(date)" | tee -a "${log}"
}

# Run remaining categories in parallel (Counting may already be downloading).
PIDS=()
for pair in \
  "haodong123/RoboMemArena-Multi-Object-Occlusion:Multi-Object-Occlusion" \
  "haodong123/RoboMemArena-Multi-Object-Sequence:Multi-Object-Sequence" \
  "haodong123/RoboMemArena-Multi-Object-Transferring:Multi-Object-Transferring"; do
  IFS=: read -r ms_id local_name <<< "${pair}"
  if [[ ! -d "${DATA_ROOT}/${local_name}" ]] || [[ "$(find "${DATA_ROOT}/${local_name}" -name '*.hdf5' 2>/dev/null | wc -l)" -lt 10 ]]; then
    download_one "${ms_id}" "${local_name}" &
    PIDS+=($!)
  else
    echo "[SKIP] ${local_name} already has hdf5 files"
  fi
done

# Ensure Counting category (may already be in progress).
if [[ ! -d "${DATA_ROOT}/Multi-Object-Counting" ]] || [[ "$(find "${DATA_ROOT}/Multi-Object-Counting" -name '*.hdf5' 2>/dev/null | wc -l)" -lt 10 ]]; then
  download_one "haodong123/RoboMemArena-Multi-Object-Counting" "Multi-Object-Counting" &
  PIDS+=($!)
fi

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "[OK] All ModelScope categories downloaded under ${DATA_ROOT}"
du -sh "${DATA_ROOT}"/* 2>/dev/null
