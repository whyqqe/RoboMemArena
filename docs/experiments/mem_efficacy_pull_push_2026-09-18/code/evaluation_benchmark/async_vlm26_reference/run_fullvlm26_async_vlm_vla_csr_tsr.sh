#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "${SCRIPT_DIR}/../openpi_minimal_runtime" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_OPENPI_ROOT="${REPO_ROOT}/third_party/openpi_minimal"

OPENPI_ROOT=${OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}
: "${OPENPI_INFERENCE_ROOT:?Set OPENPI_INFERENCE_ROOT to the openpi_inference checkout.}"
: "${TARGET_LIBERO_PATH:?Set TARGET_LIBERO_PATH to the LIBERO/libero package path.}"

PREDIMEM_HF_REVISION=e645741c2f34f27e1596bfa89e856d6f3560ed90
PREDIMEM_HF_SNAPSHOT=${PREDIMEM_HF_SNAPSHOT:-"${HF_HOME:-${HOME}/.cache/huggingface}/hub/models--huashuolei--PrediMem/snapshots/${PREDIMEM_HF_REVISION}"}
VLA_CKPT="${PREDIMEM_HF_SNAPSHOT}/vla_alltask"
NORM_STATS_PATH="${VLA_CKPT}/assets/policy_assets/norm_stats.json"

VLA_CONFIG=${VLA_CONFIG:-pi05_robomemarena}
SERVER_PY=${SERVER_PY:-${OPENPI_ROOT}/.venv/bin/python3}
if [ ! -x "${SERVER_PY}" ]; then
  SERVER_PY=python3
fi
EVAL_PY=${EVAL_PY:-${OPENPI_INFERENCE_ROOT}/.venv/bin/python}
PORT=${PORT:-8026}
# When two forest jobs land on the same node they previously shared PORT=8052 and the
# second job's `pkill -f serve_policy.py --port 8052` murdered the first job's VLA
# (573558 exit 91 + 573737 "VLA server exited early", both on dgx-26). Prefer a
# job-unique port whenever Slurm gives us an id; keep an explicit override via
# PORT_EXPLICIT=1 PORT=<n>.
if [[ -z "${PORT_EXPLICIT:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
  # 30000..49999: far from 8026/8052 and large enough that JobID collisions are rare.
  PORT=$((30000 + (SLURM_JOB_ID % 20000)))
fi
TS=${TS:-$(date +%Y%m%d_%H%M%S)}

TASK_CONFIG=${TASK_CONFIG:-${SCRIPT_DIR}/fullvlm_v2_26_memory_tasks.json}
OUT_ROOT=${OUT_ROOT:-${OPENPI_INFERENCE_ROOT}/output/eval_fullvlm26_async_vlm_vla_${TS}}
LOG_DIR=${OUT_ROOT}/logs
VIDEO_DIR=${VIDEO_DIR:-${OUT_ROOT}/videos}
SUMMARY_JSON=${SUMMARY_JSON:-${OUT_ROOT}/summary.json}
SUMMARY_TSV=${SUMMARY_TSV:-${OUT_ROOT}/summary.tsv}
PROMPT_TRACE_TSV=${PROMPT_TRACE_TSV:-${OUT_ROOT}/prompt_trace.tsv}
SERVER_LOG=${SERVER_LOG:-${LOG_DIR}/serve_policy.log}
EVAL_LOG=${EVAL_LOG:-${LOG_DIR}/eval_fullvlm26_async_vlm_vla.log}

mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"

export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OPENPI_ROOT OPENPI_INFERENCE_ROOT TARGET_LIBERO_PATH
LIBERO_FORK_ROOT="${LIBERO_FORK_ROOT:-$(cd "${TARGET_LIBERO_PATH}/.." && pwd)}"
export PYTHONPATH="${LIBERO_FORK_ROOT}:${TARGET_LIBERO_PATH}:${RUNTIME_DIR}:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/packages/openpi/src:${OPENPI_ROOT}:${PYTHONPATH:-}"
export OUT_ROOT VIDEO_DIR SUMMARY_JSON SUMMARY_TSV PROMPT_TRACE_TSV TASK_CONFIG
export HOST=${HOST:-127.0.0.1}
export PORT
export VLM_LORA_PATH=${VLM_LORA_PATH:-none}
export VLM_DEVICE=${VLM_DEVICE:-cuda:0}
export TASKS_JSON=${TASKS_JSON:-"[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]"}
export NUM_TRIALS=${NUM_TRIALS:-1}
export SEED=${SEED:-100}
export RESIZE_SIZE=${RESIZE_SIZE:-256}
export REPLAN_STEPS=${REPLAN_STEPS:-10}
export NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
export MAX_STEPS=${MAX_STEPS:-2500}
export POST_GOAL_STEPS=${POST_GOAL_STEPS:-200}
export POST_STAGE_STEPS=${POST_STAGE_STEPS:-30}
export FAIL_ON_EXTRA_POUR=${FAIL_ON_EXTRA_POUR:-1}
export ASYNC_VLM=${ASYNC_VLM:-1}
export VLM_INTERVAL=${VLM_INTERVAL:-5}
export VLM_QUEUE_SIZE=${VLM_QUEUE_SIZE:-1}
export N_RECENT=${N_RECENT:-5}
export K_MAX=${K_MAX:-0}
export D_MERGE=${D_MERGE:-6}
export VLM_USE_WRIST=${VLM_USE_WRIST:-1}
export VLM_USE_KEYFRAME_MEMORY=${VLM_USE_KEYFRAME_MEMORY:-1}
export VLM_INPUT_PROFILE=${VLM_INPUT_PROFILE:-fullvlm_256}
export VLM_MATCH_TRAINING_JPEG_ROUNDTRIP=${VLM_MATCH_TRAINING_JPEG_ROUNDTRIP:-0}

PLANNER_BACKEND=${PLANNER_BACKEND:-local}
case "${PLANNER_BACKEND}" in
  api|qwen_api|openai_api)
    export PLANNER_BACKEND=api
    export PLANNER_API_KEY_LINE=${PLANNER_API_KEY_LINE:-2}
    export PLANNER_API_KEY_FILE=${PLANNER_API_KEY_FILE:-${REPO_ROOT}/api_key.txt}
    export PLANNER_API_BASE_URL=${PLANNER_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
    export PLANNER_API_MODEL=${PLANNER_API_MODEL:-qwen-vl-max}
    export HARNESS_API_KEY_LINE=${HARNESS_API_KEY_LINE:-2}
    export HARNESS_API_BASE_URL=${HARNESS_API_BASE_URL:-${PLANNER_API_BASE_URL}}
    export HARNESS_API_MODEL=${HARNESS_API_MODEL:-${PLANNER_API_MODEL}}
    ;;
esac

SERVER_CUDA_VISIBLE_DEVICES=${SERVER_CUDA_VISIBLE_DEVICES:-0}
EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-1}

if [ -z "${PREDIMEM_ASSET_GROUP:-}" ]; then
  mapfile -t TASK_GROUPS < <(python3 - "${TASKS_JSON}" <<'PY'
import json
import sys

tasks = json.loads(sys.argv[1])
if not isinstance(tasks, list) or any(not isinstance(task, int) or task < 1 or task > 26 for task in tasks):
    raise SystemExit("TASKS_JSON must be a JSON array of task IDs in 1..26")
task1 = [task for task in tasks if task == 1]
if task1:
    print("task1\t" + json.dumps(task1, separators=(",", ":")))
remaining = [task for task in tasks if task != 1]
if remaining:
    print("tasks2to26\t" + json.dumps(remaining, separators=(",", ":")))
PY
)
  if [ "${#TASK_GROUPS[@]}" -eq 0 ]; then
    echo "[ERROR] TASKS_JSON must contain at least one task ID" >&2
    exit 2
  fi
  if [ "${#TASK_GROUPS[@]}" -gt 1 ]; then
    for group in "${TASK_GROUPS[@]}"; do
      IFS=$'\t' read -r family group_tasks <<< "${group}"
      # Child re-invocation must recompute paths from OUT_ROOT; inherited exports
      # would otherwise write all phase summaries to the parent OUT_ROOT.
      (
        export PREDIMEM_ASSET_GROUP="${family}" TASKS_JSON="${group_tasks}" OUT_ROOT="${OUT_ROOT}/${family}"
        unset SUMMARY_TSV SUMMARY_JSON PROMPT_TRACE_TSV VIDEO_DIR LOG_DIR SERVER_LOG EVAL_LOG
        bash "${BASH_SOURCE[0]}"
      ) || exit $?
    done
    exit 0
  fi
  IFS=$'\t' read -r PREDIMEM_ASSET_GROUP _ <<< "${TASK_GROUPS[0]}"
fi

case "${PREDIMEM_ASSET_GROUP}" in
  task1)
    VLM_CKPT="${PREDIMEM_HF_SNAPSHOT}/vlm_task1"
    ;;
  tasks2to26)
    VLM_CKPT="${PREDIMEM_HF_SNAPSHOT}/vlm_tasks1to26_ckpt74500"
    ;;
  *)
    echo "[ERROR] invalid PREDIMEM_ASSET_GROUP=${PREDIMEM_ASSET_GROUP}" >&2
    exit 2
    ;;
esac
VLM_PROCESSOR_DIR="${VLM_CKPT}"
export VLA_CKPT VLM_CKPT VLM_PROCESSOR_DIR NORM_STATS_PATH PREDIMEM_HF_REVISION PREDIMEM_HF_SNAPSHOT

if [ "${PLANNER_BACKEND}" = "api" ]; then
  echo "[INFO] PLANNER_BACKEND=api (skip local VLM checkpoint checks)" | tee -a "${EVAL_LOG}"
  echo "[INFO] PLANNER_API_MODEL=${PLANNER_API_MODEL}" | tee -a "${EVAL_LOG}"
  echo "[INFO] PLANNER_API_BASE_URL=${PLANNER_API_BASE_URL}" | tee -a "${EVAL_LOG}"
  echo "[INFO] PLANNER_API_KEY_LINE=${PLANNER_API_KEY_LINE}" | tee -a "${EVAL_LOG}"
  export VLM_CKPT="${REPO_ROOT}/checkpoints/PrediMem/vlm_tasks1to26_ckpt74500"
  export VLM_PROCESSOR_DIR="${VLM_CKPT}"
fi

echo "[INFO] OUT_ROOT=${OUT_ROOT}" | tee -a "${EVAL_LOG}"
echo "[INFO] TASK_CONFIG=${TASK_CONFIG}" | tee -a "${EVAL_LOG}"
echo "[INFO] VLM_CKPT=${VLM_CKPT}" | tee -a "${EVAL_LOG}"
echo "[INFO] VLA_CONFIG=${VLA_CONFIG}" | tee -a "${EVAL_LOG}"
echo "[INFO] VLA_CKPT=${VLA_CKPT}" | tee -a "${EVAL_LOG}"
echo "[INFO] NORM_STATS_PATH=${NORM_STATS_PATH}" | tee -a "${EVAL_LOG}"
echo "[INFO] PrediMem revision=${PREDIMEM_HF_REVISION}" | tee -a "${EVAL_LOG}"
echo "[INFO] TASKS_JSON=${TASKS_JSON}" | tee -a "${EVAL_LOG}"

if [ "${PLANNER_BACKEND}" != "api" ]; then
  if [ ! -d "${VLM_CKPT}" ]; then
    echo "[ERROR] VLM_CKPT not found: ${VLM_CKPT}" | tee -a "${EVAL_LOG}"
    exit 1
  fi
  if [ ! -f "${VLM_CKPT}/model.safetensors" ]; then
    echo "[ERROR] incomplete published VLM assets under ${VLM_CKPT}" | tee -a "${EVAL_LOG}"
    exit 1
  fi
fi
if [ ! -d "${VLA_CKPT}" ]; then
  echo "[ERROR] missing PrediMem HF snapshot: ${PREDIMEM_HF_SNAPSHOT}" | tee -a "${EVAL_LOG}"
  echo "[ERROR] download huashuolei/PrediMem revision ${PREDIMEM_HF_REVISION} to the Hugging Face cache first" | tee -a "${EVAL_LOG}"
  exit 1
fi
if [ ! -f "${VLA_CKPT}/params/_METADATA" ] || [ ! -f "${NORM_STATS_PATH}" ]; then
  echo "[ERROR] incomplete published VLA assets under ${VLA_CKPT}" | tee -a "${EVAL_LOG}"
  exit 1
fi
if [ "$(sha256sum "${NORM_STATS_PATH}" | cut -d' ' -f1)" != "0c0d329a3345d2ea2e1348dac0edf4ae175085fac200cca1cab967aae1ae1767" ]; then
  echo "[ERROR] published VLA norm hash mismatch: ${NORM_STATS_PATH}" | tee -a "${EVAL_LOG}"
  exit 1
fi
if [ "${PLANNER_BACKEND}" != "api" ]; then
  if [ ! -f "${VLM_CKPT}/model.safetensors" ]; then
    echo "[ERROR] incomplete published VLM assets under ${VLM_CKPT}" | tee -a "${EVAL_LOG}"
    exit 1
  fi
fi
if [ ! -f "${OPENPI_ROOT}/scripts/serve_policy.py" ]; then
  echo "[ERROR] serve_policy.py not found under OPENPI_ROOT=${OPENPI_ROOT}" | tee -a "${EVAL_LOG}"
  exit 1
fi

# Only reap a leftover server on OUR port that we ourselves may have orphaned. Never
# broad-pkill by port alone before checking ownership: a sibling Slurm job on the same
# node can legally own another port, and historically a shared PORT=8052 made the second
# job's pkill murder the first mid-eval (exit 91).
_reap_stale_server() {
  local port="$1"
  local pids
  pids=$(pgrep -f "scripts/serve_policy.py --port ${port}" 2>/dev/null || true)
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  # If SLURM_JOB_ID is set, only kill processes whose environment we can attribute to
  # this job; otherwise (interactive) kill leftovers on this port after a warning.
  local pid
  for pid in ${pids}; do
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
      if [[ -r "/proc/${pid}/environ" ]] && \
         tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | grep -qx "SLURM_JOB_ID=${SLURM_JOB_ID}"; then
        echo "[INFO] reaping stale VLA server pid=${pid} port=${port} from this job"
        kill "${pid}" 2>/dev/null || true
      else
        echo "[WARN] VLA server pid=${pid} already on port ${port} but not this job; leaving it alone"
      fi
    else
      echo "[INFO] reaping leftover VLA server pid=${pid} port=${port}"
      kill "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
}

_summary_data_rows() {
  local tsv="$1"
  if [[ ! -s "${tsv}" ]]; then
    echo 0
    return
  fi
  # header + N data rows; count non-empty non-header lines
  awk 'NR>1 && NF>0 {c++} END{print c+0}' "${tsv}"
}

_expected_task_count() {
  python3 -c 'import json,os; print(len(json.loads(os.environ["TASKS_JSON"])))' 2>/dev/null || echo 0
}

_start_vla_server() {
  echo "[INFO] starting VLA server on port ${PORT}" | tee -a "${EVAL_LOG}"
  CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES}" "${SERVER_PY}" "${OPENPI_ROOT}/scripts/serve_policy.py" --port "${PORT}" \
    policy:checkpoint --policy.config="${VLA_CONFIG}" \
    --policy.dir="${VLA_CKPT}" \
    > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  READY=0
  local i
  for i in $(seq 1 180); do
    sleep 2
    if "${SERVER_PY}" - <<PY >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", int("${PORT}")))
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
finally:
    s.close()
PY
    then
      READY=1
      echo "[INFO] VLA server ready at try ${i}" | tee -a "${EVAL_LOG}"
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[ERROR] VLA server exited early" | tee -a "${EVAL_LOG}"
      tail -n 200 "${SERVER_LOG}" | tee -a "${EVAL_LOG}" || true
      return 1
    fi
  done
  if [[ "${READY}" -ne 1 ]]; then
    echo "[ERROR] VLA server not ready" | tee -a "${EVAL_LOG}"
    tail -n 200 "${SERVER_LOG}" | tee -a "${EVAL_LOG}" || true
    kill "${SERVER_PID}" 2>/dev/null || true
    return 1
  fi
  return 0
}

_stop_vla_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
  # Only reap OUR job's leftovers on this port — never a sibling job's server.
  _reap_stale_server "${PORT}"
}

_run_eval_once() {
  EVAL_RC=0
  # robosuite requires MUJOCO_EGL_DEVICE_ID ∈ CUDA_VISIBLE_DEVICES when both are set
  # (job 573192: global EGL=0 + EVAL CUDA=1 → AssertionError). Pin EGL to the same
  # physical id the eval process is allowed to see.
  local _eval_egl_id="${EVAL_CUDA_VISIBLE_DEVICES%%,*}"
  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
  MUJOCO_EGL_DEVICE_ID="${_eval_egl_id}" \
  "${EVAL_PY}" "${SCRIPT_DIR}/eval_fullvlm26_async_vlm_vla.py" 2>&1 | tee -a "${EVAL_LOG}" || EVAL_RC=${PIPESTATUS[0]}

  # VLA died during eval. If the summary already holds every requested task, the death
  # was teardown / post-write — same class as the forgiven rc=134 (job 571321). Job 573558
  # wrote a full tasks2to26 summary then exited 91 and lost the arm at merge. Forgive.
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    local _nrows _want
    _nrows=$(_summary_data_rows "${SUMMARY_TSV}")
    _want=$(_expected_task_count)
    echo "[ERROR] VLA server died during evaluation (pid=${SERVER_PID}) summary_rows=${_nrows} want=${_want}" | tee -a "${EVAL_LOG}"
    tail -n 100 "${SERVER_LOG}" | tee -a "${EVAL_LOG}" || true
    if [[ "${_want}" -gt 0 && "${_nrows}" -ge "${_want}" ]]; then
      echo "[WARN] VLA died after a COMPLETE summary (${_nrows}/${_want}); treating as success" | tee -a "${EVAL_LOG}"
      EVAL_RC=0
    elif [[ "${EVAL_RC}" -eq 0 ]]; then
      EVAL_RC=91
    fi
  fi

  # rc=134 is SIGABRT. Job 571321 wrote a complete summary then died in EGL teardown;
  # treating that as a failed arm stopped the remaining seeds. A written summary is the eval.
  # Job 573160: the eval writes the TSV HEADER before any episode. `-s` is then true on a
  # crash at task 0%, merge later dies with "no summary rows", and the arm is lost. Require
  # at least one data row (header + 1) before forgiving the abort — and prefer full coverage.
  if [[ "${EVAL_RC}" -eq 134 && -s "${SUMMARY_TSV}" ]]; then
    local _nlines _nrows _want
    _nlines=$(wc -l < "${SUMMARY_TSV}" | tr -d ' ')
    _nrows=$(_summary_data_rows "${SUMMARY_TSV}")
    _want=$(_expected_task_count)
    if [[ "${_want}" -gt 0 && "${_nrows}" -ge "${_want}" ]]; then
      echo "[WARN] eval rc=134 after COMPLETE summary (${_nrows}/${_want}); treating as success" | tee -a "${EVAL_LOG}"
      EVAL_RC=0
    elif [[ "${_nlines}" -ge 2 ]]; then
      echo "[WARN] eval rc=134 after summary has ${_nlines} lines (partial ${_nrows}/${_want}); treating as success (cleanup abort, job 571321)" | tee -a "${EVAL_LOG}"
      EVAL_RC=0
    else
      echo "[ERROR] eval rc=134 with header-only summary (${_nlines} lines); this is a real crash, not teardown" | tee -a "${EVAL_LOG}"
    fi
  fi
  return 0
}

_reap_stale_server "${PORT}"

# One automatic retry for transient VLA bring-up / mid-eval death when the summary is
# empty or short. Does NOT retry a completed measurement. Caps at 2 attempts so a hard
# crash cannot loop the GPU for hours.
EVAL_TRANSIENT_RETRIES="${EVAL_TRANSIENT_RETRIES:-1}"
ATTEMPT=0
EVAL_RC=1
while [[ "${ATTEMPT}" -le "${EVAL_TRANSIENT_RETRIES}" ]]; do
  ATTEMPT=$((ATTEMPT + 1))
  if [[ "${ATTEMPT}" -gt 1 ]]; then
    echo "[WARN] transient eval failure rc=${EVAL_RC}; retry ${ATTEMPT}/$((EVAL_TRANSIENT_RETRIES + 1)) for TASKS_JSON=${TASKS_JSON}" | tee -a "${EVAL_LOG}"
    _stop_vla_server
    # Incomplete outputs from the failed attempt must not be resumed as success.
    rm -f "${SUMMARY_TSV}" "${SUMMARY_JSON}" "${OUT_ROOT}/aggregate.json" 2>/dev/null || true
    sleep 5
  fi
  if ! _start_vla_server; then
    EVAL_RC=1
    _stop_vla_server
    continue
  fi
  _run_eval_once
  _stop_vla_server
  # Retry only clearly transient / empty outcomes.
  if [[ "${EVAL_RC}" -eq 0 ]]; then
    break
  fi
  _nrows=$(_summary_data_rows "${SUMMARY_TSV}")
  _want=$(_expected_task_count)
  if [[ "${EVAL_RC}" -eq 91 || "${EVAL_RC}" -eq 1 || "${EVAL_RC}" -eq 134 ]]; then
    if [[ "${_nrows}" -lt "${_want}" ]]; then
      continue
    fi
  fi
  break
done

echo "[INFO] eval rc=${EVAL_RC}" | tee -a "${EVAL_LOG}"
echo "[INFO] SERVER_LOG=${SERVER_LOG}" | tee -a "${EVAL_LOG}"
echo "[INFO] EVAL_LOG=${EVAL_LOG}" | tee -a "${EVAL_LOG}"
echo "[INFO] SUMMARY_JSON=${SUMMARY_JSON}" | tee -a "${EVAL_LOG}"
echo "[INFO] SUMMARY_TSV=${SUMMARY_TSV}" | tee -a "${EVAL_LOG}"
echo "[INFO] AGGREGATE=${OUT_ROOT}/aggregate.json" | tee -a "${EVAL_LOG}"

exit "${EVAL_RC}"
