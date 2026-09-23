#!/usr/bin/env bash
#SBATCH --job-name=rma-stage1
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/RoboMemArena/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/RoboMemArena/outputs/slurm/%x-%j.err

# =========================================================================================
# Stage-1 base harness runner (next_plan.md)
#
# Defaults are intentionally CHEAP and SAFE:
#   MEMPROF=h0                 memory-causal profile (no retry / no stage-anchor oracle)
#   PLANNER_API_MODEL=gpt-6-luna   cheaper CloseAI planner (parity ~ gemini-3.8-flash)
#   TASKS_JSON=[5]             single hard task smoke
#   NUM_TRIALS=1               one trial; expand only after smoke is green
#   EARLY_STOP_ON_QUOTA=1      kill evaluator on API_QUOTA_EXHAUSTED
#   PLANNER_MAX_CONSECUTIVE_FAILURES=3
#   unique OUT_BASE            includes job id; collision lock inside run_26x1.sbatch
#
# SUBMIT (from repo root)
#   # cheapest smoke: H0 + pullmem + task5 x1
#   sbatch experiments/mem_efficacy/run_stage1.sbatch
#
#   # Hlegacy reproduction line (retry ON; still task5 x1 until budget is confirmed)
#   MEMPROF=hlegacy ARM_OVERRIDE="nomem pushmem pullmem" \
#     sbatch experiments/mem_efficacy/run_stage1.sbatch
#
#   # expand trials only after smoke
#   NUM_TRIALS_SCOPE=10 ARM_OVERRIDE="nomem pushmem pullmem" \
#     sbatch experiments/mem_efficacy/run_stage1.sbatch
# =========================================================================================

set -euo pipefail

ROOT="/project/peilab/why/RoboMemArena"
export ROOT
cd "${ROOT}"
EXP_DIR="${ROOT}/experiments/mem_efficacy"
JOB_TAG="${SLURM_JOB_ID:-manual}"

# --- Stage-1 defaults -----------------------------------------------------------------
# sbatch inherits the submitting shell. An interactive probe that exported
# PLANNER_API_MAX_TOKENS=256 would otherwise shrink the real run (job 609328).
# Only neutralize undersized leftovers; intentional ARM_OVERRIDE / MEMPROF still win.
_tok="${PLANNER_API_MAX_TOKENS:-}"
if [[ -n "${_tok}" && "${_tok}" =~ ^[0-9]+$ && "${_tok}" -lt 1024 ]]; then
  echo "[WARN] ignoring inherited PLANNER_API_MAX_TOKENS=${_tok} (<1024); using Stage-1 default"
  unset PLANNER_API_MAX_TOKENS
fi
unset _tok

export MEMPROF="${MEMPROF:-h0}"
export PLANNER_API_MODEL="${PLANNER_API_MODEL:-gpt-6-luna}"
export HARNESS_API_MODEL="${HARNESS_API_MODEL:-${PLANNER_API_MODEL}}"
# Reasoning models burn completion tokens on CoT; 2048 is enough for smoke JSON;
# raise to 4096 only when a full grid is approved.
export PLANNER_API_MAX_TOKENS="${PLANNER_API_MAX_TOKENS:-2048}"
export HARNESS_API_MAX_TOKENS="${HARNESS_API_MAX_TOKENS:-512}"
export PLANNER_MAX_CONSECUTIVE_FAILURES="${PLANNER_MAX_CONSECUTIVE_FAILURES:-3}"
export EARLY_STOP_ON_QUOTA="${EARLY_STOP_ON_QUOTA:-1}"

export ARM_OVERRIDE="${ARM_OVERRIDE:-pullmem}"
export TASKS_JSON_SCOPE="${TASKS_JSON_SCOPE:-[5]}"
export SEED_LIST="${SEED_LIST:-100}"
export NUM_TRIALS_SCOPE="${NUM_TRIALS_SCOPE:-1}"

# Unique output root: profile + model + job id. Never reuse a previous OUT_BASE.
_safe_model="$(printf '%s' "${PLANNER_API_MODEL}" | tr '/: ' '___')"
export OUT_BASE_OVERRIDE="${OUT_BASE_OVERRIDE:-${EXP_DIR}/results/stage1_${MEMPROF}_${_safe_model}_${JOB_TAG}}"
unset _safe_model

mkdir -p "${ROOT}/outputs/slurm" "${OUT_BASE_OVERRIDE}"

echo "=============================================================================="
echo "Stage-1 runner"
echo "  MEMPROF                         = ${MEMPROF}"
echo "  PLANNER_API_MODEL               = ${PLANNER_API_MODEL}"
echo "  PLANNER_API_MAX_TOKENS          = ${PLANNER_API_MAX_TOKENS}"
echo "  PLANNER_MAX_CONSECUTIVE_FAILURES= ${PLANNER_MAX_CONSECUTIVE_FAILURES}"
echo "  EARLY_STOP_ON_QUOTA             = ${EARLY_STOP_ON_QUOTA}"
echo "  ARM_OVERRIDE                    = ${ARM_OVERRIDE}"
echo "  TASKS_JSON_SCOPE                = ${TASKS_JSON_SCOPE}"
echo "  SEED_LIST                       = ${SEED_LIST}"
echo "  NUM_TRIALS_SCOPE                = ${NUM_TRIALS_SCOPE}"
echo "  OUT_BASE_OVERRIDE               = ${OUT_BASE_OVERRIDE}"
echo "=============================================================================="

# --- Preflight (no GPU, no paid planner call beyond the model probe) --------------------
EVAL_PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${EVAL_PY}" ]]; then
  EVAL_PY="$(command -v python3)"
fi

echo
echo "[STAGE1] profile preflight"
"${EVAL_PY}" "${EXP_DIR}/profile_preflight.py" \
  --arms nomem pushmem pullmem \
  --profiles "${MEMPROF}"
echo "[STAGE1] profile preflight OK"

echo
echo "[STAGE1] planner model probe (tiny vision call)"
export PLANNER_API_KEY_FILE="${PLANNER_API_KEY_FILE:-${ROOT}/closeai_api.txt}"
export PLANNER_API_BASE_URL="${PLANNER_API_BASE_URL:-https://api.closeai-asia.com/v1}"
"${EVAL_PY}" "${EXP_DIR}/planner_model_probe.py"
echo "[STAGE1] planner model probe OK"

# Hand off to the shared mem_efficacy runner. Its #SBATCH lines are comments when executed
# via bash; all Stage-1 knobs are already in the environment.
echo
echo "[STAGE1] handing off to run_26x1.sbatch"
exec bash "${EXP_DIR}/run_26x1.sbatch"
