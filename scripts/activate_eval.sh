#!/usr/bin/env bash
# Activate eval venv (VLM planner + LIBERO/MuJoCo simulation).
# Usage: source scripts/activate_eval.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_common.sh"
robomem_setup_cache || return 1 2>/dev/null || exit 1

EVAL_VENV="${EVAL_VENV:-${ROOT}/.venv}"
if [[ ! -f "${EVAL_VENV}/bin/activate" ]]; then
  echo "[ERROR] Missing venv at ${EVAL_VENV}. Run: bash scripts/setup_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${EVAL_VENV}/bin/activate"
cd "${ROOT}"
echo "[OK] RoboMemArena eval env ($(python --version))"
echo "[OK] OPENPI_INFERENCE_ROOT=${OPENPI_INFERENCE_ROOT}"
echo "[OK] TARGET_LIBERO_PATH=${TARGET_LIBERO_PATH}"
echo "[OK] Cache -> ${CACHE_ROOT} / ${ROBOMEMARENA_CACHE}"
