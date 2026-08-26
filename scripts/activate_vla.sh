#!/usr/bin/env bash
# Activate VLA policy-server venv (OpenPI minimal runtime).
# Usage: source scripts/activate_vla.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_common.sh"
robomem_setup_cache || return 1 2>/dev/null || exit 1

VLA_VENV="${VLA_VENV:-${OPENPI_ROOT}/.venv}"
if [[ ! -f "${VLA_VENV}/bin/activate" ]]; then
  echo "[ERROR] Missing venv at ${VLA_VENV}. Run: bash scripts/setup_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${VLA_VENV}/bin/activate"
cd "${OPENPI_ROOT}"
echo "[OK] RoboMemArena VLA env ($(python --version))"
echo "[OK] OPENPI_ROOT=${OPENPI_ROOT}"
echo "[OK] Cache -> ${CACHE_ROOT} / ${ROBOMEMARENA_CACHE}"
