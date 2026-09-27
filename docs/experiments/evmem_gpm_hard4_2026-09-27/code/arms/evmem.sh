#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / arm: EVMEM (GPM L0+L1)
# Fluent≈nomem; NEED predicates; C-gate; gated evidence with tagged frames.
# =========================================================================================

: "${ROOT:?arms/evmem.sh requires ROOT to be exported}"

# shellcheck source=nomem.sh
source "$(dirname "${BASH_SOURCE[0]}")/nomem.sh"

export MEMEXP_DIR="${ROOT}/experiments/mem_efficacy"
_evmem_pysite="${MEMEXP_DIR}/pysite"
case ":${PYTHONPATH:-}:" in
  *":${_evmem_pysite}:"*) ;;
  *) export PYTHONPATH="${_evmem_pysite}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
unset _evmem_pysite

export MEMEXP_EVMEM=1
export MEMEXP_EVMEM_REPORT="${MEMEXP_EVMEM_REPORT:-${OUT_ROOT:-/tmp}/memexp_evmem_report.json}"
export MEMEXP_EVMEM_MAX_ROUNDS="${MEMEXP_EVMEM_MAX_ROUNDS:-2}"
export MEMEXP_EVMEM_LOOKAHEAD="${MEMEXP_EVMEM_LOOKAHEAD:-4}"
export MEMEXP_EVMEM_EVIDENCE_CAP="${MEMEXP_EVMEM_EVIDENCE_CAP:-6}"
export MEMEXP_EVMEM_GATE_ATTEMPTS="${MEMEXP_EVMEM_GATE_ATTEMPTS:-3}"
export MEMEXP_EVMEM_GATE_STALL="${MEMEXP_EVMEM_GATE_STALL:-3}"
export MEMEXP_EVMEM_ATTRACTOR_REPEAT="${MEMEXP_EVMEM_ATTRACTOR_REPEAT:-2}"

export PMH_REDACT_STAGE=1
export MEMEXP_ARM_NAME="evmem"
export MEMEXP_ARM_CLASS="treatment_nomem_plus_evmem"
export MEMEXP_EXPECTED_DIFF="PYTHONPATH PMH_REDACT_STAGE"
