#!/usr/bin/env bash
# ECHO: Planner + frozen VLA + Harness + proactive prospective evidence.
# Existing nomem runner/config preserved; ECHO alone owns the planner hook.
: "${ROOT:?arms/echo.sh requires ROOT}"
source "$(dirname "${BASH_SOURCE[0]}")/nomem.sh"
export MEMEXP_DIR="${ROOT}/experiments/mem_efficacy"
case ":${PYTHONPATH:-}:" in
  *":${MEMEXP_DIR}/pysite:"*) ;;
  *) export PYTHONPATH="${MEMEXP_DIR}/pysite${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
unset MEMEXP_DIAL MEMEXP_BOLT MEMEXP_EVMEM MEMEXP_AOM MEMEXP_EVMEM_SR MEMEXP_EVMEM_PIC MEMEXP_PULL_ENABLE MEMEXP_ER || true
export MEMEXP_ECHO=1
export MEMEXP_ECHO_REPORT="${MEMEXP_ECHO_REPORT:-${OUT_ROOT:-/tmp}/memexp_echo_report.json}"
export PMH_REDACT_STAGE=1
export MEMEXP_ARM_NAME=echo
export MEMEXP_ARM_CLASS=treatment_nomem_plus_echo
export MEMEXP_EXPECTED_DIFF="PYTHONPATH PMH_REDACT_STAGE"
