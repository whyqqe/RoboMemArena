#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / arm: DIAL
# Diagnosis-Informed Action Law.
#
# ONE scheduler for planner + VLA strategy + proactive memory + harness, driven by a
# posterior over WHY credit is not moving.  See dial/README.md for the evidence.
#
# What this arm does NOT do, and that is the point:
#   * it does not rewrite the VLA prompt (job 582641: -11..-22pp)   -> bind.py observer only
#   * it does not wrap infer_primitive_via_api (BOLT's hook was inert for a whole arm)
#   * it has no rejecting action at all: {RESAMPLE, QUERY, RESYNC, PERSIST}
#     so the same-obligation absorbing loop cannot be entered
#
# Phase-1 defaults (matches the archived evidence):
#   archived reward prior      ON   (dial/attribution.py over results/)
#   server-side redaction      ON   (PMH_REDACT_STAGE=1, same as every sibling arm)
#   reward-model bandit        ON   (stratified Thompson over attempt cells)
#   proactive memory channel   ON   (QUERY is an action with a cost, not a rule)
# =========================================================================================

: "${ROOT:?arms/dial.sh requires ROOT to be exported}"

# shellcheck source=nomem.sh
source "$(dirname "${BASH_SOURCE[0]}")/nomem.sh"

export MEMEXP_DIR="${ROOT}/experiments/mem_efficacy"
_dial_pysite="${MEMEXP_DIR}/pysite"
case ":${PYTHONPATH:-}:" in
  *":${_dial_pysite}:"*) ;;
  *) export PYTHONPATH="${_dial_pysite}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
unset _dial_pysite

# DIAL owns the planner binding — clear every sibling that binds the same object.
unset MEMEXP_EVMEM || true
unset MEMEXP_EVMEM_SR || true
unset MEMEXP_EVMEM_PIC || true
unset MEMEXP_AOM || true
unset MEMEXP_BOLT || true
unset MEMEXP_PULL_ENABLE || true
unset MEMEXP_ER || true

export MEMEXP_DIAL=1
export MEMEXP_DIAL_REPORT="${MEMEXP_DIAL_REPORT:-${OUT_ROOT:-/tmp}/memexp_dial_report.json}"

# --- telemetry -----------------------------------------------------------------------------
# `_build_messages` is the one guaranteed-live lifecycle point (the BOLT v1 lesson: a hook
# that only fires on `reset_episode` archives nothing).
export MEMEXP_DIAL_REPORT_EVERY="${MEMEXP_DIAL_REPORT_EVERY:-25}"

export PMH_REDACT_STAGE=1
export MEMEXP_ARM_NAME="dial"
export MEMEXP_ARM_CLASS="treatment_nomem_plus_dial"
export MEMEXP_EXPECTED_DIFF="PYTHONPATH PMH_REDACT_STAGE"
