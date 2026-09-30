#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / arm: BOLT-Sync
# Belief–Obligation–Ledger–Telemetry with Clock Synchronization.
#
# Phase-1 defaults:
#   execution-graph seeding ALWAYS on
#   dual-clock sync on (gap window configurable)
#   rule-based evidence router + starvation floor on
#   geometry / safety / vision-verify / short-term token / world-model OFF
# =========================================================================================

: "${ROOT:?arms/bolt.sh requires ROOT to be exported}"

# shellcheck source=nomem.sh
source "$(dirname "${BASH_SOURCE[0]}")/nomem.sh"

export MEMEXP_DIR="${ROOT}/experiments/mem_efficacy"
_bolt_pysite="${MEMEXP_DIR}/pysite"
case ":${PYTHONPATH:-}:" in
  *":${_bolt_pysite}:"*) ;;
  *) export PYTHONPATH="${_bolt_pysite}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
unset _bolt_pysite

# BOLT owns the planner binding — clear sibling memory arms that bind the same object.
unset MEMEXP_EVMEM || true
unset MEMEXP_EVMEM_SR || true
unset MEMEXP_EVMEM_PIC || true
unset MEMEXP_AOM || true
unset MEMEXP_PULL_ENABLE || true
unset MEMEXP_ER || true

export MEMEXP_BOLT=1
export MEMEXP_BOLT_REPORT="${MEMEXP_BOLT_REPORT:-${OUT_ROOT:-/tmp}/memexp_bolt_report.json}"

# --- dual-clock window (seconds) ----------------------------------------------------------
export MEMEXP_BOLT_GAP_MIN="${MEMEXP_BOLT_GAP_MIN:-0}"
export MEMEXP_BOLT_GAP_MAX="${MEMEXP_BOLT_GAP_MAX:-30}"

# --- router / floor -----------------------------------------------------------------------
export MEMEXP_BOLT_ROUTER="${MEMEXP_BOLT_ROUTER:-1}"
export MEMEXP_BOLT_FLOOR="${MEMEXP_BOLT_FLOOR:-1}"
export MEMEXP_BOLT_STARVE_LIMIT="${MEMEXP_BOLT_STARVE_LIMIT:-4}"
export MEMEXP_BOLT_RETRY_BUDGET="${MEMEXP_BOLT_RETRY_BUDGET:-3}"

# --- progress guard (v2) -------------------------------------------------------------------
# Max consecutive attempts on the same obligation with Φ unchanged before an
# instrumental (never-verifiable) node is demoted out of ACTIVE selection.
# A node whose completion the environment can never confirm must not be allowed
# to consume the whole episode: measured on v1, task 22 spent 96% and task 5 62%
# of its planner steps re-emitting one unreachable primitive.
export MEMEXP_BOLT_STALL_LIMIT="${MEMEXP_BOLT_STALL_LIMIT:-2}"

# --- telemetry (v2) ------------------------------------------------------------------------
# v1 archived zero reports because the only write hook hung off `reset_episode`,
# which never observes a live seeded context.  Reports are now flushed from
# `_build_messages` every N planner steps.
export MEMEXP_BOLT_REPORT_EVERY="${MEMEXP_BOLT_REPORT_EVERY:-25}"

# --- Phase-2/3/4 modules (OFF by default; open one at a time) -----------------------------
export MEMEXP_BOLT_GEOMETRY="${MEMEXP_BOLT_GEOMETRY:-0}"
export MEMEXP_BOLT_SAFETY="${MEMEXP_BOLT_SAFETY:-0}"
export MEMEXP_BOLT_VISION_VERIFY="${MEMEXP_BOLT_VISION_VERIFY:-0}"
export MEMEXP_BOLT_STM="${MEMEXP_BOLT_STM:-0}"          # PRISM-style short-term tokens
export MEMEXP_BOLT_WORLD_MODEL="${MEMEXP_BOLT_WORLD_MODEL:-0}"

export PMH_REDACT_STAGE=1
export MEMEXP_ARM_NAME="bolt"
export MEMEXP_ARM_CLASS="treatment_nomem_plus_bolt"
export MEMEXP_EXPECTED_DIFF="PYTHONPATH PMH_REDACT_STAGE"
