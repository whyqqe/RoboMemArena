#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / arm: EVMEM-AOM  (Arbitrated Obligation Memory)
#
# WHAT THIS IS
#   One obligation type, one arbitration law, one board, and admissibility DERIVED from the
#   obligation graph. It is the unification of two previously separate designs:
#     * EvMem-GPM  (`arms/evmem.sh`) supplies the SCHEDULER. Its `decide_mode` gates evidence on
#       behavioural error (attempts / stall / attractor) -- a signal that does not require the
#       Planner to notice it is stuck, which is the failure the pull arm documents as 85%.
#     * memexp_er   (`arms/pullmem_er.sh`) supplies the TYPES. Its `_ADDRESS_KINDS` states how
#       the truth of an address is obtained, which is the same distinction GPM's
#       `is_open_close_stage` makes for stages.
#
# WHY NOT JUST CONCATENATE THEM
#   Because that yields two ledgers, two schedulers and three prompt blocks that each re-decide
#   policy -- exactly the failure PIC-MEM hit in all four of its versions (v1 PIN flood / v2
#   unreachable operators / v3 evidence starvation / v4 net-zero over GPM). AOM's claim is that
#   GPM's and ER's behaviours are two DEGENERATE POINTS of one law:
#       if SETTLED -> advance;  if ACTUAL and not sat_act -> act;  if DERIVED -> derive;
#       if not sat_ret -> retrieve;  else -> stagnant
#   GPM = this law at {ACTUAL, PERCEPTUAL} x {retrieve = harness auto-look}. ER = this law with
#   the `act` branch removed and `retrieve` delegated to the model.
#
# THE THREE AXES THIS ARM MEASURES (each has an ablation knob, so it can be switched off alone)
#   MEMEXP_AOM_DERIVED=0     removes the DERIVED mode. This is the headline claim: without it
#                            the "already poured twice?" question has no representation at all,
#                            which is why `extra_pour_detected` is 0 on EVERY archived task-8 arm.
#   MEMEXP_AOM_GRAPH_GATE=0  leaves only the empty/label checks, so the arm runs GPM's regex-free
#                            gate. Measures what the obligation graph buys over a pattern table.
#   MEMEXP_AOM_FLOOR=0       removes safety floor F. Kept ON by default because OFF reproduces
#                            PIC-MEM v3's starved cadence (4 evidence steps/episode vs GPM's 8-15)
#                            and scored 27.8 where GPM scored 66.7 on the same seeds.
#
# HONEST EXPECTATION ON TASK 8
#   Task 8's scored stages are ACTUAL, so neither PERCEPTUAL nor DERIVED is exercised. AOM should
#   therefore land NEAR GPM (46.7) rather than above it. The informative outputs are the
#   COUNTERS (n_stagnant, n_dep_rejects, derived_values), which turn four failure modes that all
#   render as "score 0" today into four distinguishable ones. A >20pp gain over GPM on n=10 would
#   be evidence of something outside the law and must be investigated rather than reported.
# =========================================================================================

: "${ROOT:?arms/evmem_aom.sh requires ROOT to be exported}"

# shellcheck source=nomem.sh
source "$(dirname "${BASH_SOURCE[0]}")/nomem.sh"

export MEMEXP_DIR="${ROOT}/experiments/mem_efficacy"
_evmem_pysite="${MEMEXP_DIR}/pysite"
case ":${PYTHONPATH:-}:" in
  *":${_evmem_pysite}:"*) ;;
  *) export PYTHONPATH="${_evmem_pysite}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
unset _evmem_pysite

# AOM owns the planner binding. Every sibling memory arm binds the SAME object
# (`harness.api_vlm_planner`), so leaving one of these set would install two ledgers on one
# planner -- the state AOM exists to eliminate.
unset MEMEXP_EVMEM || true
unset MEMEXP_EVMEM_SR || true
unset MEMEXP_EVMEM_PIC || true

export MEMEXP_AOM=1
export MEMEXP_AOM_REPORT="${MEMEXP_AOM_REPORT:-${OUT_ROOT:-/tmp}/memexp_aom_report.json}"

# --- the law's thresholds -----------------------------------------------------------------
# A* and the stall gate are GPM's own values ON PURPOSE: the arbitration must be the only
# difference, so that a cadence change cannot be attributed to a re-tuned gate.
export MEMEXP_AOM_GATE_ATTEMPTS="${MEMEXP_AOM_GATE_ATTEMPTS:-3}"
export MEMEXP_AOM_GATE_STALL="${MEMEXP_AOM_GATE_STALL:-3}"
# R* has no GPM counterpart (GPM never leaves evidence mode). It bounds how long the board keeps
# calling an obligation NEW before it calls it STAGNANT.
export MEMEXP_AOM_RET_MAX="${MEMEXP_AOM_RET_MAX:-6}"
# Re-arm act after this many consecutive stagnant verdicts (0 disables re-arm).
export MEMEXP_AOM_STAG_ACT_REARM="${MEMEXP_AOM_STAG_ACT_REARM:-3}"
export MEMEXP_AOM_ATTRACTOR_REPEAT="${MEMEXP_AOM_ATTRACTOR_REPEAT:-2}"
export MEMEXP_AOM_LOOKAHEAD="${MEMEXP_AOM_LOOKAHEAD:-4}"
export MEMEXP_AOM_EVIDENCE_CAP="${MEMEXP_AOM_EVIDENCE_CAP:-6}"
export MEMEXP_AOM_MAX_ROUNDS="${MEMEXP_AOM_MAX_ROUNDS:-2}"

# --- the three axes ----------------------------------------------------------------------
export MEMEXP_AOM_DERIVED="${MEMEXP_AOM_DERIVED:-1}"
export MEMEXP_AOM_GRAPH_GATE="${MEMEXP_AOM_GRAPH_GATE:-1}"
export MEMEXP_AOM_FLOOR="${MEMEXP_AOM_FLOOR:-1}"

export PMH_REDACT_STAGE=1
export MEMEXP_ARM_NAME="evmem_aom"
export MEMEXP_ARM_CLASS="treatment_nomem_plus_aom"
export MEMEXP_EXPECTED_DIFF="PYTHONPATH PMH_REDACT_STAGE"
