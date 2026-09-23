#!/usr/bin/env bash
# =========================================================================================
# Profile: H0  —  memory-causal experiment profile (Stage 1 of next_plan.md)
#
# WHAT THIS IS
#   Same Planner / VLA / tasks / write program as Hlegacy, but EVERY behavioural recovery
#   channel is OFF. Memory arms still WRITE the same dense frame store; they only change
#   what reaches the Planner. Stage judgment is for offline scoring only.
#
# WHY EACH SWITCH IS HERE (scars, not preferences)
#   HARNESS_MAX_RETRIES=0
#     A retry is a second chance. Pairing a memory arm that happens to fail early against a
#     baseline that gets two more attempts is not a memory effect. Job 587161 already showed
#     how often the stall ladder alone rewrites the trajectory.
#   HARNESS_SUBTASK_OVERRIDE=0 / HARNESS_FORCE_VLM_REPLAN=0 / HARNESS_STAGE_CHECKPOINT=0
#     These are CONTROL actions, not evidence. Leaving them on in H0 would let the harness
#     rewrite the VLA prompt or rewind state — precisely the confound next_plan forbids.
#   HARNESS_PERSIST_MEMORY=0
#     Cross-attempt memory is a second trial's worth of history. H0 trials start blank.
#   MEM_STAGE_ANCHOR=0
#     Stage-anchor pins keyframes from the evaluator's BDDL check_fn — i.e. ground-truth
#     stage completion. That is an oracle signal. H0 replaces it with the dense store
#     (MEM_KF_STORE_INTERVAL, set by the memory arms) plus observable subtask changes.
#   HARNESS_CROSS_TASK_LTM* / HARNESS_STAGE_LEDGER_GUARD* purged
#     Opt-in RSI / ledger channels must not leak into a memory-causal run. The bare prefixes
#     cover bank paths and caps added later.
#
# USAGE (AFTER sourcing an arm — order is load-bearing)
#   source experiments/mem_efficacy/arms/pullmem.sh
#   source experiments/mem_efficacy/profiles/h0.sh
# =========================================================================================

export MEMPROF_NAME="h0"
export MEMPROF_ALLOWS_RETRY=0
export MEMPROF_ALLOWS_STAGE_ANCHOR=0

# --- recovery channels OFF ----------------------------------------------------------------
export HARNESS_MAX_RETRIES=0
export HARNESS_SUBTASK_OVERRIDE=0
export HARNESS_STAGE_CHECKPOINT=0
export HARNESS_SMART_RETRY=0
export HARNESS_FORCE_VLM_REPLAN=0
export HARNESS_RELEASE_ON_RETRY=0
export HARNESS_PERSIST_MEMORY=0
# Stall telemetry may still fire (for audit), but with override/replan off it cannot act.
# Leave HARNESS_STALL_STEPS as the arm set it.

# --- oracle stage-anchor OFF; observable dense store stays as the arm declared it ---------
export MEM_STAGE_ANCHOR=0

# --- purge opt-in RSI / ledger channels that live outside MEMEXP_* ------------------------
for _pv in $(compgen -v 2>/dev/null | grep -E '^(HARNESS_CROSS_TASK_LTM|HARNESS_STAGE_LEDGER_GUARD)' || true); do
  unset "${_pv}"
done
unset _pv
export HARNESS_CROSS_TASK_LTM=0
export HARNESS_STAGE_LEDGER_GUARD=0

export MEMPROF_EXPECT_HARNESS_MAX_RETRIES=0
export MEMPROF_EXPECT_HARNESS_SUBTASK_OVERRIDE=0
export MEMPROF_EXPECT_HARNESS_STAGE_CHECKPOINT=0
