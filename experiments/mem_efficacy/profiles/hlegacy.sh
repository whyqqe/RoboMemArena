#!/usr/bin/env bash
# =========================================================================================
# Profile: Hlegacy  —  "old-result reproduction" profile (Stage 1 of next_plan.md)
#
# WHAT THIS IS
#   The 2026-09-18 pull/push/nomem configuration, including harness retry / checkpoint /
#   subtask override and MEM_STAGE_ANCHOR=1 on the memory arms. This profile does NOT
#   change anything relative to the current arms/*.sh; it exists so the stage-1 runner
#   can name the profile it is running and refuse to mix it with H0 in one job.
#
# WHAT THIS IS NOT
#   Not the causal memory experiment. Recovery is ON here, so any score difference between
#   arms is entangled with retry opportunity. Use H0 for causal claims.
#
# USAGE (after sourcing an arm)
#   source experiments/mem_efficacy/profiles/hlegacy.sh
# =========================================================================================

export MEMPROF_NAME="hlegacy"
export MEMPROF_ALLOWS_RETRY=1
export MEMPROF_ALLOWS_STAGE_ANCHOR=1

# Identity only. The arm scripts already set the recovery knobs; re-asserting them here
# would hide a future arm that forgot them. The stage-1 preflight asserts the values below
# match what the arm left in the environment.
export MEMPROF_EXPECT_HARNESS_MAX_RETRIES=2
export MEMPROF_EXPECT_HARNESS_SUBTASK_OVERRIDE=1
export MEMPROF_EXPECT_HARNESS_STAGE_CHECKPOINT=1
