#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / arm: NOMEM  (the baseline)
#
# DEFINITION
#   RMA official protocol (26 tasks, seed 100)
#   + HarnessVLA-style recovery Harness
#   + gemini-3.8-flash Planner over an external API
#   + frozen pi05_robomemarena VLA
#   - every channel by which past observations can reach the Planner
#
# WHY THE MEMORY CHANNELS ARE THE THING BEING SWITCHED
#   The claim under test is "memory improves control". For that claim to be measurable, the
#   baseline must be identical to the treatment except for the memory channels. This file
#   therefore sources `official_protocol.sh` verbatim and then makes exactly two deviations,
#   both marked DEVIATION below.
#
# USAGE
#   ROOT=/project/peilab/why/RoboMemArena source arms/nomem.sh
# =========================================================================================

: "${ROOT:?arms/nomem.sh requires ROOT to be exported}"

# --- 0. Start from the official protocol, unmodified -------------------------------------
# shellcheck source=official_protocol.sh
source "$(dirname "${BASH_SOURCE[0]}")/official_protocol.sh"

# --- 1. Purge any residual memory-layer variables ----------------------------------------
# This is not hygiene. Several arms in `scripts/run_harness_variant.sh` export PMH_* / PACT_*
# / SEAM* variables, and a bash `export` does not disappear because this arm does not mention
# it. A single leftover PMH_ENABLE=1 makes `create_pmh_store_if_needed()` build a store, and
# this arm silently becomes a memory arm -- a fact that is INVISIBLE in the score and only
# detectable in a process-level census. Purging here plus asserting in `validate_arm.py`
# closes that hole.
#
# `MEM_` is purged as well. `MEM_STAGE_ANCHOR` and the other `memory_system` knobs select WHICH
# memory bank the Planner is shown, so a stale `MEM_STAGE_ANCHOR=1` inherited from another
# arm's shell would turn on a memory channel in the baseline.
#
# `MEMEXP_` IS purged, and that entry is load-bearing rather than defensive. Note that `MEM_`
# alone does NOT cover it: `MEMEXP_PULL_ENABLE` has no underscore after `MEM`, so `^MEM_` does
# not match it. Measured consequence, reproduced before this line existed: within one job that
# runs the arms in sequence, `pullmem` followed by `nomem` left `MEMEXP_PULL_ENABLE=1` in the
# baseline, `pysite/` still on PYTHONPATH, and the baseline Planner HOOKED -- i.e. the
# no-memory baseline quietly became a memory arm. The score would have looked like a memory
# effect while the control had the treatment installed.
#
# `HARNESS_CROSS_TASK_LTM` IS purged for the same reason, and it is a SECOND instance of the
# same defect rather than a widening of the first. The `rsimem` / `shamltm` arms carry their
# treatment on `HARNESS_CROSS_TASK_LTM*` keys -- they cannot use `MEMEXP_`, because CHECK 1's
# structural comparison drops that prefix by convention and an arm whose only difference is
# dropped reports an EMPTY diff, which CHECK 1 rejects. So the treatment lives on a `HARNESS_`
# key, which no `MEM_`/`MEMEXP_` prefix reaches, and the failure mode is exactly the pullmem one:
# arms run in sequence in a single job, so `nomem` after `rsimem` would inherit
# `HARNESS_CROSS_TASK_LTM=1` and inject the experience block into the CONTROL. Caught by CHECK 9
# (`rsimem->nomem: differs in ['HARNESS_CROSS_TASK_LTM', ...]`, `<absent>` vs `1`) before any GPU
# was spent. Unanchored on purpose: the bare prefix covers the bank path and the step threshold
# too, including keys added later.
#
# `HARNESS_STAGE_LEDGER_GUARD` is purged for the same serial-sourcing reason. The `rsiguard`
# arm (local path) puts its treatment on that prefix; without this line a subsequent nomem
# would inherit the action-level override channel.
for _pv in $(compgen -v 2>/dev/null | grep -E '^(PMH_|PACT_|KAIROS_|PROACTIVE_|ACE_|PCAM_|HPM_|SCEC_|CGMH_|MUSCLE_|SEAM|MEM_|MEMEXP_|HARNESS_CROSS_TASK_LTM|HARNESS_STAGE_LEDGER_GUARD)' || true); do
  unset "${_pv}"
done
unset _pv

# --- 1b. Strip this experiment's sitecustomize directory from PYTHONPATH -------------------
# Same defect, second half. `pullmem.sh` prepends `experiments/mem_efficacy/pysite` to
# PYTHONPATH so that `sitecustomize.py` can install the binding. That survives every
# variable purge, because PYTHONPATH is not a memory variable -- it is plumbing. So the
# baseline would still import the sitecustomize hook and re-activate the binding from
# `MEMEXP_PULL_ENABLE`... except that the purge above removed the flag, which is why the two
# halves have to be fixed together: either one alone leaves a half-configured hook.
#
# Entries are matched on the directory rather than compared as a whole string, so a PYTHONPATH
# that also carries unrelated paths keeps them. Written without touching IFS: the obvious
# `IFS=':' read -r -a` idiom has to save and restore IFS, and restoring it to a value captured
# while unset yields an EMPTY IFS, which silently disables word splitting for everything that
# runs afterwards.
if [[ -n "${PYTHONPATH:-}" ]]; then
  _py_keep=""
  _py_rest="${PYTHONPATH}"
  while [[ -n "${_py_rest}" ]]; do
    case "${_py_rest}" in
      *:*) _py_p="${_py_rest%%:*}"; _py_rest="${_py_rest#*:}" ;;
      *)   _py_p="${_py_rest}";     _py_rest="" ;;
    esac
    [[ -z "${_py_p}" ]] && continue
    # Anything under mem_efficacy may carry the hook; drop it.
    case "${_py_p}" in
      *mem_efficacy*) continue ;;
    esac
    _py_keep="${_py_keep:+${_py_keep}:}${_py_p}"
  done
  if [[ -n "${_py_keep}" ]]; then
    export PYTHONPATH="${_py_keep}"
  else
    unset PYTHONPATH
  fi
  unset _py_keep _py_rest _py_p
fi

# --- 2. Planner backend: external API (the model is pinned by the sbatch to gemini) -------
export PLANNER_BACKEND=api
# `closeai_api.txt` holds a single key, so line 1 is the key.
export PLANNER_API_KEY_LINE=1

# --- 3. Every other memory / meta controller is off --------------------------------------
#   PROACTIVE_MODE=off  -> `create_pmh_store_if_needed()` returns None, so `pmh_store is None`
#                          and the PMH read path (decide step, tool rounds, visual bank,
#                          address-demand queue) does not EXIST rather than being disabled.
#   The remaining flags switch off the sibling memory modules and the meta controller.
#   MUSCLE_ENABLE is deliberately left unset: `MuscleMemoryConfig.from_env()` defaults to
#   False, so `MuscleMemoryRing.create()` returns None and System-1 never runs.
export META_ENABLE=0
export BCM_ENABLE=0
export SCEC_ENABLE=0
export ACE_ENABLE=0
export PCAM_ENABLE=0
export HPM_ENABLE=0
export CGMH_GATE=0
export PROACTIVE_MODE=off

# --- 4. Harness ON (this is the "HarnessVLA" half; identical in both arms) ----------------
export HARNESS_ENABLE=1
export HARNESS_VLA_HINTS=0
export HARNESS_SUBTASK_OVERRIDE=1
export HARNESS_STAGE_CHECKPOINT=1
export HARNESS_RELEASE_ON_RETRY=1
export HARNESS_STALL_STEPS=80
export HARNESS_SMART_RETRY=1
export HARNESS_RETRY_SKIP_SCORE=95
export HARNESS_FORCE_VLM_REPLAN=1
export HARNESS_API_PLANNER=0
export HARNESS_PERSIST_MEMORY=1
export HARNESS_GLOBAL_RULES=${ROOT}/evaluation_benchmark/harness/global_rules.json
export HARNESS_MAX_RETRIES=2
export HARNESS_RETRY_REQUIRE_PROGRESS=1
export HARNESS_SKIP_RETRY_TASKS='[18,22]'

# --- 5. Memory-system bank selection: OFF, matching the official protocol ----------------
# 0 is the official value: the official sbatch does not set MEM_STAGE_ANCHOR, and
# `memory_system/config.py` defaults it to False. Stated explicitly rather than left to the
# default so that `validate_arm.py` can assert it is off in the baseline and that every
# treatment arm flips it deliberately rather than inheriting it.
export MEM_STAGE_ANCHOR=0

# =========================================================================================
# DEVIATION 1 of 2 -- channel A (textual evidence) OFF
#
#   HARNESS_VLM_CONTEXT=0
#     -> HarnessConfig.inject_vlm_context = False
#     -> HarnessController._refresh_vlm_context() returns early, `vlm_context` stays ""
#     -> compose_planner_context(harness=...) returns ""
#     -> the "Harness memory context" block never enters the Planner prompt
#
#   Content suppressed by this flag:
#     - episode_evidence lexical-search hits (the only long-horizon text recall path)
#     - best prior attempt's completed stages / stage score
#     - global rules loaded from global_rules.json
#     - "Current incomplete stage" / "Progress stalled for N steps"
#     - "Suggested primitive for recovery"
#
#   NOT suppressed by this flag (verified by reading the call sites):
#     - `subtask_override`, which flows through `consume_subtask_override()` into the
#       evaluator loop and rewrites the VLA prompt. That is a CONTROL action, not evidence
#       handed to the Planner, so it is present in both arms. See README "Known confound".
# =========================================================================================
export HARNESS_VLM_CONTEXT=0

# =========================================================================================
# DEVIATION 2 of 2 -- channel B (historical keyframe images) OFF
#
#   VLM_USE_KEYFRAME_MEMORY=0
#     -> ApiMemoryPlanner.use_keyframe_memory = False
#     -> `J_hist` stops accumulating, `K_indices_abs` stays [], `memory_main_frames` stays []
#     -> the "Historical keyframes from moments before the current step" block never enters
#        the Planner prompt
#
#   N_RECENT / K_MAX / D_MERGE keep their OFFICIAL values from official_protocol.sh, so the
#   two arms still see an identical RECENT window (that is the current observation, not
#   memory) and the only difference is whether past frames may be injected.
# =========================================================================================
export VLM_USE_KEYFRAME_MEMORY=0

# --- 6. Mechanism redaction: OFF, matching the official protocol -------------------------
# RMA's official runner never sets PMH_REDACT_STAGE, and the memory question in this
# benchmark is already real by task design: `task_block` / `brief_description` for the
# Occlusion suite (e.g. task 4) describe the *procedure* ("remember which drawer already
# contains an object") without naming the drawer. So the Planner cannot obtain the answer
# from the prompt text, and there is nothing for redaction to withhold. Leaving it off keeps
# both arms on the official configuration. Set MEMEXP_REDACT=1 to turn it on as an ablation.
export PMH_REDACT_STAGE="${MEMEXP_REDACT:-0}"

# --- 7. Arm self-identification (read back by the census, never by the runtime) ----------
export MEMEXP_ARM_NAME="nomem"
export MEMEXP_ARM_CLASS="baseline_no_memory"
