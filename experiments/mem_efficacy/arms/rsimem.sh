#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / arm: RSIMEM  (treatment -- cross-task experience memory)
#
# DEFINITION
#   The `nomem` baseline, unmodified, plus exactly one mechanism: a bank of experience entries
#   mined from the measured failure traces of this benchmark, keyed by task FAMILY and the stage
#   the planner is stuck on, rendered into the Planner's prompt while a container-opening stage
#   has been stalling. Nothing else changes -- see DEVIATION 3 below.
#
# WHY THIS IS NOT `pushmem` WITH DIFFERENT CONTENT
#   `pushmem` is the visual-keyframe channel (channel B). This is a textual channel, and it is
#   deliberately NOT routed through channel A (`HARNESS_VLM_CONTEXT`). Two measurements force
#   that separation:
#
#     * the 5-seed paired study: `memory_kf` (visual only) +8.2 pp on 5-of-5 seeds, and
#       `memory_ctx` (visual + this text channel) +4.0 pp on 3-of-5. Adding the text HALVED the
#       gain and dropped a unanimous result to a majority one.
#     * job 593253, on THIS benchmark arm: channel A on top of `pushmem` scored 51.0 against
#       `nomem` 67.2, i.e. -13.5 pp, 5 negative cells, 0 positive. The mechanism is legible in
#       the traces: the block named the current incomplete stage, and the Planner re-emitted that
#       stage's primitive while its stall counter climbed 1 -> 9.
#
#   So channel A stays OFF (`HARNESS_VLM_CONTEXT=0`, inherited from `nomem` and asserted equal
#   across every arm by `validate_arm.py` CHECK 5), and the treatment gets its own switch.
#
# WHAT THE ENTRY SAYS, AND WHY IT SAYS THAT
#   Measured on the 50-episode `nomem` baseline (tasks 12/13/17/20/23, seeds 100-109): the
#   episodes that reached 100.0 spent 55% of their planning steps on an opening primitive, left
#   that sub-goal within two attempts, and then named a TASK OBJECT (`reach_to_cookies`,
#   `pick_up_popcorn`, `pick_up_butter`, `place_cream_into_microwave`) -- 4.6 distinct non-opening
#   primitives on average. The episodes that scored 0.0 spent 96% of their steps there, stayed on
#   the container for 18-19 consecutive planning steps, and named 1.2. The entry asks for exactly
#   the measured difference: change WHAT the command names.
#
#   It does NOT name the current stage, which is the string the -13.5 pp result traces to. It is
#   retrieved BY stage, but the rendered text withholds it.
#
# THE CONFOUND THIS ARM CANNOT REMOVE, STATED SO IT IS NOT DISCOVERED LATER
#   The measured contrast above is OBSERVATIONAL and its causal direction is not established:
#   successful episodes are SHORT (task 23: 7-13 planning steps at 100.0, against 19-28 at 0.0),
#   so they have fewer opportunities to repeat at all. And an independent container-state probe
#   (`volo_container_state_probe.py`) judged the container CLOSED or PARTIALLY OPEN at the stuck
#   decision points with `moved_between_frames: false` in every case -- the container is not
#   moving, so the failure is partly MOTOR-side and the repeated command is not itself wrong.
#   The arm comparison is RANDOMISED, which is what an observational correlation cannot be: arm
#   assignment is independent of episode difficulty, so a paired delta cannot be explained by
#   "these episodes happened to be easier". `shamltm` covers the other half -- whether any
#   targeted block at that moment would do as well as this one.
#
# THE EXPECTATION, WRITTEN DOWN BEFORE THE RUN
#   A null is the most likely outcome, because the probe says the container does not move and no
#   planner-side text can move it. Stating that here is the point: a null then informs instead of
#   disappointing, and a POSITIVE result is surprising enough to require the sham arm before it
#   is believed.
#
# USAGE
#   ROOT=/project/peilab/why/RoboMemArena source arms/rsimem.sh
# =========================================================================================

: "${ROOT:?arms/rsimem.sh requires ROOT to be exported}"

# --- 0. Start from the baseline, unmodified ----------------------------------------------
# Not a convenience. Sourcing `nomem` is what inherits the full `MEM_` / `PMH_` / `MEMEXP_`
# purge, the PYTHONPATH strip, the channel A/B deviations and the official-protocol alignment,
# so this arm cannot drift from the control by editing two files in parallel. The ONLY thing
# this file adds is the block below, which is what makes DEVIATION 3 a one-item set that
# CHECK 1 can verify by set equality rather than by inspection.
# shellcheck source=nomem.sh
source "$(dirname "${BASH_SOURCE[0]}")/nomem.sh"

# =========================================================================================
# DEVIATION 3 of 3 -- cross-task experience memory (RSI), the only treatment here
#
#   HARNESS_CROSS_TASK_LTM=1
#     -> `harness/cross_task_ltm.py:enabled()` is true
#     -> `HarnessController.on_stage_progress` renders the entry on EVERY step, before the
#        early-return branches (so it also runs on the stall path), and stores it in
#        `_ltm_context`
#     -> `compose_planner_context` picks it up through `get_cross_task_context()` -- a SEPARATE
#        source discovered by `hasattr`, not through `get_vlm_context`
#     -> the block enters the Planner prompt as its own paragraph
#
#   Two properties are load-bearing and both are asserted, not assumed:
#
#     (1) IT IS INERT WHEN OFF. Every gate defaults to "no block": the env switch, the family
#         lookup, the `open`-stage test, and the step threshold all have to agree before any
#         text is produced. `validate_arm.py` CHECK 5 asserts the baseline's composed context is
#         EMPTY (`compose_len == 0`), so a leak here fails PREFLIGHT rather than quietly moving
#         the control that every memory delta is measured against. Measured off-path: len=0.
#
#     (2) IT ACTUALLY FIRES. This is a TREATMENT, not instrumentation, so an arm that runs and
#         injects nothing is void rather than merely unlucky -- the same defect class as job
#         586700 (`J_hist` all-empty and the arm still scored). `cross_task_ltm.ltm_stats()`
#         counts calls/fired/failed/load_error, and the census reads them.
#
#   THE THRESHOLD. `HARNESS_CROSS_TASK_LTM_MIN_STEPS=240` is the number of ENV steps spent on the
#   current stage before the entry appears. The Planner is called every ~80 env steps on these
#   tasks (measured: task 20, 28 calls over ~2470 steps), so 240 is the third call -- the point
#   at which the failing episodes have already used two attempts. It is a knob rather than a
#   constant so the trigger can be moved without editing code, and it is declared in
#   MEMEXP_EXPECTED_DIFF for that reason.
#
#   THE BANK IS A FILE, and an unreadable or schema-mismatched bank is an ERROR, not a silent
#   no-op: `_load` logs at ERROR and counts `load_error`.
# =========================================================================================
export HARNESS_CROSS_TASK_LTM=1
export HARNESS_CROSS_TASK_LTM_BANK="${ROOT}/experiments/mem_efficacy/rsi_ltm/rsimem.json"
export HARNESS_CROSS_TASK_LTM_MIN_STEPS=240

# --- 1. Arm self-identification (read back by the census, never by the runtime) ----------
export MEMEXP_ARM_NAME="rsimem"
export MEMEXP_ARM_CLASS="treatment_cross_task_experience"

# --- 2. Declared deviation set (CHECK 1 proves the diff EQUALS this, not merely contains it) --
export MEMEXP_EXPECTED_DIFF="HARNESS_CROSS_TASK_LTM HARNESS_CROSS_TASK_LTM_BANK HARNESS_CROSS_TASK_LTM_MIN_STEPS"
