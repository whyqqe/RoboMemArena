#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / arm: SHAMLTM  (control -- form control for `rsimem`)
#
# DEFINITION
#   `rsimem`, with the entry's CONTENT replaced by an entry minted from a disjoint family. The
#   two arms are identical in schema, trigger, header, entry count, and (to 2.8%) character
#   length. The only thing that differs is whether the text can apply to the task being run.
#
# WHY THIS ARM IS MANDATORY AND NOT A NICETY
#   Any text in a prompt changes behaviour. Without this arm, "the experience memory helped" and
#   "a block of targeted text at the stuck moment helped" are the SAME NUMBER, and no amount of
#   statistics on the treatment alone can separate them -- the treatment arm is randomised
#   against a text-FREE control, so it measures content and form together.
#
#   This is not a hypothetical failure mode in this repository. The channel-A result (-13.5 pp,
#   job 593253) was a TEXT effect, and it was only attributable to its content because the visual
#   channel had been measured separately (`memory_kf` +8.2 pp alone, +4.0 pp with the text added,
#   same 5 seeds). The sham is how that separation is obtained in one job instead of two.
#
# HOW THE CONTENT WAS CHOSEN, AND WHY THIS WAY
#   From the counting / pour family's own vocabulary (`02_Pour_One`, `03_Pour_Two`) and its
#   objects: vessels, pouring, `pour_into_<vessel>`. It is structurally identical to the
#   treatment -- same sentence shape, same "two attempts then name something else" claim, same
#   falsifiability -- and operationally irrelevant on an occlusion or occluded-location task,
#   because those tasks contain no pour stage and no vessel.
#
#   The obvious alternative, permuting the treatment's own object names into a different family's
#   containers, was rejected: on a drawer task a sentence about a microwave is still ACCIDENTALLY
#   good advice (open the thing, then place), so it would be a second treatment rather than a
#   control. The pour entry cannot be accidentally right.
#
# WHAT IT CAN AND CANNOT ESTABLISH
#   It CAN separate content from form. It CANNOT by itself make a null result meaningful for the
#   treatment -- a null in both arms is consistent with both "the entry was inert" and "the
#   mechanism is real but underpowered at 50 episodes". Power is answered separately, from the
#   paired variance this experiment measures.
#
# USAGE
#   ROOT=/project/peilab/why/RoboMemArena source arms/shamltm.sh
# =========================================================================================

: "${ROOT:?arms/shamltm.sh requires ROOT to be exported}"

# --- 0. Start from the TREATMENT so every knob except the content is shared by construction --
# Sourcing `rsimem` rather than `nomem` is what makes this a form control: trigger, threshold,
# header, and the whole inherited baseline arrive through one path, so the arms cannot drift
# apart in anything except the bank file. It also keeps MEMEXP_EXPECTED_DIFF identical in shape,
# which CHECK 1 then verifies as a set equality.
# shellcheck source=rsimem.sh
source "$(dirname "${BASH_SOURCE[0]}")/rsimem.sh"

# =========================================================================================
# The ONLY change: point the treatment at a bank whose content cannot transfer.
#
# Note that this re-exports a key that `rsimem` already exported, which is why CHECK 1 still
# reports the same three keys for both arms -- the declared deviation set is unchanged, only the
# VALUE of one of them moves. That is deliberate: a sham that also changed the trigger would
# confound content with timing, and the timing is the mechanism.
# =========================================================================================
export HARNESS_CROSS_TASK_LTM_BANK="${ROOT}/experiments/mem_efficacy/rsi_ltm/shamltm.json"

# --- 1. Arm self-identification (read back by the census, never by the runtime) ----------
export MEMEXP_ARM_NAME="shamltm"
export MEMEXP_ARM_CLASS="control_sham_content"
