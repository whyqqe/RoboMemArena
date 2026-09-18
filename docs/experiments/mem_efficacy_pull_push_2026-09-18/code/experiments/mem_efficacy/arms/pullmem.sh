#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / arm: PULLMEM  (treatment -- small push + agent-chosen dereferencing)
#
# DEFINITION
#   arms/nomem.sh, plus a small always-on memory push and a set of tools the Planner may call
#   WHILE planning.
#
# WHY THIS ARM EXISTS IN ADDITION TO pushmem
#   `pushmem` has to decide, in advance, what will matter. It can only do that through a proxy
#   (stage boundaries, subtask changes), so it must fail whenever relevance is determined LATER.
#   The Occlusion tasks in this benchmark are exactly that case: three drawers are each observed
#   once, and WHICH one is the target depends on which one turned out to be non-empty -- a fact
#   that does not exist until all three have been looked at. No a-priori salience rule can rank
#   the second observation above the first, because at that moment neither is more important.
#
#   This arm separates the two roles that `pushmem` conflates:
#     * GETTING evidence happens on demand, chosen by the Planner, who is the only party that
#       knows which address matters right now;
#     * KEEPING it available is still the harness's job, so the push remains.
#   That is the hybrid the design calls for, and it is also what keeps the read path from
#   depending on a model's willingness to ask -- the failure that emptied HM's image channel and
#   left PMH with a read path that had 0 searches in it.
#
# WHAT IS PUSHED (small, capped, and never an answer)
#   * the last few actions the Planner itself emitted, with frame pointers
#   * the addresses that have been OBSERVED but not yet RESOLVED
#   Pushing the second list is what makes the pull usable. It converts "notice that you are
#   missing something" -- a judgement PMH delegated to the model and got wrong 85% of the time --
#   into "read a list of open addresses", which is a structural fact.
#
# WHAT IS NOT PUSHED
#   The PrediMem keyframe bank itself. `VLM_USE_KEYFRAME_MEMORY=1` so the bank is BUILT
#   (nominations + stage anchors + spread), but the binding strips those frames from the prompt.
#   The only images the Planner sees besides the live window are the ones IT asked for. That is
#   what makes `n_frames_served` interpretable as gain rather than as volume.
#
# USAGE
#   ROOT=/project/peilab/why/RoboMemArena source arms/pullmem.sh
# =========================================================================================

: "${ROOT:?arms/pullmem.sh requires ROOT to be exported}"

# shellcheck source=nomem.sh
source "$(dirname "${BASH_SOURCE[0]}")/nomem.sh"

# -----------------------------------------------------------------------------------------
# The tool module must be importable, and the binding must install itself before the evaluator
# imports the planner. `_memexp_pysite.sh` puts this directory on PYTHONPATH and turns on the
# memory-content correction that `pushmem` also uses -- shared, so the two memory arms differ only
# in their declared channels rather than in which harness code they run. It must come AFTER
# `nomem.sh`, which strips this directory from PYTHONPATH on purpose.
# -----------------------------------------------------------------------------------------
# shellcheck source=_memexp_pysite.sh
source "$(dirname "${BASH_SOURCE[0]}")/_memexp_pysite.sh"

export MEMEXP_PULL_ENABLE=1

# Where the binding writes its own counters. The census reads this; asserting the tool loop from
# inside the process that ran it is the only way to tell "the Planner chose not to search" apart
# from "the search path was never reachable".
export MEMEXP_PULL_REPORT="${MEMEXP_PULL_REPORT:-${OUT_ROOT:-/tmp}/memexp_pull_report.json}"

# --- Channel A: the harness's own textual evidence, OFF ------------------------------------
# Was ON, on the argument that "the comparison of interest is who SELECTS, so turning it off
# would change what is available as well as who chooses it". That argument assumed channel A was
# neutral. It is not: it is measured as a net LOSS (see pushmem.sh, and
# docs/experiments/dual_track_report.md: memory_kf +8.2 pp / 5-of-5 seeds vs memory_ctx +4.0 pp /
# 3-of-5). Leaving it on in one arm and off in the other would have made push-vs-pull a
# comparison of the TEXT as well as of the selection rule.
#
# Off in ALL THREE arms makes channel A a controlled constant. What is left differing between
# `pushmem` and `pullmem` is exactly the selection rule for the images: the harness choosing them
# a priori (stage boundaries and subtask changes) versus the Planner choosing them on demand.
#
# This does NOT empty this arm's memory. The small push (recent actions, observed-but-unresolved
# addresses) is rendered by this experiment's own binding -- `memexp_bind._wrap_build_messages`
# calls `substrate.render_push()` and appends it to the messages -- and the substrate is fed from
# the Planner's OWN previous subtask (`substrate.note_action(prev, step_of_prev)`), not from
# channel A. So turning channel A off removes the harness's diluting text and leaves the pull
# mechanism's inputs intact.
export HARNESS_VLM_CONTEXT=0

# --- Channel B: PrediMem bank BUILT, but NOT pushed ----------------------------------------
# PrediMem's bank (nominations + stage/subtask anchors + temporal spread) is the same store
# `pushmem` builds. The difference that makes this arm a PULL is that the bank is NOT injected
# into every prompt: `memexp_bind._wrap_build_messages` strips `memory_*_frames` before the
# official builder emits them, and tools serve from `K_indices_abs` on demand. So:
#   * GETTING evidence is agent-chosen (PMH pull over a PrediMem store);
#   * KEEPING it available is still the harness's job (the small address push stays).
# Leaving `VLM_USE_KEYFRAME_MEMORY=0` would build no bank at all, and every tool serve would fall
# back to the LOOKAHEAD neighbourhood of the mint step -- which is exactly "no PrediMem".
export VLM_USE_KEYFRAME_MEMORY=1
export MEM_STAGE_ANCHOR=1
export K_MAX=8
export D_MERGE=4
export N_RECENT=7
export MEM_KF_SPREAD=8
export MEM_KF_NOMINATION_PROMPT=1

# --- Dense frame record ------------------------------------------------------------------
# Same reason as `pushmem`, and it matters MORE here. This arm's contract is that historical
# frames arrive only on demand, so the retrieval bank IS the memory -- there is no second channel
# to cover for it. But the bank's candidates come from `planner.frame_store_main`, which used to be
# filled only from the context windows of planner calls that reached the model: with the official
# `VLM_INTERVAL=5` / `VLM_QUEUE_SIZE=1`, submissions evict each other and an episode of ~2470 env
# steps yielded only 3-9 calls, i.e. ~35 stored frames (1.4% of the episode, weighted to its start).
#
# A pull arm whose store holds the opening scene can only ever SERVE the opening scene, and would
# have reported "the agent asked and got nothing useful" as a property of pull-style retrieval
# rather than of an empty record. This stores one frame every 5 env steps from the env loop, so the
# thing being pulled over is an actual record of the episode.
export MEM_KF_STORE_INTERVAL=5

# --- The tool loop ------------------------------------------------------------------------
# MEMEXP_PULL_TOOLS=0 keeps the small push but disables calling tools. That is the
# single-variable control for "does agent-chosen dereferencing help, given identical pushed
# content", obtainable as an override rather than a second arm file so the two conditions cannot
# drift apart:
#     MEMEXP_PULL_TOOLS=0 ARM_OVERRIDE=pullmem sbatch ...
export MEMEXP_PULL_TOOLS="${MEMEXP_PULL_TOOLS:-1}"

# --- WHICH tool registry -------------------------------------------------------------------
# `memexp_tools_search` adds `search_memory(query)`: free-text search over every record the
# episode has established, scored on the address AND the value, returning the matching records
# together with their frames.
#
# The address-only registry (`memexp_tools`) remains available and unchanged for an ablation. It is
# not the default for a structural reason: `query_world` takes an ADDRESS -- an exact string the
# model must copy out of the pushed block. That inverts retrieval. A model that can name
# `contents(middle drawer)` can usually already see what it refers to, and a model that cannot name
# it has no way to reach the record at all. The archived run's evidence is exactly that shape: 5
# calls, every one of them on an address that had just been advertised.
#
# The contract repair in that same module is NOT optional, and CHECK 15 asserts it: the previous
# spec taught the model to finish with `{"primitive": ...}`, a key the planner's parser reads as the
# EMPTY STRING -- `json.loads` succeeds, so nothing raises, the prose fallback never fires, and the
# step yields no action at all. With the contract repaired the prompt no longer asks the model to
# obey a rule and break it in the same breath, which is what makes the call rate a measurement of
# the model rather than of the prompt.
export MEMEXP_TOOLS_MODULE="${MEMEXP_TOOLS_MODULE:-memexp_tools_search}"
# A fuse, not a schedule. The loop ends when the model emits a primitive; this only bounds a
# model that loops. `cap_hits` in the report must be 0 in a healthy run -- a non-zero value means
# either the cap is too small or a tool result is not answering.
#
# 3 -> 2, AND THE SPEC NOW STATES THE NUMBER. Job 595132 ran this at 3 while the tool spec promised
# "as many times as you need", and measured `cap_hits=2`: the Planner was told an unbounded budget
# and then refused. The cap is a fuse against a model that loops (the observed per-step rate was
# 1.40 calls, so 2 leaves one retry after a genuine miss), but a fuse the prompt does not mention
# makes `cap_hits` a measurement of the prompt. `memexp_tools_search.spec_text()` now reads this
# same variable, so the two can no longer disagree.
export MEMEXP_PULL_MAX_ROUNDS="${MEMEXP_PULL_MAX_ROUNDS:-2}"
# Frames returned per dereference, and how far ahead of an action to look. Small: the container
# is still in motion at the action step, so the frame after it is the useful one.
export MEMEXP_LOOKAHEAD="${MEMEXP_LOOKAHEAD:-3}"

# --- Arm self-identification --------------------------------------------------------------
export MEMEXP_ARM_NAME="pullmem"
export MEMEXP_ARM_CLASS="treatment_push_small_plus_agent_pull"

# The exact set of keys that may differ from the baseline (order-independent).
# `validate_arm.py` asserts equality against this list.
#
# Channel B knobs match `pushmem` on PURPOSE: both arms build the same PrediMem-shaped bank, so
# the only remaining difference is who SELECTS the frames (harness push vs agent pull). The pull
# binding itself is installed by `PYTHONPATH`. `MEMEXP_*` keys are dropped by CHECK 1's
# structural comparison by convention; CHECK 15 accounts for the tool registry positively.
export MEMEXP_EXPECTED_DIFF="VLM_USE_KEYFRAME_MEMORY MEM_STAGE_ANCHOR K_MAX D_MERGE N_RECENT MEM_KF_SPREAD MEM_KF_NOMINATION_PROMPT MEM_KF_STORE_INTERVAL PYTHONPATH"
