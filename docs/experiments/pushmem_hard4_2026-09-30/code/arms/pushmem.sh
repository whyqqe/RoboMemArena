#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / arm: PUSHMEM  (treatment -- push-style memory, HM-like)
#
# DEFINITION
#   arms/nomem.sh, plus the memory channels that RMA pushes into the Planner prompt WITHOUT
#   the Planner asking for them. No tool call, no retrieval decision, no query string: the
#   Python layer decides what enters the prompt, every plan step.
#
# WHAT "PUSH" MEANS HERE, AND WHY IT IS NOT A NEW MECHANISM
#   This arm is assembled entirely from RMA's own code paths. `api_vlm_planner._build_messages`
#   unconditionally appends two memory blocks when the fields below are populated:
#
#     "Harness memory context (read-time evidence; use with historical keyframes):"
#         <- api_vlm_planner.py:2754, fed by `harness_extra_context`
#     "Historical keyframes from moments before the current step in the same execution (N):"
#         <- api_vlm_planner.py:2779, fed by `memory_main_frames`
#
#   Neither block is gated on anything the Planner emitted. That is the defining property of
#   this arm, and it is also what the official protocol means by memory: its README describes
#   the official mechanism as "unlimited historical keyframes (K_MAX=0)" and "task-conditioned
#   VLM prompting with historical keyframes and recent visual context".
#
# WHY MEM_STAGE_ANCHOR=1 IS REQUIRED, NOT OPTIONAL
#   The official protocol leaves MEM_STAGE_ANCHOR unset, which defaults to 0. In that mode the
#   image block is a pure function of `J_hist`:
#
#       K_indices_abs = build_visual_memory(J_hist, ...)      api_vlm_planner.py:4770
#
#   and `J_hist` holds the Planner's OWN `keyframe_positions` nominations:
#
#       self.J_hist.append(j_abs)                             api_vlm_planner.py:4714
#
#   That works with the official local PrediMem VLM, which is trained to emit the field. A
#   general API Planner is not, so `J_hist` stays a list of empty lists and the image block is
#   empty at EVERY step -- the channel is on, the prompt block is never emitted. The code
#   already documents this state (api_vlm_planner.py:1489-1491, and `kf_n=0` from job 586700).
#
#   MEM_STAGE_ANCHOR=1 switches the bank builder to RMA's `merge_keyframe_bank`, which is
#   still official code and which sources candidates from two DETERMINISTIC signals instead of
#   the Planner's willingness to nominate:
#
#       pinned_steps   -- stage boundaries, appended by the eval loop when a stage advances
#                         (eval_fullvlm26_async_vlm_vla.py:1745-1746 -> pin_keyframe)
#       salient_steps  -- subtask changes, appended on every planned-subtask change
#                         (eval_fullvlm26_async_vlm_vla.py:1226-1227 -> mark_salient_keyframe)
#
#   Those two signals the Planner cannot suppress: the first is the harness's own stage
#   machine, the second is the Planner's own subtask output. So the image block becomes
#   genuinely push-based. With `bank_max=0` (= official K_MAX=0) no cap is applied, and since
#   the Planner's nominations are still unioned in when they exist, this is a strict superset
#   of the official behaviour rather than a replacement.
#
# USAGE
#   ROOT=/project/peilab/why/RoboMemArena source arms/pushmem.sh
# =========================================================================================

: "${ROOT:?arms/pushmem.sh requires ROOT to be exported}"

# shellcheck source=nomem.sh
source "$(dirname "${BASH_SOURCE[0]}")/nomem.sh"

# -----------------------------------------------------------------------------------------
# Memory-content correction (shared with `pullmem`), and the `memfix` half of the diff.
#
#   The `path` is sourced AFTER `nomem.sh` because `nomem.sh` strips this directory from
#   PYTHONPATH on purpose. It sets MEMEXP_MEMFIX_ENABLE=1, which makes `pysite/sitecustomize.py`
#   install `memexp_memfix` in the evaluator process.
#
#   `PYTHONPATH` therefore joins this arm's declared diff. It is plumbing, not a memory channel,
#   which is why it is declared separately from the three switches below.
# -----------------------------------------------------------------------------------------
# shellcheck source=_memexp_pysite.sh
source "$(dirname "${BASH_SOURCE[0]}")/_memexp_pysite.sh"

# The PULL binding must stay OFF here: this arm's question is what the harness PUSHES, so the
# Planner must not also be given tools to pull. Set explicitly rather than relying on the absence
# of the variable, so that reading this file and reading the runtime instrument agree.
export MEMEXP_PULL_ENABLE=0

# -----------------------------------------------------------------------------------------
# Channel A -- textual evidence: OFF. This is now a DELIBERATE arm constant, not a default.
#
#   This arm used to push the harness's textual evidence every step (`HARNESS_VLM_CONTEXT=1`, the
#   HM-like content). It is off because our own 5-seed paired study attributes the gain to the
#   VISUAL channel and measures the TEXT channel as a net loss:
#
#     docs/experiments/dual_track_report.md (5 seeds, paired, ΔCSR vs baseline)
#       memory_kf   = denser keyframe bank, harness/text OFF   +8.2 pp   5/5 seeds positive
#       memory_ctx  = memory_kf PLUS VLM text context           +4.0 pp   3/5
#       memory_plus = memory_ctx PLUS stage anchor              +5.9 pp   4/5
#
#   Adding the text channel to the visual one HALVES the gain and drops it from 5/5 to 3/5 seeds.
#   The report's conclusion names the lever explicitly: "the primary lever is denser visual
#   episodic memory, NOT episodic text or the current stage-anchor implementation".
#
#   Measured locally as well, in job 593253 on this benchmark arm: with channel A on, `pushmem`
#   scored 51.0 against `nomem`'s 67.2 (-13.5 pp, bootstrap 95% CI [-27.6, -3.1], 5 negative cells
#   and 0 positive). The mechanism is visible in the task-4 traces: on every plan step the block
#   names the current incomplete stage, and the Planner then re-emits that same primitive
#   ("open top drawer") while its stall counter climbs 1 -> 9, never advancing. On `nomem` the same
#   task wanders across "grasp butter" / "close top drawer" / "03_Open_Middle_Drawer" and clears
#   stage 1 sometimes. Pushing the current stage pins the Planner to it.
#
#   WHAT IS LOST BY TURNING IT OFF: `episode_evidence` lexical hits, the best prior attempt's
#   completed stages and score, "Current incomplete stage", the stall description, and the
#   suggested recovery primitive. That is the point -- those are the diluting inputs.
#
#   NOT lost (verified by reading the call sites, and it is why this arm is still a treatment):
#     - `subtask_override`, which flows through `consume_subtask_override()` into the evaluator
#       loop and rewrites the VLA prompt. That is a CONTROL action, not evidence handed to the
#       Planner, so it is identical in every arm. See README "Known confound".
#     - the keyframe bank, which is channel B below and is this arm's experimental variable.
#
#   Channel A is now a CONSTANT across all three arms (OFF in nomem, pushmem and pullmem). That is
#   what makes `pushmem` vs `pullmem` a comparison of WHO SELECTS the images rather than of which
#   text was available to them.
# -----------------------------------------------------------------------------------------
export HARNESS_VLM_CONTEXT=0

# -----------------------------------------------------------------------------------------
# Channel B -- historical keyframe images, pushed every step
#
#   VLM_USE_KEYFRAME_MEMORY=1   produce and inject the block at all
#   MEM_STAGE_ANCHOR=1          build the bank from deterministic stage/subtask anchors
#                               instead of from Planner nominations (see the header)
#   D_MERGE / K_MAX / N_RECENT  the bank's own hyperparameters, set to the ONE tuple this
#                               project has a positive prior for. See below.
#
# WHY THE THREE BANK HYPERPARAMETERS ARE PINNED HERE
#   This arm previously inherited `official_protocol.sh`'s values (K_MAX=0, D_MERGE=6,
#   N_RECENT=5) and measured a keyframe density of ~1.67 frames/call -- against a `bank_max`
#   of 8. The bank was therefore never cap-limited; it was CANDIDATE-limited, and the cause is
#   `cluster_distance`: `merge_keyframe_bank` clusters all nominations within `d` frames and
#   keeps ONE median per cluster, so d=6 collapses almost every anchor pair into a single frame.
#
#   The values below are exactly the ones `scripts/run_harness_variant.sh:memory_kf` used, which
#   is the only memory configuration in this repository with a measured positive delta
#   (docs/experiments/dual_track_report.md: +8.2 pp, 5/5 seeds positive, on 5 seeds paired).
#   Reproducing that tuple is the point: a different mix would be a new, unmeasured configuration
#   whose result could not be attributed to anything.
#
#   WHAT THIS DOES NOT FIX, stated so it is not mistaken for a solution. `merge_keyframe_bank`
#   draws candidates ONLY from stage boundaries (`pinned_steps`) and subtask changes
#   (`salient_steps`). That is a handful of anchors clustered around transitions, never a
#   temporal SPREAD across the episode -- so the block is structurally thin no matter what these
#   three numbers are. Widening it requires a stride over the frame store, which is the SDV path
#   (`_sdv_substrate_floor`, `PMH_SDV_STRIDE`) and not this arm's mechanism. Treat the delta this
#   arm produces as a LOWER bound on what denser visual memory is worth, not as the ceiling.
#
#   `N_RECENT` is declared below because it is also the exclusion cutoff for the bank
#   (`k <= t - N + 1`), so it is a memory parameter here and not merely prompt plumbing.
# -----------------------------------------------------------------------------------------
export VLM_USE_KEYFRAME_MEMORY=1
export MEM_STAGE_ANCHOR=1
export K_MAX=8
export D_MERGE=4
export N_RECENT=7

# The additive temporal spread. See `MemorySystemConfig.kf_spread` and
# `ApiMemoryPlanner._kf_spread_union` for the measurement that motivates it: the anchor-based
# builder alone injected 1.7 frames/call against a cap of 8, i.e. the bank was candidate-limited
# and no hyperparameter of that builder could widen it.
#
# Set to the cap, so the frames the spread contributes fill the bank rather than competing with
# the anchors for room. Bounded by `K_MAX`, so the per-call image count has a hard ceiling of 8
# regardless of episode length -- which is what keeps this compatible with the cost model in
# NEXT_AGENT_BRIEF.md C.9 (10 -> 13 images/call for the official keyframe memory).
export MEM_KF_SPREAD=8

# -----------------------------------------------------------------------------------------
# Channel B, the PrediMem half: teach the Planner WHAT to nominate.
#
# PrediMem sources its bank from the model's OWN `keyframe_positions` nominations, and the paper's
# ablation (bank removed: TSR 38.5% -> 17.7%) is why that channel is load-bearing. With the local
# PrediMem VLM the nominations come from TRAINING. This arm keeps that architecture but the Planner
# is a general API model, so the policy is supplied by INSTRUCTION instead of by weights.
#
# Without this, `J_hist` is a list of empty lists (measured: `J_mean = 0.00`, `kf_n = 0`, job
# 586700), `build_visual_memory` returns [] on every step, and the official bank builder is
# STRUCTURALLY DEAD in this configuration -- not sparse, dead. `MEM_STAGE_ANCHOR=1` was the
# previous workaround, and it is why the bank was floored at ~1.7 frames/call: its candidates are
# stage boundaries and subtask changes, i.e. TRANSITIONS, never a record of the episode.
#
# WHAT THIS RESTORES, AND WHAT IT CANNOT. `merge_keyframe_bank` (official code, unchanged) unions
# the nomination-derived bank with the anchors, so elements 1 (task-adaptive candidates) and 2
# (official consolidation) of PrediMem are now both live, and element 4 (frames consumed by the
# Planner) was never altered. Element 3 -- the predictive-coding head, which is what makes a
# TRAINED model's representation sensitive to state transitions -- has no counterpart here and
# cannot be obtained without training. A prompted model has to REASON about which frame recorded a
# transition; a trained one is built to notice. So this is a substitute for that component, and the
# honest name for the result is "PrediMem's bank architecture driven by a prompted nomination
# policy", not "PrediMem".
#
# THE FALLBACK IS DELIBERATE. `MEM_KF_SPREAD` above stays on: if the model ignores the policy the
# bank still has breadth, so the arm is informative either way. `census_channels.py` reports
# `steps_with_planner_nomination` and FAILs the arm when this knob is ON and that count is zero,
# so "the instruction did nothing" can never be read as "nomination memory did not help".
export MEM_KF_NOMINATION_PROMPT=1

# --- Dense frame record ------------------------------------------------------------------
# THE ROOT CAUSE OF "THE BANK IS THE OPENING SCENE". The bank's candidate pool is
# `planner.frame_store_main`, and that store used to be filled ONLY from the context windows of
# planner calls that actually reached the model. With the official `VLM_INTERVAL=5` and
# `VLM_QUEUE_SIZE=1`, `submit_vlm_job` evicts the pending payload on every new submission, and the
# env loop is far faster than a reasoning planner -- measured on job 594338: an episode of ~2470
# env steps produced only 3-9 planner calls, so the store held ~35 frames, 1.4% of the episode,
# weighted to its start.
#
# That is why 54-62% of all bank slots were frames <= 8 and frame 0 was present on 100% of plan
# steps, on every task checked: `_kf_spread_union` strides evenly over the pool, and the pool WAS
# the opening. The bank could not span the episode because nothing had recorded the episode.
#
# This stores one frame every 5 env steps from the env loop itself, so the record is driven by the
# rollout and not by which async job survived. 5 matches `VLM_INTERVAL`, i.e. one frame per replan
# window, which is the finest granularity at which the bank's frames can ever be re-served.
# Baseline declares nothing here and defaults to 0, so the official path is unchanged.
export MEM_KF_STORE_INTERVAL=5


# --- Arm self-identification --------------------------------------------------------------
export MEMEXP_ARM_NAME="pushmem"
export MEMEXP_ARM_CLASS="treatment_push_memory"

# The exact set of keys that may differ from the baseline (order-independent).
# `validate_arm.py` asserts equality against this list.
#
# `HARNESS_VLM_CONTEXT` is no longer listed because it is no longer a difference: this arm now
# leaves it OFF, the same value the baseline holds, so channel A is a controlled constant. What
# remains is exactly this arm's mechanism -- the pushed keyframe bank (VLM_USE_KEYFRAME_MEMORY,
# MEM_STAGE_ANCHOR, MEM_KF_NOMINATION_PROMPT, MEM_KF_SPREAD and the three bank hyperparameters) --
# plus `PYTHONPATH`, which is the plumbing that installs the shared memory-content correction and
# is declared rather than silently exempted so CHECK 1 keeps its force.
export MEMEXP_EXPECTED_DIFF="VLM_USE_KEYFRAME_MEMORY MEM_STAGE_ANCHOR K_MAX D_MERGE N_RECENT MEM_KF_SPREAD MEM_KF_NOMINATION_PROMPT MEM_KF_STORE_INTERVAL PYTHONPATH"
