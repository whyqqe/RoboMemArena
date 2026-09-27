#!/usr/bin/env bash
# =========================================================================================
# mem_efficacy / RMA official protocol knobs
#
# PURPOSE
#   A single source of truth for the RMA *official* evaluation protocol, so that the two
#   arms of this experiment cannot disagree with each other or with upstream.
#
#   The values below were copied from the project's own official runner,
#   `slurm/benchmark/reproduce_all26_1seed.sbatch`, which is the artifact the README names
#   as "PrediMem on all 26 tasks (official protocol)".
#
# WHY THIS FILE EXISTS AT ALL
#   The runner `run_fullvlm26_async_vlm_vla_csr_tsr.sh` resolves each knob as
#   `${VAR:-official_default}`. That means an arm which exports the variable SILENTLY
#   OVERRIDES the official protocol. This has already happened in this repository: the
#   harness arm `api_qwen_harness_v21_redact` (the HM baseline every past memory delta was
#   measured against) exports N_RECENT=7, K_MAX=8, D_MERGE=4, whereas the official protocol
#   is N_RECENT=5, K_MAX=0, D_MERGE=6. So prior HM numbers are NOT official-protocol
#   numbers, and this experiment must not inherit that drift.
#
#   `validate_arm.py` parses the official sbatch and diffs it against this file, so any
#   upstream change to the official protocol makes the preflight FAIL instead of letting the
#   two silently diverge.
#
# USAGE
#   ROOT=/project/peilab/why/RoboMemArena source arms/official_protocol.sh
# =========================================================================================

: "${ROOT:?arms/official_protocol.sh requires ROOT to be exported}"

# --- scale -------------------------------------------------------------------------------
export TASKS_JSON='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]'
export NUM_TRIALS=1
export SEED=100

# --- episode budget ----------------------------------------------------------------------
export MAX_STEPS=2500
export POST_GOAL_STEPS=200
export POST_STAGE_STEPS=30
# Stricter-than-paper terminator: an extra pour fails the episode.
export FAIL_ON_EXTRA_POUR=1

# --- planner / VLM cadence ---------------------------------------------------------------
# ASYNC_VLM=1 with VLM_QUEUE_SIZE=1 is the official configuration. It is also the reason
# this benchmark carries large cross-run noise (queued planning jobs are dropped), so
# single-run comparisons between arms are only meaningful with this held fixed.
export ASYNC_VLM=1
export VLM_INTERVAL=5
export VLM_QUEUE_SIZE=1
export REPLAN_STEPS=10
export NUM_STEPS_WAIT=10
export RESIZE_SIZE=256
export VLM_INPUT_PROFILE=fullvlm_256
export VLM_USE_WRIST=1
export VLM_MATCH_TRAINING_JPEG_ROUNDTRIP=0

# --- memory-system hyperparameters (OFFICIAL values) -------------------------------------
#   N_RECENT=5   recent observation window length
#   K_MAX=0      NO cap on the keyframe bank. 0 does not mean "off": in
#                `api_vlm_planner.py` the cap is `self.k_max if self.k_max > 0 else
#                mem_cfg.bank_max`, and `memory_system/config.py` derives `bank_max` from
#                K_MAX, so 0 => bank_max=0 => the truncation branch is never taken.
#                The keyframe channel is instead switched by VLM_USE_KEYFRAME_MEMORY.
#   D_MERGE=6    cluster distance for `build_visual_memory`
export N_RECENT=5
export K_MAX=0
export D_MERGE=6
# Official value is 1. The no-memory arm of this experiment flips it to 0; that flip is the
# single documented deviation of channel B and nothing else in this file changes.
export VLM_USE_KEYFRAME_MEMORY=1

# --- frozen VLA --------------------------------------------------------------------------
# pi05_robomemarena, served from `checkpoints/PrediMem/vla_alltask` by
# `third_party/openpi_minimal/scripts/serve_policy.py`. Frozen for every arm.
export VLA_CONFIG=pi05_robomemarena
export PORT=8026

# --- asset group: unset so the runner splits task1 vs tasks2-26 and picks the VLM ckpt ----
#   task 1     -> vlm_task1
#   tasks 2-26 -> vlm_tasks1to26_ckpt74500
# Both groups share the same frozen VLA.
unset PREDIMEM_ASSET_GROUP

# --- GPU split: VLA server on GPU 0, evaluator (VLM planner client) on GPU 1 -------------
export SERVER_CUDA_VISIBLE_DEVICES=0
export EVAL_CUDA_VISIBLE_DEVICES=1
