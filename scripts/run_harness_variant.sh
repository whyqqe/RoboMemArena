#!/usr/bin/env bash
# Apply a named harness ablation variant via environment variables.
# Usage: source scripts/run_harness_variant.sh <variant>
set -euo pipefail

VARIANT="${1:?usage: run_harness_variant.sh VARIANT}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HARNESS_GLOBAL_RULES="${ROOT}/evaluation_benchmark/harness/global_rules.json"

# Shared eval defaults (caller may override TASKS_JSON, OUT_ROOT, etc.)
export HARNESS_API_PLANNER=0
export HARNESS_PERSIST_MEMORY=1
export HARNESS_FORCE_VLM_REPLAN=1

apply_variant() {
  case "${VARIANT}" in
    baseline)
      export HARNESS_ENABLE=0
      unset HARNESS_VLM_CONTEXT HARNESS_VLA_HINTS HARNESS_SUBTASK_OVERRIDE
      unset HARNESS_STAGE_CHECKPOINT HARNESS_RELEASE_ON_RETRY HARNESS_MAX_RETRIES HARNESS_STALL_STEPS
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=0
      export D_MERGE=6
      ;;
    v2_full)
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=1
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=1
      export HARNESS_STAGE_CHECKPOINT=1
      export HARNESS_RELEASE_ON_RETRY=1
      export HARNESS_MAX_RETRIES=2
      export HARNESS_STALL_STEPS=120
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=0
      export D_MERGE=6
      ;;
    memer_kf)
      # MemER-inspired: denser keyframe memory in VLM + full harness
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=1
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=1
      export HARNESS_STAGE_CHECKPOINT=1
      export HARNESS_RELEASE_ON_RETRY=1
      export HARNESS_MAX_RETRIES=2
      export HARNESS_STALL_STEPS=120
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=8
      export D_MERGE=4
      ;;
    ctx_only)
      # MemER/MemoryVLA read-time memory only (no execution harness)
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=1
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=0
      export HARNESS_STAGE_CHECKPOINT=0
      export HARNESS_RELEASE_ON_RETRY=0
      export HARNESS_MAX_RETRIES=0
      export HARNESS_STALL_STEPS=120
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=0
      export D_MERGE=6
      ;;
    retry_only)
      # HarnessVLA execution layer without VLM memory injection
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=0
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=0
      export HARNESS_STAGE_CHECKPOINT=1
      export HARNESS_RELEASE_ON_RETRY=1
      export HARNESS_MAX_RETRIES=2
      export HARNESS_STALL_STEPS=120
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=0
      export D_MERGE=6
      ;;
    override_only)
      # Stall analytic recovery without retry or memory context
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=0
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=1
      export HARNESS_STAGE_CHECKPOINT=0
      export HARNESS_RELEASE_ON_RETRY=0
      export HARNESS_MAX_RETRIES=0
      export HARNESS_STALL_STEPS=120
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=0
      export D_MERGE=6
      ;;
    stall_fast)
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=1
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=1
      export HARNESS_STAGE_CHECKPOINT=1
      export HARNESS_RELEASE_ON_RETRY=1
      export HARNESS_MAX_RETRIES=2
      export HARNESS_STALL_STEPS=80
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=0
      export D_MERGE=6
      ;;
    v1_hints)
      # Negative control: v1-style VLA prompt pollution
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=1
      export HARNESS_VLA_HINTS=1
      export HARNESS_SUBTASK_OVERRIDE=1
      export HARNESS_STAGE_CHECKPOINT=1
      export HARNESS_RELEASE_ON_RETRY=1
      export HARNESS_MAX_RETRIES=2
      export HARNESS_STALL_STEPS=120
      export HARNESS_SMART_RETRY=1
      export HARNESS_RETRY_SKIP_SCORE=95
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=0
      export D_MERGE=6
      ;;
    v21_combo)
      # v2.1: MemER keyframes + fast stall + full harness + smart retry
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=1
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=1
      export HARNESS_STAGE_CHECKPOINT=1
      export HARNESS_RELEASE_ON_RETRY=1
      export HARNESS_MAX_RETRIES=2
      export HARNESS_STALL_STEPS=80
      export HARNESS_SMART_RETRY=1
      export HARNESS_RETRY_SKIP_SCORE=95
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=8
      export D_MERGE=4
      export N_RECENT=7
      ;;
    v21_best)
      # P0 main: v21_combo + MemER resume hints + no retry on easy pour tasks
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=1
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=1
      export HARNESS_STAGE_CHECKPOINT=1
      export HARNESS_RELEASE_ON_RETRY=1
      export HARNESS_MAX_RETRIES=2
      export HARNESS_STALL_STEPS=80
      export HARNESS_SMART_RETRY=1
      export HARNESS_RETRY_SKIP_SCORE=95
      export HARNESS_RETRY_REQUIRE_PROGRESS=1
      export HARNESS_SKIP_RETRY_TASKS='[18,22]'
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=8
      export D_MERGE=4
      export N_RECENT=7
      ;;
    memory_kf)
      export HARNESS_ENABLE=0
      unset HARNESS_VLM_CONTEXT HARNESS_SUBTASK_OVERRIDE
      unset HARNESS_STAGE_CHECKPOINT HARNESS_RELEASE_ON_RETRY HARNESS_MAX_RETRIES
      export MEM_STAGE_ANCHOR=0
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=8
      export D_MERGE=4
      export N_RECENT=7
      ;;
    memory_ctx)
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=1
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=0
      export HARNESS_STAGE_CHECKPOINT=0
      export HARNESS_RELEASE_ON_RETRY=0
      export HARNESS_MAX_RETRIES=0
      export HARNESS_STALL_STEPS=80
      export MEM_STAGE_ANCHOR=0
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=8
      export D_MERGE=4
      export N_RECENT=7
      ;;
    memory_plus)
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=1
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=0
      export HARNESS_STAGE_CHECKPOINT=0
      export HARNESS_RELEASE_ON_RETRY=0
      export HARNESS_MAX_RETRIES=0
      export HARNESS_STALL_STEPS=80
      export MEM_STAGE_ANCHOR=1
      export MEM_SALIENCE_SUBTASK=1
      export MEM_BANK_MAX=8
      export MEM_CLUSTER_D=4
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=8
      export D_MERGE=4
      export N_RECENT=7
      ;;
    harness_exec)
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=0
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=0
      export HARNESS_STAGE_CHECKPOINT=1
      export HARNESS_RELEASE_ON_RETRY=1
      export HARNESS_MAX_RETRIES=2
      export HARNESS_STALL_STEPS=120
      export MEM_STAGE_ANCHOR=0
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=0
      export D_MERGE=6
      ;;
    harness_v21)
      export HARNESS_ENABLE=1
      export HARNESS_VLM_CONTEXT=1
      export HARNESS_VLA_HINTS=0
      export HARNESS_SUBTASK_OVERRIDE=1
      export HARNESS_STAGE_CHECKPOINT=1
      export HARNESS_RELEASE_ON_RETRY=1
      export HARNESS_MAX_RETRIES=2
      export HARNESS_STALL_STEPS=80
      export HARNESS_SMART_RETRY=1
      export HARNESS_RETRY_SKIP_SCORE=95
      export HARNESS_RETRY_REQUIRE_PROGRESS=1
      export HARNESS_SKIP_RETRY_TASKS='[18,22]'
      export MEM_STAGE_ANCHOR=0
      export VLM_USE_KEYFRAME_MEMORY=1
      export K_MAX=8
      export D_MERGE=4
      export N_RECENT=7
      ;;
    *)
      echo "[ERROR] unknown variant: ${VARIANT}" >&2
      echo "  valid: baseline memory_kf memory_ctx memory_plus harness_exec harness_v21 ..." >&2
      return 2
      ;;
  esac
}

apply_variant
echo "[INFO] harness variant=${VARIANT} HARNESS_ENABLE=${HARNESS_ENABLE:-0} K_MAX=${K_MAX:-0} STALL=${HARNESS_STALL_STEPS:-n/a}"
