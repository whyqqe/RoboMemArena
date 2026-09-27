#!/usr/bin/env bash
# =========================================================================================
# EvMem-GPM: t8 only, 1x10, h0, early-stop.
# t8 is counting-pour (Lift → Pour_One → Pour_Two onto pudding); same family as t22.
# Futility: after 6 trials, all-zero OR mean <= 5% (calibrated soft floor; nomem ~25.56).
# Mean floor kept modest: hard3 showed t8 can open with zeros then recover (floor=0 historically).
# =========================================================================================

set -uo pipefail

ROOT="/project/peilab/why/RoboMemArena"
EXP_DIR="${ROOT}/experiments/mem_efficacy"
RESULTS="${EXP_DIR}/results"

TASK=8
SEEDS="100"
TRIALS=10
PROFILE="h0"
ARM="evmem"
FUTILITY_TRIALS=6
FLOOR=5
BASELINE=25.56
WALL=4800
STIME="03:00:00"

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY=1

art="${RESULTS}/evmem_gpm_t${TASK}_${ARM}_${PROFILE}_1x10"

echo "-------------------------------------------------------------------------------"
echo "task=${TASK}  arm=${ARM}  seeds='${SEEDS}'  trials=${TRIALS}  profile=${PROFILE}"
echo "  art=${art}"
echo "  futility: after ${FUTILITY_TRIALS} trials, all-zero OR mean <= ${FLOOR}% (nomem ~${BASELINE})"
echo "  arch: GPM L0+L1 (NEED predicates + C-gate + gated evidence)"
echo "-------------------------------------------------------------------------------"

cmd=(env
  STAGE1_PROFILES="${PROFILE}"
  STAGE1_TRIALS="${TRIALS}"
  ARM_OVERRIDE="${ARM}"
  TASKS_JSON_SCOPE="[${TASK}]"
  SEED_LIST="${SEEDS}"
  STAGE1_ART_ROOT="${art}"
  STAGE1_MAX_WALL_SEC="${WALL}"
  STAGE1_ALLOW_RERUN=1
  PLANNER_MAX_CONSECUTIVE_FAILURES=3
  EARLY_STOP_ON_QUOTA=1
  FUTILITY_STOP_TRIALS="${FUTILITY_TRIALS}"
  FUTILITY_MEAN_FLOOR_PCT="${FLOOR}"
  MEMEXP_EVMEM_GATE_ATTEMPTS=3
  MEMEXP_EVMEM_GATE_STALL=3
  MEMEXP_EVMEM_ATTRACTOR_REPEAT=2
  sbatch
    --job-name="rma-evmem-gpm-t${TASK}-1x10"
    --time="${STIME}"
    "${EXP_DIR}/run_stage1.sbatch")

if [[ -n "${DRY}" ]]; then
  printf '  %q' "${cmd[@]}"; echo
  exit 0
fi

out="$("${cmd[@]}" 2>&1)"
rc=$?
echo "${out}"
if [[ ${rc} -ne 0 ]] || ! printf '%s' "${out}" | grep -q '^Submitted batch job '; then
  echo "[ERROR] submit failed rc=${rc}" >&2
  exit 1
fi
echo "[OK] submitted EvMem-GPM t${TASK} 1x10 -> ${art}"
