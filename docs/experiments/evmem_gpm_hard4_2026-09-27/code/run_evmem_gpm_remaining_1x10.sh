#!/usr/bin/env bash
# =========================================================================================
# EvMem-GPM: remaining hard tasks {5, 19, 22}, 1x10 each, h0, calibrated futility.
# Task 8 already completed (evmem_gpm_t8_*, 46.7 vs nomem ~26.7).
# =========================================================================================

set -uo pipefail

ROOT="/project/peilab/why/RoboMemArena"
EXP_DIR="${ROOT}/experiments/mem_efficacy"
RESULTS="${EXP_DIR}/results"

TASKS="5 19 22"
SEEDS="100"
TRIALS=10
PROFILE="h0"
ARM="evmem"
FUTILITY_TRIALS=6

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY=1

mean_floor_for() {
  case "$1" in
    # Absolute running-mean floors (treatment). Calibrated soft: kill clear duds, allow recovery.
    5)  printf '5'  ;;   # nomem ~13.8; prior t5 treatment used 5
    19) printf '8'  ;;   # nomem ~25.6; prior hard4 used 8
    22) printf '25' ;;   # nomem ~70; use 25 not 50 (50 false-stopped GPM-less v3)
    *)  printf '0'  ;;
  esac
}
baseline_for() {
  case "$1" in
    5)  printf '13.75' ;;
    19) printf '25.55' ;;
    22) printf '70.00' ;;
    *)  printf '0'     ;;
  esac
}
wall_for() {
  case "$1" in
    5)  printf '7200'  ;;
    19) printf '6400'  ;;
    22) printf '7200'  ;;
  esac
}
slurm_for() {
  case "$1" in
    5)  printf '04:00:00' ;;
    19) printf '03:30:00' ;;
    22) printf '04:00:00' ;;
  esac
}

submitted=0
expected=0
for task in ${TASKS}; do
  expected=$((expected + 1))
  floor="$(mean_floor_for "${task}")"
  base="$(baseline_for "${task}")"
  wall="$(wall_for "${task}")"
  stime="$(slurm_for "${task}")"
  art="${RESULTS}/evmem_gpm_t${task}_${ARM}_${PROFILE}_1x10"

  echo "-------------------------------------------------------------------------------"
  echo "task=${task}  arm=${ARM}  seeds='${SEEDS}'  trials=${TRIALS}  profile=${PROFILE}"
  echo "  art=${art}"
  echo "  futility: after ${FUTILITY_TRIALS} trials, all-zero OR mean <= ${floor}% (nomem ~${base})"
  echo "  arch: GPM L0+L1 (same as completed t8)"
  echo "-------------------------------------------------------------------------------"

  cmd=(env
    STAGE1_PROFILES="${PROFILE}"
    STAGE1_TRIALS="${TRIALS}"
    ARM_OVERRIDE="${ARM}"
    TASKS_JSON_SCOPE="[${task}]"
    SEED_LIST="${SEEDS}"
    STAGE1_ART_ROOT="${art}"
    STAGE1_MAX_WALL_SEC="${wall}"
    STAGE1_ALLOW_RERUN=1
    PLANNER_MAX_CONSECUTIVE_FAILURES=3
    EARLY_STOP_ON_QUOTA=1
    FUTILITY_STOP_TRIALS="${FUTILITY_TRIALS}"
    FUTILITY_MEAN_FLOOR_PCT="${floor}"
    MEMEXP_EVMEM_GATE_ATTEMPTS=3
    MEMEXP_EVMEM_GATE_STALL=3
    MEMEXP_EVMEM_ATTRACTOR_REPEAT=2
    sbatch
      --job-name="rma-evmem-gpm-t${task}-1x10"
      --time="${stime}"
      "${EXP_DIR}/run_stage1.sbatch")

  if [[ -n "${DRY}" ]]; then
    printf '  %q' "${cmd[@]}"; echo; echo
    continue
  fi
  out="$("${cmd[@]}" 2>&1)"
  rc=$?
  echo "${out}"
  if [[ ${rc} -ne 0 ]] || ! printf '%s' "${out}" | grep -q '^Submitted batch job '; then
    echo "[ERROR] submit failed for task=${task} rc=${rc}" >&2
    continue
  fi
  submitted=$((submitted + 1))
  echo
done

if [[ -n "${DRY}" ]]; then
  echo "[DRY-RUN] would submit ${expected} job(s)."
  exit 0
fi
echo "[OK] submitted ${submitted}/${expected} EvMem-GPM 1x10 jobs (t5/t19/t22)."
