#!/usr/bin/env bash
# =========================================================================================
# pushmem baseline: hard4 tasks {5, 8, 19, 22}, h0, seed 100, 1x10, NO futility early-stop.
#
# One job per task so a slow task cannot starve the others. FUTILITY_STOP_TRIALS=0 keeps every
# scored trial; the control-style completeness rule used for nomem in run_hard4_4x10.sh.
#
#   bash experiments/mem_efficacy/run_pushmem_hard4_1x10.sh --dry-run
#   bash experiments/mem_efficacy/run_pushmem_hard4_1x10.sh
#   TAG=rerun1 bash experiments/mem_efficacy/run_pushmem_hard4_1x10.sh
# =========================================================================================

set -uo pipefail

ROOT="/project/peilab/why/RoboMemArena"
EXP_DIR="${ROOT}/experiments/mem_efficacy"
RESULTS="${EXP_DIR}/results"

TASKS="5 8 19 22"
SEEDS="100"
TRIALS=10
PROFILE="h0"
ARM="pushmem"
TAG="${TAG:-v1}"

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY=1

wall_for() {
  case "$1" in
    5)  printf '7200'  ;;
    8)  printf '4800'  ;;
    19) printf '6400'  ;;
    22) printf '7200'  ;;
  esac
}
slurm_for() {
  case "$1" in
    5)  printf '04:00:00' ;;
    8)  printf '03:00:00' ;;
    19) printf '03:30:00' ;;
    22) printf '04:00:00' ;;
  esac
}

if [[ ! -f "${EXP_DIR}/arms/${ARM}.sh" ]]; then
  echo "[ERROR] missing arm script: arms/${ARM}.sh" >&2
  exit 2
fi

submitted=0
for task in ${TASKS}; do
  wall="$(wall_for "${task}")"
  stime="$(slurm_for "${task}")"
  art="${RESULTS}/pushmem_t${task}_${ARM}_${PROFILE}_1x10_${TAG}"

  if [[ -e "${art}" && -z "${DRY}" ]]; then
    echo "[ERROR] result path exists; choose a new TAG: ${art}" >&2
    exit 2
  fi

  echo "-------------------------------------------------------------------------------"
  echo "task=${task}  arm=${ARM}  seeds='${SEEDS}'  trials=${TRIALS}  profile=${PROFILE}  tag=${TAG}"
  echo "  art=${art}"
  echo "  futility: OFF (FUTILITY_STOP_TRIALS=0) — full 1x10 required"
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
    FUTILITY_STOP_TRIALS=0
    FUTILITY_MEAN_FLOOR_PCT=0
    FUTILITY_CONSEC_ZERO_M=0
    sbatch
      --job-name="rma-pushmem-t${task}-1x10"
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
    echo "[ERROR] sbatch failed for task ${task} (rc=${rc})" >&2
    exit 2
  fi
  submitted=$((submitted + 1))
done

echo
echo "submitted ${submitted}/4 pushmem hard4 1x10 jobs (no futility early-stop), tag=${TAG}"
