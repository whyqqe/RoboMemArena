#!/usr/bin/env bash
# =========================================================================================
# AOM: remaining hard tasks {5, 19, 22}, 1x10 each, h0, with early-stop.
#
# GATE PER TASK, AND WHY IT IS NOT THE SAME AS TASK 8's
#   Task 8 uses the consecutive-zero rule because it has no usable absolute floor (its control's
#   early running mean reaches 0.0). The other three DO have usable floors -- each is set strictly
#   below the minimum running mean any archived control cell reaches at any n >= the window, so no
#   control can trip it. That property is asserted by `selftest_futility.py` T7.
#
#   THE CONSECUTIVE-ZERO RULE IS DELIBERATELY **OFF** HERE, and that is a measurement, not a
#   preference. Replaying it (w=6, M=3) against the MATCHED seed-100 controls:
#     t5  nomem_s100 (final 13.8) -> TRIPS at k=8. The gate would have stopped the CONTROL.
#     t19 nomem_s100 (final 13.3) -> trips at k=10, i.e. only on the final trial: useless.
#     t22 nomem_s100 (final 73.3) -> clean.
#   A rule that stops task 5's own control cannot measure task 5's treatment. It stays on for
#   task 8, where the same replay over the GPM baseline (46.7) is clean, and off everywhere the
#   control would trip it.
#
#   So "带早停" here means: all-zero, OR the running mean at or below that task's calibrated
#   floor, armed after 6 scored trials. Same as the GPM run on these tasks, which keeps the
#   AOM-vs-GPM comparison single-variable in the gate as well as in the arm.
#
# USAGE
#   ./run_aom_remaining_1x10.sh --dry-run
#   ./run_aom_remaining_1x10.sh                       # submit now
#   ./run_aom_remaining_1x10.sh --after 617123        # submit, gated on job 617123 exiting 0
# =========================================================================================

set -uo pipefail

ROOT="/project/peilab/why/RoboMemArena"
EXP_DIR="${ROOT}/experiments/mem_efficacy"
RESULTS="${EXP_DIR}/results"

TASKS="5 19 22"
SEEDS="100"
TRIALS=10
PROFILE="h0"
FUTILITY_TRIALS=6
AFTER=""
ARMS_CSV="evmem_aom"
TAG="v2"

DRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY=1 ;;
    --after)     AFTER="${2:?--after needs a job id}"; shift ;;
    --after=*)   AFTER="${1#--after=}" ;;
    --arms)      ARMS_CSV="${2:?--arms needs a comma-separated list}"; shift ;;
    --arms=*)    ARMS_CSV="${1#--arms=}" ;;
    --tag)       TAG="${2:?--tag needs a label}"; shift ;;
    --tag=*)     TAG="${1#--tag=}" ;;
    -h|--help)   sed -n '1,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "[ERROR] unknown argument '$1'" >&2; exit 2 ;;
  esac
  shift
done

IFS=',' read -r -a ARMS <<<"${ARMS_CSV}"
for arm in "${ARMS[@]}"; do
  if [[ ! -f "${EXP_DIR}/arms/${arm}.sh" ]]; then
    echo "[ERROR] missing arm file: ${EXP_DIR}/arms/${arm}.sh" >&2; exit 2
  fi
done

EVAL_PY="${ROOT}/.venv/bin/python"
EVALUATOR="${ROOT}/evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py"
echo "[gate] compiling shared evaluator before any sbatch ..."
if ! "${EVAL_PY}" -c "
import ast, sys
for path in [r'''${EVALUATOR}''', r'''${EXP_DIR}/census_channels.py''']:
    try:
        ast.parse(open(path, encoding='utf-8').read())
    except SyntaxError as e:
        print(f'SYNTAX: {path}:{e.lineno}: {e.msg}', file=sys.stderr)
        sys.exit(2)
print('compile OK')
"; then
  echo "[ERROR] pre-submit compile gate FAILED" >&2
  exit 2
fi

# Absolute running-mean floors, each strictly below the minimum running mean of every archived
# control cell for that task (margin in brackets). Calibrated once in the GPM run and reused
# unchanged, so the gate is not a second experimental variable.
mean_floor_for() {
  case "$1" in
    5)  printf '5'  ;;   # nomem early floor  9.72 -> 5   (margin 4.7)
    19) printf '8'  ;;   # nomem early floor 13.32 -> 8   (margin 5.3)
    22) printf '25' ;;   # nomem early floor 33.33 -> 25  (margin 8.3)
    *)  printf '0'  ;;
  esac
}
baseline_for() {
  case "$1" in
    5)  printf '13.75' ;;  19) printf '25.55' ;;  22) printf '70.00' ;;  *) printf '0' ;;
  esac
}
wall_for() {
  case "$1" in
    5)  printf '7200' ;;  19) printf '6400' ;;  22) printf '7200' ;;  *) printf '7200' ;;
  esac
}
slurm_for() {
  case "$1" in
    5)  printf '04:00:00' ;;  19) printf '03:30:00' ;;  22) printf '04:00:00' ;;  *) printf '04:00:00' ;;
  esac
}

if [[ -n "${AFTER}" ]]; then
  echo "[gate] these jobs are submitted with --dependency=afterok:${AFTER}"
  echo "       (a futility stop exits 79, quota 77, health 78 -- any of which leaves the"
  echo "        dependency unsatisfied, so the remaining tasks are simply not launched)"
else
  echo "[gate] NO dependency: submitting immediately."
fi

submitted=0
for arm in "${ARMS[@]}"; do
  for task in ${TASKS}; do
    floor="$(mean_floor_for "${task}")"
    base="$(baseline_for "${task}")"
    wall="$(wall_for "${task}")"
    stime="$(slurm_for "${task}")"
    art="${RESULTS}/aom_t${task}_${arm}_${PROFILE}_1x10_${TAG}"

    echo "-----------------------------------------------------------------------------------"
    echo "task=${task}  arm=${arm}  seeds='${SEEDS}'  trials=${TRIALS}  profile=${PROFILE}  tag=${TAG}"
    echo "  art=${art}"
    echo "  futility: after ${FUTILITY_TRIALS} trials, all-zero OR running mean <= ${floor}%"
    echo "            (nomem reference ~${base}; consecutive-zero rule OFF here -- its replay"
    echo "             trips the task-5 control itself)"
    echo "-----------------------------------------------------------------------------------"

    cmd=(env
      STAGE1_PROFILES="${PROFILE}"
      STAGE1_TRIALS="${TRIALS}"
      ARM_OVERRIDE="${arm}"
      TASKS_JSON_SCOPE="[${task}]"
      SEED_LIST="${SEEDS}"
      STAGE1_ART_ROOT="${art}"
      STAGE1_MAX_WALL_SEC="${wall}"
      STAGE1_ALLOW_RERUN=1
      PLANNER_MAX_CONSECUTIVE_FAILURES=3
      EARLY_STOP_ON_QUOTA=1
      FUTILITY_STOP_TRIALS="${FUTILITY_TRIALS}"
      FUTILITY_MEAN_FLOOR_PCT="${floor}"
      FUTILITY_CONSEC_ZERO_M=0
      sbatch
        --job-name="rma-aom-t${task}-1x10"
        --time="${stime}")

    # `--dependency` is appended as a real argument rather than smuggled through the environment,
    # because sbatch's own flag is what Slurm enforces.
    if [[ -n "${AFTER}" ]]; then
      cmd+=(--dependency="afterok:${AFTER}")
    fi
    cmd+=("${EXP_DIR}/run_stage1.sbatch")

    if [[ -n "${DRY}" ]]; then
      echo "  [dry-run] task=${task} arm=${arm} -> ${art}"
      printf '    %q' "${cmd[@]}"; echo; echo
      continue
    fi

    out="$("${cmd[@]}" 2>&1)"
    rc=$?
    echo "${out}"
    if [[ ${rc} -ne 0 ]] || ! printf '%s' "${out}" | grep -q '^Submitted batch job '; then
      echo "[ERROR] submit failed for task=${task} arm=${arm} rc=${rc}" >&2
      continue
    fi
    submitted=$((submitted + 1))
    echo
  done
done

if [[ -n "${DRY}" ]]; then
  echo "[DRY-RUN] would submit $(( ${#ARMS[@]} * 3 )) job(s)."
  exit 0
fi
echo "[OK] submitted ${submitted} AOM job(s) for tasks '${TASKS}' arm(s) '${ARMS_CSV}'."
