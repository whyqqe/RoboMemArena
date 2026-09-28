#!/usr/bin/env bash
# =========================================================================================
# AOM v3: task 19 only, 1x10 (seed 100), h0, with a recalibrated STRICT early-stop.
#
# WHAT v3 CHANGES (the architectural fix, not a tuning change)
#   v1/v2/v3/v4 all seeded the obligation graph from `_task_specs(task_id)`, which is the SCORING
#   partition. Task 19's scoring table is three `Place_*_Cabinet2` stages; its EXECUTION partition
#   (`primitive_order`) is pick->place for each of three objects. Seeding from the scoring table
#   therefore minted a graph with NO GRASP OBLIGATION AT ALL, and the board's own discipline
#   ("serve only the ACTIVE obligation") then commanded a place -- an action that PRESUPPOSES a held
#   object -- for essentially the whole episode. Measured on the archived runs:
#       arm        pick prompts   place prompts
#       nomem          332             86
#       AOM v2          15            615
#       AOM v4          10            504
#   v3 seeds from the EXECUTION partition via the harness's OWN bridge
#   (`harness.stage_mapper.expected_primitive_for_stage`), settles each grasp derivationally with
#   the place it feeds, and stops redacting object identity on a TRANSFER task (memory_type 'T'),
#   where the object's identity is the instruction rather than the answer being withheld.
#   The change is gated by `A.should_seed_execution`: execution seeding is used ONLY when the
#   scoring partition omits a grasp that PRECEDES a scored phase. Measured over all 26 tasks:
#       t5  -> OFF (partitions identical)      t8  -> OFF (omitted phase is downstream)
#       t19 -> ON                              t22 -> OFF
#   so the archived task-5 (45.0 vs nomem 13.8) and task-8 (43.3 vs GPM 46.7) results remain valid
#   controls and this run stays a single-variable comparison.
#
# THE EARLY-STOP, RECALIBRATED (it was measuring the CONTROL)
#   v4 used FUTILITY_MEAN_FLOOR_PCT=15 and was killed at trial 9 with mean 14.8. Replayed against
#   the archived same-seed control, `hard3_t19_nomem_s100` reaches mean 14.8 at k=9 and 13.3 at
#   k=10 -- i.e. the gate fired on a trajectory indistinguishable from the no-memory control's.
#   A floor above a control's own early floor kills the control too, so v4's 14.8 is a TRUNCATION,
#   not a measurement. Floor is now 8, the calibrated value from `selftest_futility.py` (task 19
#   baseline early floor 13.32, margin 5.3).
#
#   Arming moves from trial 6 to trial 3 as requested, which is the strictest window that this
#   task's archived cells support. Calibration replayed over every archived task-19 cell
#   (arm x seed, n=16):
#     * No cell has an all-zero first three trials except `pullmem_er_s130` = [33.3, 0, 0], the one
#       genuinely dead cell (final mean 10.0) -- so the rule kills dead runs and nothing else.
#     * The floor rule never fires before the final trial on any cell.
#     * The same-seed control `nomem_s100` fires only at k=10, i.e. on the last trial, costing 0.
#   Effective rule from k=3: all three trials 0.0 -> stop; plus the calibrated floor and a
#   trailing run of 3 zeros.
#
# LIKE-FOR-LIKE BASELINES (seed 100 ONLY -- this runner uses SEEDS="100")
#   These are per-seed cells, NOT the 3-seed macro means. Comparing a seed-100 arm against a
#   3-seed macro is the same category error as the floor-vs-control-floor one above.
#       nomem_s100            13.3   <- the control this run must beat
#       pullmem_er_s100       16.6
#       evmem_s100            20.0
#       AOM v2_s100           23.3   <- the previous AOM high-water mark
#       evmem_gpm_s100        26.7   <- the strongest archived arm on this task
#
# USAGE
#   ./run_aom_t19_1x10.sh --dry-run
#   ./run_aom_t19_1x10.sh --tag v5
# =========================================================================================

set -uo pipefail

ROOT="/project/peilab/why/RoboMemArena"
EXP_DIR="${ROOT}/experiments/mem_efficacy"
RESULTS="${EXP_DIR}/results"

TASK=19
SEEDS="100"
TRIALS=10
PROFILE="h0"
FUTILITY_TRIALS=3       # arm the gate from trial 3 (was 6)
FUTILITY_CONSEC_M=3     # 3 trailing zeros; at k=3 this reads "the first three trials are all 0.0"
FLOOR=8                 # CALIBRATED for task 19. NOT 15: the same-seed control passes through
                        # mean 14.8 at k=9, so a floor of 15 fires on the control itself.
BASELINE_NOMEM_S100=13.3
BASELINE_AOM_V2_S100=23.3
BASELINE_GPM_S100=26.7
WALL=6400
STIME="03:30:00"
ARMS_CSV="evmem_aom"
TAG="v5"

DRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY=1 ;;
    --arms)      ARMS_CSV="${2:?--arms needs a comma-separated list}"; shift ;;
    --arms=*)    ARMS_CSV="${1#--arms=}" ;;
    --tag)       TAG="${2:?--tag needs a label}"; shift ;;
    --tag=*)     TAG="${1#--tag=}" ;;
    -h|--help)   sed -n '1,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
# The gates import the harness's own modules (`harness.stage_mapper` is the bridge the AOM fix
# uses), so the gate must run with the same import path the arm runs with. Without this the
# mapper import fails, every mapper-dependent assertion reports FAIL, and the gate is correct to
# refuse the submit -- but for an environment reason rather than a code reason.
GATE_PYTHONPATH="${ROOT}/evaluation_benchmark:${EXP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
echo "[gate] compiling shared evaluator + AOM modules before any sbatch ..."
if ! "${EVAL_PY}" -c "
import ast, sys
for path in [r'''${EVALUATOR}''',
             r'''${EXP_DIR}/census_channels.py''',
             r'''${EXP_DIR}/memexp_aom.py''',
             r'''${EXP_DIR}/memexp_aom_bind.py''',
             r'''${EXP_DIR}/selftest_futility.py''']:
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

echo "[gate] running AOM selftest ..."
if ! PYTHONPATH="${GATE_PYTHONPATH}" "${EVAL_PY}" "${EXP_DIR}/selftest_aom.py"; then
  echo "[ERROR] AOM selftest FAILED -- refusing to submit" >&2
  exit 2
fi

# The futility gate is the only mechanism here that kills a live run on the basis of a SCORE, and
# a mis-set floor is invisible in the results (a stopped arm and a failing arm look identical).
# T9 asserts that the parameters THIS script passes cannot fire on the archived same-seed control
# before its final trial.
echo "[gate] running FUTILITY calibration selftest (T9: this runner's own parameters) ..."
if ! FUTILITY_WATCH_UNDER_TEST="${FUTILITY_TRIALS}" \
     FUTILITY_FLOOR_UNDER_TEST="${FLOOR}" \
     FUTILITY_CONSEC_UNDER_TEST="${FUTILITY_CONSEC_M}" \
     FUTILITY_TASK_UNDER_TEST="${TASK}" \
     FUTILITY_SEED_UNDER_TEST="${SEEDS%%,}" \
     PYTHONPATH="${GATE_PYTHONPATH}" \
     "${EVAL_PY}" "${EXP_DIR}/selftest_futility.py"; then
  echo "[ERROR] FUTILITY selftest FAILED -- refusing to submit" >&2
  exit 2
fi

submitted=0
for arm in "${ARMS[@]}"; do
  art="${RESULTS}/aom_t${TASK}_${arm}_${PROFILE}_1x10_${TAG}"

  echo "-----------------------------------------------------------------------------------"
  echo "task=${TASK}  arm=${arm}  seeds='${SEEDS}'  trials=${TRIALS}  profile=${PROFILE}  tag=${TAG}"
  echo "  art=${art}"
  echo "  futility STRICT: armed from ${FUTILITY_TRIALS} trials ->"
  echo "            all-zero, OR running mean <= ${FLOOR}% (calibrated), OR last ${FUTILITY_CONSEC_M} trials all 0.0"
  echo "            (LIKE-FOR-LIKE, seed 100: nomem ${BASELINE_NOMEM_S100} | AOM v2 ${BASELINE_AOM_V2_S100} | GPM ${BASELINE_GPM_S100})"
  echo "-----------------------------------------------------------------------------------"

  cmd=(env
    STAGE1_PROFILES="${PROFILE}"
    STAGE1_TRIALS="${TRIALS}"
    ARM_OVERRIDE="${arm}"
    TASKS_JSON_SCOPE="[${TASK}]"
    SEED_LIST="${SEEDS}"
    STAGE1_ART_ROOT="${art}"
    STAGE1_MAX_WALL_SEC="${WALL}"
    STAGE1_ALLOW_RERUN=1
    PLANNER_MAX_CONSECUTIVE_FAILURES=3
    EARLY_STOP_ON_QUOTA=1
    FUTILITY_STOP_TRIALS="${FUTILITY_TRIALS}"
    FUTILITY_MEAN_FLOOR_PCT="${FLOOR}"
    FUTILITY_CONSEC_ZERO_M="${FUTILITY_CONSEC_M}"
    sbatch
      --job-name="rma-aom-t${TASK}-1x10"
      --time="${STIME}"
      "${EXP_DIR}/run_stage1.sbatch")

  if [[ -n "${DRY}" ]]; then
    echo "  [dry-run] task=${TASK} arm=${arm} -> ${art}"
    printf '    %q' "${cmd[@]}"; echo; echo
    continue
  fi

  out="$("${cmd[@]}" 2>&1)"
  rc=$?
  echo "${out}"
  if [[ ${rc} -ne 0 ]] || ! printf '%s' "${out}" | grep -q '^Submitted batch job '; then
    echo "[ERROR] submit failed for task=${TASK} arm=${arm} rc=${rc}" >&2
    continue
  fi
  submitted=$((submitted + 1))
  echo
done

if [[ -n "${DRY}" ]]; then
  echo "[DRY-RUN] would submit ${#ARMS[@]} job(s)."
  exit 0
fi
echo "[OK] submitted ${submitted} AOM t19 job(s) arm(s) '${ARMS_CSV}' tag=${TAG}."
