#!/usr/bin/env bash
# =========================================================================================
# BOLT-Sync: task 19 only, 1x10 (seed 100), h0, STRICT early-stop.
#
# WHY THIS CELL
#   Task 19 is the transfer task that exposed AOM's scoring-vs-execution partition defect.
#   BOLT always seeds from the BDDL execution partition, so this cell is the first
#   like-for-like measurement of whether the unified Arbiter + dual-clock + Claim ledger
#   moves the score where AOM v6 sat at 19.99 vs GPM 26.67.
#
# STRICT EARLY-STOP (same calibration as AOM t19 v6 — DO NOT raise the floor)
#   Armed from trial 3. Three rules, any one fires -> rc=79:
#     1. all scored trials so far are 0.0
#     2. running mean <= FLOOR=8  (calibrated: same-seed nomem min early floor is 13.32;
#        a floor of 15 previously killed the CONTROL itself at k=9)
#     3. last CONSEC_M=3 trials are all 0.0
#   selftest_futility.py T10 replays THESE exact parameters against nomem_s100 and must
#   pass before any sbatch is submitted.
#
# LIKE-FOR-LIKE BASELINES (seed 100 ONLY)
#   nomem_s100       13.3
#   pullmem_er_s100  16.6
#   evmem_s100       20.0
#   AOM v6_s100      20.0
#   evmem_gpm_s100   26.7   <- beat target
#
# USAGE
#   ./run_bolt_t19_1x10.sh --dry-run
#   ./run_bolt_t19_1x10.sh --tag v1
# =========================================================================================

set -uo pipefail

ROOT="/project/peilab/why/RoboMemArena"
EXP_DIR="${ROOT}/experiments/mem_efficacy"
RESULTS="${EXP_DIR}/results"

TASK=19
SEEDS="100"
TRIALS=10
PROFILE="h0"
FUTILITY_TRIALS=3       # arm from trial 3
FUTILITY_CONSEC_M=3     # 3 trailing zeros
FLOOR=8                 # CALIBRATED for task 19 (control early floor 13.32; margin 5.3)
BASELINE_NOMEM_S100=13.3
BASELINE_AOM_V6_S100=20.0
BASELINE_GPM_S100=26.7
WALL=6400
STIME="03:30:00"
ARMS_CSV="bolt"
TAG="v1"

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
GATE_PYTHONPATH="${ROOT}/evaluation_benchmark:${ROOT}/evaluation_benchmark/scripts:${EXP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[gate] compiling evaluator + BOLT modules before any sbatch ..."
if ! "${EVAL_PY}" -c "
import ast, sys, pathlib
paths = [
    r'''${EVALUATOR}''',
    r'''${EXP_DIR}/selftest_futility.py''',
    r'''${EXP_DIR}/selftest_bolt.py''',
    r'''${EXP_DIR}/memexp_bolt_bind.py''',
]
bolt = pathlib.Path(r'''${EXP_DIR}/bolt''')
paths += sorted(str(p) for p in bolt.glob('*.py'))
for path in paths:
    try:
        ast.parse(open(path, encoding='utf-8').read())
    except SyntaxError as e:
        print(f'SYNTAX: {path}:{e.lineno}: {e.msg}', file=sys.stderr)
        sys.exit(2)
print(f'compile OK ({len(paths)} files)')
"; then
  echo "[ERROR] pre-submit compile gate FAILED" >&2
  exit 2
fi

echo "[gate] running BOLT selftest ..."
if ! PYTHONPATH="${GATE_PYTHONPATH}" "${EVAL_PY}" "${EXP_DIR}/selftest_bolt.py"; then
  echo "[ERROR] BOLT selftest FAILED -- refusing to submit" >&2
  exit 2
fi

echo "[gate] running FUTILITY calibration selftest (this runner's own parameters) ..."
if ! FUTILITY_WATCH_UNDER_TEST="${FUTILITY_TRIALS}" \
     FUTILITY_FLOOR_UNDER_TEST="${FLOOR}" \
     FUTILITY_CONSEC_UNDER_TEST="${FUTILITY_CONSEC_M}" \
     FUTILITY_TASK_UNDER_TEST="${TASK}" \
     FUTILITY_SEED_UNDER_TEST="${SEEDS%%,*}" \
     PYTHONPATH="${GATE_PYTHONPATH}" \
     "${EVAL_PY}" "${EXP_DIR}/selftest_futility.py"; then
  echo "[ERROR] FUTILITY selftest FAILED -- refusing to submit" >&2
  exit 2
fi

submitted=0
for arm in "${ARMS[@]}"; do
  art="${RESULTS}/bolt_t${TASK}_${arm}_${PROFILE}_1x10_${TAG}"

  echo "-----------------------------------------------------------------------------------"
  echo "task=${TASK}  arm=${arm}  seeds='${SEEDS}'  trials=${TRIALS}  profile=${PROFILE}  tag=${TAG}"
  echo "  art=${art}"
  echo "  futility STRICT: armed from ${FUTILITY_TRIALS} trials ->"
  echo "            all-zero, OR running mean <= ${FLOOR}% (calibrated), OR last ${FUTILITY_CONSEC_M} trials all 0.0"
  echo "            (LIKE-FOR-LIKE seed100: nomem ${BASELINE_NOMEM_S100} | AOM v6 ${BASELINE_AOM_V6_S100} | GPM ${BASELINE_GPM_S100})"
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
      --job-name="rma-bolt-t${TASK}-1x10"
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
echo "[OK] submitted ${submitted} BOLT t19 job(s) arm(s) '${ARMS_CSV}' tag=${TAG}."
