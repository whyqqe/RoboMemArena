#!/usr/bin/env bash
# =========================================================================================
# AOM v6 (current workspace): hard-3 remaining tasks {5, 8, 22}, 1x10 each, seed 100, h0.
#
# WHY THESE THREE
#   Task 19 already ran under the same code path (tag v6, job 617508, mean 20.0). This runner
#   covers the other three tasks of the hard suite so the v6 claim can be scored against GPM
#   on every seed-100 cell that has an archived GPM number.
#
# TARGET (LIKE-FOR-LIKE, seed 100 ONLY)
#   Beat the archived GPM mean on the same seed. Macro / multi-seed averages are not the bar.
#       task 5  : GPM 20.0   | AOM v2 archive 45.0  (already above; v6 must not regress below GPM)
#       task 8  : GPM 46.7   | AOM v2 archive 43.3  (must clear 46.7)
#       task 22 : GPM 73.3   | no AOM archive on this seed
#
# EARLY-STOP: armed from trial 3 (stricter than the historical w=6 runners)
#   FUTILITY_STOP_TRIALS=3 for every task.
#   Absolute floor: calibrated BELOW each matched control's min running mean at k>=3, so the
#   gate cannot fire on the no-memory control's own early trajectory (the class of error that
#   truncated AOM t19 v4 at 14.8).
#       t5  floor=5   (nomem early floor 9.72)
#       t8  floor=0   DISABLED: nomem early floor is 0.0; no absolute threshold separates a
#                     failing arm from an unlucky control (same calibration as every prior t8 run)
#       t22 floor=25  (nomem early floor 33.33)
#   Consecutive-zero: M is the STRICTEST value that does NOT early-kill the GPM cell we are
#   trying to beat (replayed over archived GPM seed-100 traces):
#       t5  M=5  (GPM max consecutive zeros = 4, recovers to 75 on trial 10; M=3 kills GPM at k=8)
#       t8  M=3  (GPM max consecutive zeros = 2; clean)
#       t22 M=3  (GPM never trips)
#   All-zero still fires from k=3 on every task.
#
# USAGE
#   ./run_aom_v6_hard3_1x10.sh --dry-run
#   ./run_aom_v6_hard3_1x10.sh
#   ./run_aom_v6_hard3_1x10.sh --tag v6
# =========================================================================================

set -uo pipefail

ROOT="/project/peilab/why/RoboMemArena"
EXP_DIR="${ROOT}/experiments/mem_efficacy"
RESULTS="${EXP_DIR}/results"

TASKS="5 8 22"
SEEDS="100"
TRIALS=10
PROFILE="h0"
FUTILITY_TRIALS=3
ARMS_CSV="evmem_aom"
TAG="v6"

DRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY=1 ;;
    --arms)      ARMS_CSV="${2:?--arms needs a comma-separated list}"; shift ;;
    --arms=*)    ARMS_CSV="${1#--arms=}" ;;
    --tag)       TAG="${2:?--tag needs a label}"; shift ;;
    --tag=*)     TAG="${1#--tag=}" ;;
    --tasks)     TASKS="${2:?--tasks needs a space-separated list}"; shift ;;
    --tasks=*)   TASKS="${1#--tasks=}" ;;
    -h|--help)   sed -n '1,55p' "${BASH_SOURCE[0]}"; exit 0 ;;
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

mean_floor_for() {
  case "$1" in
    5)  printf '5'  ;;
    8)  printf '0'  ;;
    22) printf '25' ;;
    *)  printf '0'  ;;
  esac
}
consec_for() {
  case "$1" in
    5)  printf '5'  ;;   # M=3 falsely kills archived GPM_s100 at k=8
    8)  printf '3'  ;;
    22) printf '3'  ;;
    *)  printf '3'  ;;
  esac
}
gpm_target_for() {
  case "$1" in
    5)  printf '20.0' ;;
    8)  printf '46.7' ;;
    22) printf '73.3' ;;
    *)  printf '0' ;;
  esac
}
wall_for() {
  case "$1" in
    5)  printf '7200' ;;
    8)  printf '6400' ;;
    22) printf '7200' ;;
    *)  printf '7200' ;;
  esac
}
slurm_for() {
  case "$1" in
    5)  printf '04:00:00' ;;
    8)  printf '03:30:00' ;;
    22) printf '04:00:00' ;;
    *)  printf '04:00:00' ;;
  esac
}

EVAL_PY="${ROOT}/.venv/bin/python"
EVALUATOR="${ROOT}/evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py"
GATE_PYTHONPATH="${ROOT}/evaluation_benchmark:${EXP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[gate] compiling shared evaluator + AOM modules before any sbatch ..."
if ! "${EVAL_PY}" -c "
import ast, sys
for path in [r'''${EVALUATOR}''',
             r'''${EXP_DIR}/census_channels.py''',
             r'''${EXP_DIR}/memexp_aom.py''',
             r'''${EXP_DIR}/memexp_aom_bind.py''']:
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

# Assert the gate parameters of EVERY task in this runner against that task's archived GPM
# seed-100 cell: the gate must not kill the target we are trying to beat before its final trial.
echo "[gate] futility vs archived GPM (must not early-kill the beat-GPM target) ..."
cd "${EXP_DIR}"
for task in ${TASKS}; do
  floor="$(mean_floor_for "${task}")"
  consec="$(consec_for "${task}")"
  if ! FUTILITY_WATCH_UNDER_TEST="${FUTILITY_TRIALS}" \
       FUTILITY_FLOOR_UNDER_TEST="${floor}" \
       FUTILITY_CONSEC_UNDER_TEST="${consec}" \
       FUTILITY_TASK_UNDER_TEST="${task}" \
       FUTILITY_SEED_UNDER_TEST="${SEEDS%%,}" \
       PYTHONPATH="${GATE_PYTHONPATH}" \
       "${EVAL_PY}" - <<'PY'
import csv, os, sys
from selftest_futility import load_programs, run_verdict, _cells, RESULTS

watch = int(os.environ["FUTILITY_WATCH_UNDER_TEST"])
floor = float(os.environ["FUTILITY_FLOOR_UNDER_TEST"])
cz = int(os.environ["FUTILITY_CONSEC_UNDER_TEST"])
task = os.environ["FUTILITY_TASK_UNDER_TEST"]
seed = os.environ["FUTILITY_SEED_UNDER_TEST"]
_, verdict = load_programs()
# Prefer the GPM cell; fall back to nomem only if GPM is absent (should not happen for 5/8/22).
cands = _cells(f"evmem_gpm_t{task}_*/h0/*/prompt_trace.tsv")
cands = [p for p in cands if f"s{seed}" in p] or cands
if not cands:
    print(f"[FAIL] no archived GPM for task {task} seed {seed}", file=sys.stderr)
    sys.exit(2)
scores = []
with open(cands[0]) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r.get("stage_score_pct") not in (None, ""):
            scores.append(float(r["stage_score_pct"]))
n = len(scores)
fired = None
for k in range(1, n + 1):
    if k < watch:
        continue
    ser = ",".join(f"{x:g}" for x in scores[:k])
    v = run_verdict(verdict, k, max(scores[:k]), sum(scores[:k]) / k, floor, watch,
                    cz=cz, ser=ser)
    if v != "wait":
        fired = (k, v)
        break
if fired is not None and fired[0] < n:
    print(f"[FAIL] t{task}: gate early-kills GPM at k={fired[0]} ({fired[1]}) "
          f"w={watch} floor={floor} cz={cz}; scores={scores}", file=sys.stderr)
    sys.exit(2)
print(f"  [PASS] t{task}: GPM survives under w={watch} floor={floor} cz={cz} "
      f"(mean={sum(scores)/n:.1f})")
PY
  then
    echo "[ERROR] futility vs GPM FAILED for task ${task} -- refusing to submit" >&2
    exit 2
  fi
done

submitted=0
for arm in "${ARMS[@]}"; do
  for task in ${TASKS}; do
    floor="$(mean_floor_for "${task}")"
    consec="$(consec_for "${task}")"
    gpm="$(gpm_target_for "${task}")"
    wall="$(wall_for "${task}")"
    stime="$(slurm_for "${task}")"
    art="${RESULTS}/aom_t${task}_${arm}_${PROFILE}_1x10_${TAG}"

    echo "-----------------------------------------------------------------------------------"
    echo "task=${task}  arm=${arm}  seeds='${SEEDS}'  trials=${TRIALS}  profile=${PROFILE}  tag=${TAG}"
    echo "  art=${art}"
    echo "  TARGET: beat GPM seed100 = ${gpm}"
    echo "  futility STRICT from trial ${FUTILITY_TRIALS}:"
    echo "            all-zero, OR running mean <= ${floor}%, OR last ${consec} trials all 0.0"
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
      FUTILITY_CONSEC_ZERO_M="${consec}"
      sbatch
        --job-name="rma-aom-t${task}-v6"
        --time="${stime}"
        "${EXP_DIR}/run_stage1.sbatch")

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
  echo "[DRY-RUN] would submit $(( ${#ARMS[@]} * $(wc -w <<<"${TASKS}") )) job(s)."
  exit 0
fi
echo "[OK] submitted ${submitted} AOM v6 hard-3 job(s) tasks='${TASKS}' arm(s)='${ARMS_CSV}' tag=${TAG}."
echo "     Success criterion per cell: mean(stage) > GPM seed100 (t5>20.0, t8>46.7, t22>73.3)."
