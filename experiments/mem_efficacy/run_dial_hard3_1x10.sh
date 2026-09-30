#!/usr/bin/env bash
# =========================================================================================
# DIAL: hard-3 tasks {5, 8, 22}, 1x10 each, seed 100, h0, STRICT early-stop.
#
# TARGET (LIKE-FOR-LIKE, seed 100 ONLY) — beat archived GPM on the same seed:
#   t5  GPM 20.0      (evmem_gpm_t5  mean 20.00, 5/10 zero)
#   t8  GPM 46.7      (evmem_gpm_t8  mean 46.68, 4/10 zero)
#   t22 GPM 73.3      (evmem_gpm_t22 mean 73.33, 1/10 zero)
#
# ARCHITECTURE (see dial/README.md).  One scheduler, driven by a posterior over WHY credit
# is not moving, over the action space {RESAMPLE, QUERY, RESYNC, PERSIST}:
#   bottleneck.py  the posterior (5 causes) + online lambda_hat / hold_horizon
#   strategy.py    attempt-cell space + hierarchical prior + stratified Thompson sampling
#   policy.py      the single law; PERSIST is bounded, no rejecting action exists
#   attribution.py the archived reward stream (9262 labelled attempts, 465 credited)
#   bind.py        planner-channel injection; observes the VLA prompt and never rewrites it
#
# WHAT THE ARCHIVE SAYS THIS BUYS (all measured, dial/falsify_report.json):
#   T1 attempt predicts credit   chi2=581, p<0.001, +0.0043 nats/ep held out
#                                lift: pick_up 0.25 vs grasp 0.099, same obligation
#   T2 gate tightness            within-arm t8 ladder spans 40.01 points with the
#                                architecture fixed (GPM 41.67 > AOM 37.92 > BOLT 29.27)
#   T3 credit lag vs horizon     median 25, p90 130 steps; GPM's 8-step and AOM's 12-step
#                                horizons would cancel 79.8% / 66.0% of successful attempts
#   T4 the objective is escape   score mass is bimodal: 39% exactly 0.0, 35% >= 66.6.
#                                The KPI is P(escape 0.0), not mean score.
#
# EARLY-STOP (arm-identical to the AOM/BOLT hard3 runners, so a kill is comparable):
#   t5  floor=5   M=5     t8  floor=0 (disabled)  M=3     t22 floor=25  M=3
#   Pre-submit gates: compile + dial.selftest + futility replay proving GPM survives.
#
# USAGE
#   ./run_dial_hard3_1x10.sh --dry-run
#   ./run_dial_hard3_1x10.sh --tag v1
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
ARMS_CSV="dial"
TAG="v1"

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
    -h|--help)   sed -n '1,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
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

mean_floor_for() { case "$1" in 5) printf '5';; 8) printf '0';; 22) printf '25';; *) printf '0';; esac; }
consec_for()     { case "$1" in 5) printf '5';; 8) printf '3';; 22) printf '3';; *) printf '3';; esac; }
gpm_target_for() { case "$1" in 5) printf '20.0';; 8) printf '46.7';; 22) printf '73.3';; *) printf '0';; esac; }
wall_for()       { case "$1" in 5) printf '7200';; 8) printf '6400';; 22) printf '7200';; *) printf '7200';; esac; }
slurm_for()      { case "$1" in 5) printf '04:00:00';; 8) printf '03:30:00';; 22) printf '04:00:00';; *) printf '04:00:00';; esac; }

EVAL_PY="${ROOT}/.venv/bin/python"
EVALUATOR="${ROOT}/evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py"
GATE_PYTHONPATH="${ROOT}/evaluation_benchmark:${ROOT}/evaluation_benchmark/scripts:${ROOT}/evaluation_benchmark/openpi_minimal_runtime:${EXP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[gate] compiling evaluator + DIAL modules before any sbatch ..."
if ! "${EVAL_PY}" -c "
import ast, sys, pathlib
paths = [
    r'''${EVALUATOR}''',
    r'''${EXP_DIR}/selftest_futility.py''',
    r'''${EXP_DIR}/memexp_dial_bind.py''',
]
dial = pathlib.Path(r'''${EXP_DIR}/dial''')
paths += sorted(str(p) for p in dial.glob('*.py'))
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

echo "[gate] running DIAL selftest (78 checks: invariants, attribution, sims, bind safety) ..."
cd "${EXP_DIR}"
if ! PYTHONPATH="${GATE_PYTHONPATH}:${EXP_DIR}" "${EVAL_PY}" -m dial.selftest; then
  echo "[ERROR] DIAL selftest FAILED -- refusing to submit" >&2
  exit 2
fi

echo "[gate] DIAL falsification suite (T1..T4 must all pass) ..."
if ! PYTHONPATH="${GATE_PYTHONPATH}:${EXP_DIR}" "${EVAL_PY}" -m dial.falsify; then
  echo "[ERROR] DIAL falsification FAILED -- refusing to submit" >&2
  exit 2
fi

echo "[gate] futility vs archived GPM (must not early-kill the beat-GPM target) ..."
for task in ${TASKS}; do
  floor="$(mean_floor_for "${task}")"
  consec="$(consec_for "${task}")"
  if ! FUTILITY_WATCH_UNDER_TEST="${FUTILITY_TRIALS}" \
       FUTILITY_FLOOR_UNDER_TEST="${floor}" \
       FUTILITY_CONSEC_UNDER_TEST="${consec}" \
       FUTILITY_TASK_UNDER_TEST="${task}" \
       FUTILITY_SEED_UNDER_TEST="${SEEDS%%,*}" \
       PYTHONPATH="${GATE_PYTHONPATH}" \
       "${EVAL_PY}" - <<'PY'
import csv, os, sys
from selftest_futility import load_programs, run_verdict, _cells

watch = int(os.environ["FUTILITY_WATCH_UNDER_TEST"])
floor = float(os.environ["FUTILITY_FLOOR_UNDER_TEST"])
cz = int(os.environ["FUTILITY_CONSEC_UNDER_TEST"])
task = os.environ["FUTILITY_TASK_UNDER_TEST"]
seed = os.environ["FUTILITY_SEED_UNDER_TEST"]
_, verdict = load_programs()
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
    art="${RESULTS}/dial_t${task}_${arm}_${PROFILE}_1x10_${TAG}"

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
        --job-name="rma-dial-t${task}-1x10"
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
echo "[OK] submitted ${submitted} DIAL hard-3 job(s) tasks='${TASKS}' arm(s)='${ARMS_CSV}' tag=${TAG}."
echo "     Success criterion: mean(stage) > GPM seed100 (t5>20.0, t8>46.7, t22>73.3)."
