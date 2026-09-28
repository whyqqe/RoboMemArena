#!/usr/bin/env bash
# =========================================================================================
# AOM: task 8, 1x10, h0, STRICT early-stop.  Then, if it passes, the remaining hard tasks.
#
# WHY THIS FILE EXISTS
#   The new architecture (AOM) has to be tested on the ONE task where the previous architecture
#   family (GPM / SR / PIC) has a measured number, under a futility gate the user asked to be
#   STRICTER than before ("连续挂0则停"). The gate is the part that has to be argued, not typed:
#   a false fire destroys a cell's remaining trials and is INDISTINGUISHABLE in the results from
#   an arm that genuinely fails, so the calibration is spelled out below and asserted in
#   `selftest_futility.py` T8b/T9.
#
# THE GATE, AND WHY IT IS `w=6, M=3`
#   FUTILITY_CONSEC_ZERO_M=3  stop once the LAST 3 scored trials are all 0.0.
#   FUTILITY_STOP_TRIALS=6    arm all rules only after 6 scored trials (the shared window).
#   FUTILITY_MEAN_FLOOR_PCT=0 NO absolute floor. Task 8 is the task where an absolute floor has
#                             no separating power: its no-memory control's early running mean
#                             reaches 0.0, so **FLOOR=35 fires on that control at trial 6**
#                             (scores 0/66.7/33.3/66.7/0/0 -> mean 27.8 <= 35). The previous PIC
#                             run used 35 and that is exactly the gate that produced its early
#                             stops. A gate that kills the control cannot measure the treatment.
#
#   The window is load-bearing, and the calibration is a REPLAY over the 113 archived cells in
#   `results/`:
#     M=2, w=6 -> trips the task-8 GPM baseline ITSELF at k=8 ([...,100,0,0]). GPM's 10-seed scores
#                 are 66.7/100/66.7/0/66.7/100/0/0/66.7/0, so its longest zero run is 2: M=2 is a
#                 VARIANCE detector, not a futility one. Rejected.
#     M=3, w=6 -> the GPM baseline NEVER trips. This is the configuration used here.
#     M=3, w=3 -> trips 21 archived cells, several of which recovered (evmem_sr_need_t8 trips at
#                 k=3 with mean 0.0 and finishes at 27.8). Arming early is what causes false stops.
#
# WHAT THIS GATE COSTS, STATED UP FRONT
#   At w=6/M=3 the rule would have stopped 9 archived cells early. Every one of them has
#   mean@trip ~= final (e.g. 22.2 -> 22.2, 13.3 -> 13.3), i.e. they were already dead; none is the
#   task-8 GPM baseline the treatment is compared against. That list is PRINTED by
#   `python selftest_futility.py` (T9) rather than buried here, so the cost stays visible if the
#   archive grows.
#
# HONEST EXPECTATION
#   Task 8's scored stages are ACTUAL, so AOM's PERCEPTUAL and DERIVED modes are NOT exercised.
#   AOM should land NEAR GPM (46.7), not above it. The informative outputs are the counters
#   (n_stagnant / n_dep_rejects / derived_values) in `memexp_aom_report.json`, plus a gate that
#   did NOT fire -- which is itself the evidence that the run is complete rather than truncated.
#
# USAGE
#   ./run_aom_t8_1x10.sh --dry-run
#   ./run_aom_t8_1x10.sh                          # arm evmem_aom
#   ./run_aom_t8_1x10.sh --arms evmem_aom,evmem_aom_noderived
#   ./run_aom_t8_1x10.sh --then-remaining          # chain t5/t19/t22 behind `afterok` on t8
# =========================================================================================

set -uo pipefail

ROOT="/project/peilab/why/RoboMemArena"
EXP_DIR="${ROOT}/experiments/mem_efficacy"
RESULTS="${EXP_DIR}/results"

TASK=8
SEEDS="100"
TRIALS=10
PROFILE="h0"
FUTILITY_TRIALS=6
FUTILITY_CONSEC_M=3
FLOOR=0                     # see the calibration note above: task 8 gets NO absolute floor
BASELINE_GPM=46.7
WALL=4800
STIME="03:00:00"

DRY=""
THEN=""
ARMS_CSV="evmem_aom"
# Fresh artifact tag. The v1 dirs under aom_t*_evmem_aom_h0_1x10 hold the 617134-617137
# crash forensics (IndentationError in the shared evaluator); they must not be overwritten,
# and a novelty check with --allow-rerun would still leave two incomplete cells mixed in one
# tree. Default is therefore a new tag; override with --tag if you really want to reuse.
TAG="v2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)         DRY=1 ;;
    --then-remaining)  THEN=1 ;;
    --arms)            ARMS_CSV="${2:?--arms needs a comma-separated list}"; shift ;;
    --arms=*)          ARMS_CSV="${1#--arms=}" ;;
    --tag)             TAG="${2:?--tag needs a label}"; shift ;;
    --tag=*)           TAG="${1#--tag=}" ;;
    -h|--help)         sed -n '1,70p' "${BASH_SOURCE[0]}"; exit 0 ;;
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

# PRE-SUBMIT GATE. Jobs 617134-617137 each spent ~90s and a VLA server start to discover that
# `eval_fullvlm26_async_vlm_vla.py` had an IndentationError at line 2689 -- a fault that is
# visible in milliseconds with `ast.parse` and that no amount of AOM tuning can paper over.
# Refusing to submit when the shared evaluator cannot parse is the only way that class of
# failure stops costing GPU queue time. Census is checked for the same reason: a broken
# census turned the real diagnosis into an UnboundLocalError and hid the evaluator crash.
EVAL_PY="${ROOT}/.venv/bin/python"
EVALUATOR="${ROOT}/evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py"
echo "[gate] compiling shared evaluator + census before any sbatch ..."
if ! "${EVAL_PY}" -c "
import ast, sys
paths = [
    r'''${EVALUATOR}''',
    r'''${EXP_DIR}/census_channels.py''',
    r'''${EXP_DIR}/memexp_aom.py''',
    r'''${EXP_DIR}/memexp_aom_bind.py''',
]
for path in paths:
    try:
        ast.parse(open(path, encoding='utf-8').read())
    except SyntaxError as e:
        print(f'SYNTAX: {path}:{e.lineno}: {e.msg}', file=sys.stderr)
        sys.exit(2)
print('compile OK')
"; then
  echo "[ERROR] pre-submit compile gate FAILED -- refusing to queue GPU jobs behind a" >&2
  echo "        SyntaxError the evaluator would hit on its first import." >&2
  exit 2
fi

echo "-----------------------------------------------------------------------------------"
echo "AOM  task=${TASK}  arms='${ARMS_CSV}'  seeds='${SEEDS}'  trials=${TRIALS}  profile=${PROFILE}"
echo "  tag=${TAG}  (artifact dirs: aom_t*_*_${PROFILE}_1x10_${TAG})"
echo "  futility: after ${FUTILITY_TRIALS} trials -> stop if EVERY trial is 0.0"
echo "            or if the LAST ${FUTILITY_CONSEC_M} scored trials are ALL 0.0"
echo "            (no absolute floor: task 8's control reaches an early mean of 0.0)"
echo "  reference: GPM (evmem) on the same seeds = ${BASELINE_GPM}"
echo "  arch: AOM -- one obligation type {ACTUAL,PERCEPTUAL,DERIVED}, one arbitration law,"
echo "        one board, admissibility derived from the obligation graph"
if [[ -n "${THEN}" ]]; then
  echo "  chain: t5/t19/t22 will be submitted with --dependency=afterok on the t8 jobs"
fi
echo "-----------------------------------------------------------------------------------"

submitted=0
JOBIDS=""
for arm in "${ARMS[@]}"; do
  art="${RESULTS}/aom_t${TASK}_${arm}_${PROFILE}_1x10_${TAG}"
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
    echo "  [dry-run] arm=${arm} -> art=${art}"
    printf '    %q' "${cmd[@]}"; echo; echo
    continue
  fi

  out="$("${cmd[@]}" 2>&1)"
  rc=$?
  echo "${out}"
  if [[ ${rc} -ne 0 ]] || ! printf '%s' "${out}" | grep -q '^Submitted batch job '; then
    echo "[ERROR] submit failed for arm=${arm} rc=${rc}" >&2
    exit 1
  fi
  jid="$(printf '%s' "${out}" | sed -n 's/^Submitted batch job \([0-9]\+\).*/\1/p' | tail -1)"
  submitted=$((submitted + 1))
  JOBIDS="${JOBIDS}${JOBIDS:+,}${jid}"
  echo "[OK] submitted AOM t${TASK} arm=${arm} job=${jid} -> ${art}"
  echo
done

if [[ -n "${DRY}" ]]; then
  if [[ -n "${THEN}" ]]; then
    echo "  [dry-run] then: ${EXP_DIR}/run_aom_remaining_1x10.sh --after <t8-jobid> --tag ${TAG}"
  fi
  exit 0
fi

echo "[OK] submitted ${submitted} AOM t${TASK} job(s): ${JOBIDS}"

if [[ -z "${THEN}" ]]; then
  echo
  echo "To start the remaining hard tasks ONLY IF t8 passes, run:"
  echo "  ${EXP_DIR}/run_aom_remaining_1x10.sh --after ${JOBIDS%%,*} --tag ${TAG}"
  exit 0
fi

# The chain. `afterok` is the gate the user asked for and it is enforced by Slurm rather than by
# this script's bookkeeping: run_stage1.sbatch exits 79 on a futility stop, 77 on quota and 78 on
# a health stop, so a t8 that was cut short leaves the dependency UNSATISFIED and the remaining
# tasks are never launched. A t8 that completes all ten trials exits 0 and releases them.
# Note the deliberate consequence: if t8 dies for ANY non-zero reason the chain also stops, which
# is the safe direction -- three 4-hour GPU jobs must not be spent behind an unread t8 log.
echo
echo "[chain] releasing t5/t19/t22 only if t8 exits 0 (no futility/quota/health stop)"
"${EXP_DIR}/run_aom_remaining_1x10.sh" --after "${JOBIDS%%,*}" --arms "${ARMS_CSV}" --tag "${TAG}"
