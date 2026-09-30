#!/usr/bin/env bash
# ECHO t8 h0 1x10, seed 100; after 3 trials stop at mean <=40 or 3 trailing zeros.
# The archived same-seed GPM running mean remains >40 at every checked prefix.
set -euo pipefail
ROOT=/project/peilab/why/RoboMemArena
EXP_DIR="${ROOT}/experiments/mem_efficacy"
PY="${ROOT}/.venv/bin/python"
TAG="${1:-v2_strict40}"
[[ "${TAG}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid tag" >&2; exit 2; }
cd "${EXP_DIR}"
echo '[gate] AST + ECHO selftest'
"${PY}" -m compileall -q echo memexp_echo_bind.py pysite/sitecustomize.py
PYTHONPATH="${EXP_DIR}:${ROOT}/evaluation_benchmark:${ROOT}/evaluation_benchmark/scripts" "${PY}" -m echo.selftest
# Exact futility logic and archived same-seed GPM, not a hand-copied approximation.
echo '[gate] replay strict futility on same-seed GPM'
FUTILITY_WATCH_UNDER_TEST=3 FUTILITY_FLOOR_UNDER_TEST=40 FUTILITY_CONSEC_UNDER_TEST=3 \
FUTILITY_TASK_UNDER_TEST=8 FUTILITY_SEED_UNDER_TEST=100 PYTHONPATH="${EXP_DIR}" \
"${PY}" - <<'PY'
import csv, sys
from selftest_futility import load_programs, run_verdict, _cells
_, verdict = load_programs()
cells = [p for p in _cells('evmem_gpm_t8_*/h0/*/prompt_trace.tsv') if 's100' in p]
if len(cells) != 1:
    raise SystemExit(f'expected exactly one GPM seed100 reference, got {cells}')
with open(cells[0]) as fh:
    scores = [float(row['stage_score_pct']) for row in csv.DictReader(fh, delimiter='\t')]
if len(scores) != 10:
    raise SystemExit(f'GPM reference incomplete: {scores}')
for k in range(3, 10):
    series = ','.join(f'{v:g}' for v in scores[:k])
    result = run_verdict(verdict, k, max(scores[:k]), sum(scores[:k])/k, 40, 3,
                         cz=3, ser=series)
    if result != 'wait':
        raise SystemExit(f'futility would early-kill GPM at trial {k}: {result}')
print(f'GPM survives: mean={sum(scores)/10:.1f}')
PY
ART="${EXP_DIR}/results/echo_t8_echo_h0_1x10_${TAG}"
if [[ -e "${ART}" ]]; then
  echo "[ERROR] result path exists; choose a new tag: ${ART}" >&2
  exit 2
fi
if [[ "${ECHO_DRY_RUN:-0}" == 1 ]]; then
  echo "[dry-run] output=${ART} task=8 arm=echo seed100 trials=10 watch=3 floor=40 consecutive_zero=3"
  exit 0
fi
STAGE1_PROFILES=h0 STAGE1_TRIALS=10 ARM_OVERRIDE=echo TASKS_JSON_SCOPE='[8]' \
SEED_LIST=100 STAGE1_ART_ROOT="${ART}" STAGE1_MAX_WALL_SEC=6400 \
STAGE1_ALLOW_RERUN=1 PLANNER_MAX_CONSECUTIVE_FAILURES=3 EARLY_STOP_ON_QUOTA=1 \
FUTILITY_STOP_TRIALS=3 FUTILITY_MEAN_FLOOR_PCT=40 FUTILITY_CONSEC_ZERO_M=3 \
sbatch --job-name=rma-echo-t8-1x10 --time=03:30:00 "${EXP_DIR}/run_stage1.sbatch"
