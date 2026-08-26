#!/usr/bin/env bash
# Monitor RoboMemArena Slurm jobs / eval outputs.
ROOT=/project/peilab/why/RoboMemArena
JOB_ID="${1:-}"

echo "=== queue ==="
squeue -u "${USER}" -o "%.18i %.12P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null || true

if [[ -n "${JOB_ID}" ]]; then
  echo "=== job ${JOB_ID} ==="
  scontrol show job "${JOB_ID}" 2>/dev/null | head -20 || true
  echo "=== slurm out (tail) ==="
  ls -t "${ROOT}/outputs/slurm/"*${JOB_ID}* 2>/dev/null | head -4
  for f in "${ROOT}/outputs/slurm/"*${JOB_ID}*.out "${ROOT}/outputs/slurm/"*${JOB_ID}*.err; do
    [[ -f "$f" ]] || continue
    echo "--- $f ---"
    tail -n 40 "$f"
  done
  OUT=""
  for pat in "eval_repro_all26_1seed_${JOB_ID}" "eval_repro_task1_${JOB_ID}" "eval_smoke_task1_${JOB_ID}"; do
    if [[ -d "${ROOT}/outputs/${pat}" ]]; then
      OUT="${ROOT}/outputs/${pat}"
      break
    fi
  done
  if [[ -n "${OUT}" ]]; then
    echo "=== OUT_ROOT=${OUT} ==="
    ls -la "${OUT}" 2>/dev/null | head -20
    echo "--- summary.tsv ---"
    cat "${OUT}/summary.tsv" 2>/dev/null || true
    echo "--- aggregate.json ---"
    cat "${OUT}/aggregate.json" 2>/dev/null || true
    echo "--- eval log tail ---"
    tail -n 50 "${OUT}/logs/eval_fullvlm26_async_vlm_vla.log" 2>/dev/null || true
    echo "--- server log tail ---"
    tail -n 30 "${OUT}/logs/serve_policy.log" 2>/dev/null || true
  fi
else
  echo "Usage: bash scripts/monitor_eval.sh [JOB_ID]"
  echo "Latest smoke outputs:"
  ls -td "${ROOT}/outputs/eval_smoke_task1_"* 2>/dev/null | head -5 || echo "(none yet)"
fi
