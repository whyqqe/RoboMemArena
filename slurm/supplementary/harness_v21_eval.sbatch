#!/usr/bin/env bash
#SBATCH --job-name=rma-v21
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# v2.1 harness eval: baseline + v2_full + v21_combo (3 seeds).
# 8 memory-heavy tasks, compare CSR/TSR vs shared baseline.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_BASE="${ROOT}/outputs/harness_v21_${JOB_TAG}"
BASELINE_OUT="${OUT_BASE}/baseline_s100"
V2_OUT="${OUT_BASE}/v2_full_s100"

# Same subset as 529824 A/B (strong memory signal)
TASKS='[1,4,5,11,14,16,18,22]'
SEEDS=(100 101 102)

mkdir -p "${ROOT}/outputs/slurm" "${OUT_BASE}"
cd "${ROOT}"

module load slurm "nvhpc-hpcx-cuda12/23.11" 2>/dev/null || true

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
for _glu_dir in \
  /usr/lib/x86_64-linux-gnu \
  /usr/lib64 \
  /cm/shared/apps/mesa/*/lib; do
  if [[ -f "${_glu_dir}/libGLU.so.1" || -f "${_glu_dir}/libGLU.so" ]]; then
    export LD_LIBRARY_PATH="${_glu_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    break
  fi
done
unset _glu_dir

# shellcheck disable=SC1091
source "${ROOT}/scripts/env_common.sh"
robomem_setup_modules
robomem_setup_cache

export OPENPI_ROOT="${ROOT}/third_party/openpi_minimal"
export OPENPI_INFERENCE_ROOT="${ROOT}"
export LIBERO_FORK_ROOT="${ROOT}/evaluation_benchmark/libero_fork"
export TARGET_LIBERO_PATH="${LIBERO_FORK_ROOT}/libero"
export PREDIMEM_HF_SNAPSHOT="${ROOT}/checkpoints/PrediMem"
export SERVER_PY="${OPENPI_ROOT}/.venv/bin/python"
export EVAL_PY="${ROOT}/.venv/bin/python"

export TASKS_JSON="${TASKS}"
export NUM_TRIALS=1
export PORT=8026
export RESIZE_SIZE=256
export REPLAN_STEPS=10
export NUM_STEPS_WAIT=10
export MAX_STEPS=2500
export POST_GOAL_STEPS=200
export POST_STAGE_STEPS=30
export FAIL_ON_EXTRA_POUR=1
export ASYNC_VLM=1
export VLM_INTERVAL=5
export VLM_QUEUE_SIZE=1
export VLM_USE_WRIST=1
export VLM_INPUT_PROFILE=fullvlm_256
export VLM_MATCH_TRAINING_JPEG_ROUNDTRIP=0
export VLA_CONFIG=pi05_robomemarena
unset PREDIMEM_ASSET_GROUP

NGPU=$("${EVAL_PY}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "${NGPU}" -lt 2 ]]; then
  echo "[ERROR] need >=2 GPUs; got ${NGPU}" >&2
  exit 2
fi
export SERVER_CUDA_VISIBLE_DEVICES=0
export EVAL_CUDA_VISIBLE_DEVICES=1

run_variant() {
  local variant="$1"
  local out_root="$2"
  local seed="$3"
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/run_harness_variant.sh" "${variant}"
  export OUT_ROOT="${out_root}"
  export SEED="${seed}"
  echo "[INFO] === variant=${variant} seed=${seed} OUT_ROOT=${OUT_ROOT} ==="
  cd "${ROOT}/evaluation_benchmark/async_vlm26_reference"
  bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
  bash "${ROOT}/scripts/merge_eval_outputs.sh" "${OUT_ROOT}"
}

echo "[INFO] hostname=$(hostname) job=${JOB_TAG}"
echo "[INFO] OUT_BASE=${OUT_BASE} TASKS_JSON=${TASKS_JSON}"

run_variant baseline "${BASELINE_OUT}" 100
run_variant v2_full "${V2_OUT}" 100

V21_PATHS=()
for s in "${SEEDS[@]}"; do
  p="${OUT_BASE}/v21_combo_s${s}"
  run_variant v21_combo "${p}" "${s}"
  V21_PATHS+=("${p}")
done

echo "[INFO] Comparing v2_full vs baseline..."
bash "${ROOT}/scripts/compare_harness_abtest.sh" "${BASELINE_OUT}" "${V2_OUT}" | tee "${OUT_BASE}/compare_v2_vs_baseline.txt"

echo "[INFO] Comparing v21 seeds vs baseline..."
for s in "${SEEDS[@]}"; do
  p="${OUT_BASE}/v21_combo_s${s}"
  echo "--- v21_combo seed=${s} ---"
  bash "${ROOT}/scripts/compare_harness_abtest.sh" "${BASELINE_OUT}" "${p}"
done | tee "${OUT_BASE}/compare_v21_vs_baseline.txt"

python3 - "${OUT_BASE}" "${BASELINE_OUT}" "${V2_OUT}" "${V21_PATHS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_base = Path(sys.argv[1])
baseline = Path(sys.argv[2])
v2 = Path(sys.argv[3])
v21_paths = [Path(p) for p in sys.argv[4:]]

def agg(p):
    f = p / "aggregate.json"
    return json.loads(f.read_text()) if f.is_file() else {}

def rows(p):
    tsv = p / "summary.tsv"
    if not tsv.is_file():
        return {}
    lines = tsv.read_text().splitlines()
    if len(lines) <= 1:
        return {}
    hdr = lines[0].split("\t")
    out = {}
    for line in lines[1:]:
        parts = line.split("\t")
        row = dict(zip(hdr, parts))
        out[int(row["task_id"])] = float(row["goal_success_rate"])
    return out

b = agg(baseline)
v2a = agg(v2)
v21_aggs = [agg(p) for p in v21_paths]

print("# v2.1 Evaluation Summary")
print()
print(f"| Config | TSR | CSR | ΔCSR vs baseline |")
print(f"| baseline | {b.get('macro_stage_success_rate',0):.3f} | {b.get('macro_goal_success_rate',0):.3f} | — |")
print(
    f"| v2_full | {v2a.get('macro_stage_success_rate',0):.3f} | "
    f"{v2a.get('macro_goal_success_rate',0):.3f} | "
    f"{v2a.get('macro_goal_success_rate',0)-b.get('macro_goal_success_rate',0):+.3f} |"
)
if v21_aggs:
    mean_csr = sum(a.get("macro_goal_success_rate", 0) for a in v21_aggs) / len(v21_aggs)
    mean_tsr = sum(a.get("macro_stage_success_rate", 0) for a in v21_aggs) / len(v21_aggs)
    print(f"| v21_combo (mean {len(v21_aggs)} seeds) | {mean_tsr:.3f} | {mean_csr:.3f} | {mean_csr-b.get('macro_goal_success_rate',0):+.3f} |")
    for i, (p, a) in enumerate(zip(v21_paths, v21_aggs)):
        print(f"| v21_combo s{100+i} | {a.get('macro_stage_success_rate',0):.3f} | {a.get('macro_goal_success_rate',0):.3f} | {a.get('macro_goal_success_rate',0)-b.get('macro_goal_success_rate',0):+.3f} |")

# Per-task mean CSR for v21 across seeds
b_rows = rows(baseline)
if v21_paths:
    task_ids = sorted(b_rows.keys())
    print()
    print("Per-task CSR (v21 mean vs baseline):")
    for tid in task_ids:
        v21_csrs = [rows(p).get(tid, 0.0) for p in v21_paths]
        m = sum(v21_csrs) / len(v21_csrs)
        d = m - b_rows.get(tid, 0.0)
        if abs(d) > 1e-4:
            print(f"  task{tid}: baseline={b_rows[tid]:.3f} v21_mean={m:.3f} delta={d:+.3f}")

summary = {
    "baseline": {"path": str(baseline), **b},
    "v2_full": {"path": str(v2), **v2a},
    "v21_combo": [{"path": str(p), **a} for p, a in zip(v21_paths, v21_aggs)],
}
if v21_aggs:
    summary["v21_combo_macro_mean"] = {
        "macro_stage_success_rate": mean_tsr,
        "macro_goal_success_rate": mean_csr,
    }
(out_base / "v21_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
report = out_base / "v21_report.md"
report.write_text(
    "# Harness v2.1 Eval\n\nSee v21_summary.json and compare_*.txt\n",
    encoding="utf-8",
)
print(f"[OK] wrote {out_base / 'v21_summary.json'}")
PY

echo "[OK] v2.1 eval complete. OUT_BASE=${OUT_BASE}"
