#!/usr/bin/env bash
# Compare baseline vs harness A/B outputs (same tasks/seeds).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_ROOT="${1:?usage: compare_harness_abtest.sh BASELINE_OUT HARNESS_OUT}"
HARNESS_ROOT="${2:?}"

for out in "${BASELINE_ROOT}" "${HARNESS_ROOT}"; do
  if [[ ! -f "${out}/summary.tsv" ]] && [[ -d "${out}/task1" || -d "${out}/tasks2to26" ]]; then
    echo "[INFO] merging ${out} ..."
    bash "${ROOT}/scripts/merge_eval_outputs.sh" "${out}"
  fi
done

python3 - "${BASELINE_ROOT}" "${HARNESS_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

baseline = Path(sys.argv[1])
harness = Path(sys.argv[2])


def load_summary(path: Path) -> dict[int, dict]:
    rows = {}
    tsv = path / "summary.tsv"
    if not tsv.is_file():
        return rows
    lines = tsv.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        row = dict(zip(header, parts))
        rows[int(row["task_id"])] = row
    return rows


def load_aggregate(path: Path) -> dict:
    p = path / "aggregate.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


b_rows = load_summary(baseline)
h_rows = load_summary(harness)
b_agg = load_aggregate(baseline)
h_agg = load_aggregate(harness)

print(f"Baseline: {baseline}")
print(f"Harness:  {harness}")
print()

if b_agg:
    print(
        "Baseline aggregate: "
        f"TSR={float(b_agg.get('macro_stage_success_rate', 0)):.3f} "
        f"CSR={float(b_agg.get('macro_goal_success_rate', 0)):.3f}"
    )
if h_agg:
    print(
        "Harness aggregate:  "
        f"TSR={float(h_agg.get('macro_stage_success_rate', 0)):.3f} "
        f"CSR={float(h_agg.get('macro_goal_success_rate', 0)):.3f}"
    )
print()

common = sorted(set(b_rows) & set(h_rows))
if not common:
    print("[WARN] no overlapping tasks in summary.tsv")
    raise SystemExit(0)

print("Per-task delta (Harness - Baseline):")
print("task_id  dTSR    dCSR    base_TSR  harness_TSR  base_CSR  harness_CSR")
improved_tsr = improved_csr = 0
for tid in common:
    b = b_rows[tid]
    h = h_rows[tid]
    b_tsr = float(b["stage_success_rate"])
    h_tsr = float(h["stage_success_rate"])
    b_csr = float(b["goal_success_rate"])
    h_csr = float(h["goal_success_rate"])
    d_tsr = h_tsr - b_tsr
    d_csr = h_csr - b_csr
    improved_tsr += int(d_tsr > 0)
    improved_csr += int(d_csr > 0)
    print(
        f"{tid:7d}  {d_tsr:+.3f}  {d_csr:+.3f}  "
        f"{b_tsr:.3f}     {h_tsr:.3f}        {b_csr:.3f}     {h_csr:.3f}"
    )

print()
print(f"Tasks improved TSR: {improved_tsr}/{len(common)}")
print(f"Tasks improved CSR: {improved_csr}/{len(common)}")
if b_agg and h_agg:
    d_tsr = float(h_agg.get("macro_stage_success_rate", 0)) - float(b_agg.get("macro_stage_success_rate", 0))
    d_csr = float(h_agg.get("macro_goal_success_rate", 0)) - float(b_agg.get("macro_goal_success_rate", 0))
    print(f"Macro delta TSR: {d_tsr:+.3f}")
    print(f"Macro delta CSR: {d_csr:+.3f}")
PY
