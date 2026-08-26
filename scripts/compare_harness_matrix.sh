#!/usr/bin/env bash
# Compare multiple harness variant outputs against a shared baseline.
# Usage: compare_harness_matrix.sh BASELINE_ROOT VARIANT_ROOT [VARIANT_ROOT ...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${1:?usage: compare_harness_matrix.sh BASELINE_ROOT VARIANT_ROOT ...}"
shift

python3 - "${BASELINE}" "$@" <<'PY'
import json
import sys
from pathlib import Path

baseline_root = Path(sys.argv[1])
variant_roots = [Path(p) for p in sys.argv[2:]]


def load_rows(path: Path) -> dict[int, dict]:
    tsv = path / "summary.tsv"
    if not tsv.is_file():
        return {}
    lines = tsv.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return {}
    header = lines[0].split("\t")
    rows = {}
    for line in lines[1:]:
        parts = line.split("\t")
        row = dict(zip(header, parts))
        rows[int(row["task_id"])] = row
    return rows


def load_agg(path: Path) -> dict:
    p = path / "aggregate.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def fmt_rate(x: float) -> str:
    return f"{x:.3f}"


b_rows = load_rows(baseline_root)
b_agg = load_agg(baseline_root)
b_csr = float(b_agg.get("macro_goal_success_rate", 0))
b_tsr = float(b_agg.get("macro_stage_success_rate", 0))

print("# Harness Ablation Matrix")
print()
print(f"Baseline: `{baseline_root}`")
print(f"- TSR={fmt_rate(b_tsr)} CSR={fmt_rate(b_csr)}")
print()
print("| Variant | TSR | CSR | ΔTSR | ΔCSR | improved CSR |")
print("|---------|-----|-----|------|------|--------------|")

for vroot in variant_roots:
    name = vroot.name
    if name.startswith("variant_"):
        name = name[len("variant_"):]
    agg = load_agg(vroot)
    rows = load_rows(vroot)
    tsr = float(agg.get("macro_stage_success_rate", 0))
    csr = float(agg.get("macro_goal_success_rate", 0))
    common = sorted(set(b_rows) & set(rows))
    improved = sum(
        1 for tid in common
        if float(rows[tid]["goal_success_rate"]) > float(b_rows[tid]["goal_success_rate"])
    )
    print(
        f"| {name} | {fmt_rate(tsr)} | {fmt_rate(csr)} | "
        f"{tsr - b_tsr:+.3f} | {csr - b_csr:+.3f} | {improved}/{len(common) or '-'} |"
    )

print()
print("## Per-task CSR delta vs baseline")
for vroot in variant_roots:
    name = vroot.name
    if name.startswith("variant_"):
        name = name[len("variant_"):]
    rows = load_rows(vroot)
    common = sorted(set(b_rows) & set(rows))
    if not common:
        continue
    deltas = []
    for tid in common:
        d = float(rows[tid]["goal_success_rate"]) - float(b_rows[tid]["goal_success_rate"])
        if abs(d) > 1e-6:
            deltas.append(f"t{tid}{d:+.2f}")
    if deltas:
        print(f"- **{name}**: {', '.join(deltas)}")
PY
