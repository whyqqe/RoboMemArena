#!/usr/bin/env bash
# Aggregate multi-seed harness main experiment (paired baseline vs harness per seed).
# Usage: aggregate_multiseed_main.sh OUT_BASE [baseline_prefix] [harness_prefix]
set -euo pipefail

OUT_BASE="${1:?usage: aggregate_multiseed_main.sh OUT_BASE}"
BASE_PREFIX="${2:-baseline_s}"
HARNESS_PREFIX="${3:-v21_best_s}"

python3 - "${OUT_BASE}" "${BASE_PREFIX}" "${HARNESS_PREFIX}" <<'PY'
import json
import math
import sys
from pathlib import Path

out_base = Path(sys.argv[1])
base_prefix = sys.argv[2]
harness_prefix = sys.argv[3]


def agg(path: Path) -> dict:
    p = path / "aggregate.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def rows(path: Path) -> dict[int, float]:
    tsv = path / "summary.tsv"
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


def find_runs(prefix: str) -> list[tuple[int, Path]]:
    runs = []
    for p in sorted(out_base.iterdir()):
        if not p.is_dir() or not p.name.startswith(prefix):
            continue
        suffix = p.name[len(prefix):]
        try:
            seed = int(suffix)
        except ValueError:
            continue
        runs.append((seed, p))
    return sorted(runs)


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


base_runs = find_runs(base_prefix)
harness_runs = find_runs(harness_prefix)
base_by_seed = dict(base_runs)
harness_by_seed = dict(harness_runs)
common_seeds = sorted(set(base_by_seed) & set(harness_by_seed))

print("# Multi-seed Main Experiment Summary")
print()
print(f"OUT_BASE: `{out_base}`")
print(f"Seeds: {common_seeds}")
print()

base_csr = [agg(base_by_seed[s]).get("macro_goal_success_rate", 0) for s in common_seeds]
base_tsr = [agg(base_by_seed[s]).get("macro_stage_success_rate", 0) for s in common_seeds]
har_csr = [agg(harness_by_seed[s]).get("macro_goal_success_rate", 0) for s in common_seeds]
har_tsr = [agg(harness_by_seed[s]).get("macro_stage_success_rate", 0) for s in common_seeds]
delta_csr = [h - b for h, b in zip(har_csr, base_csr)]

bm, bs = mean_std(base_csr)
hm, hs = mean_std(har_csr)
dm, ds = mean_std(delta_csr)
tm, ts = mean_std(har_tsr)
btm, bts = mean_std(base_tsr)

print("| Config | CSR mean±std | TSR mean±std |")
print("|--------|--------------|--------------|")
print(f"| baseline | {bm:.3f}±{bs:.3f} | {btm:.3f}±{bts:.3f} |")
print(f"| harness | {hm:.3f}±{hs:.3f} | {tm:.3f}±{ts:.3f} |")
print(f"| delta (h-b) | {hm-bm:+.3f} (paired Δ mean {dm:+.3f}±{ds:.3f}) | {tm-btm:+.3f} |")
print()

print("Per-seed CSR:")
for s in common_seeds:
    b = agg(base_by_seed[s]).get("macro_goal_success_rate", 0)
    h = agg(harness_by_seed[s]).get("macro_goal_success_rate", 0)
    print(f"  seed {s}: baseline={b:.3f} harness={h:.3f} delta={h-b:+.3f}")

# Per-task mean delta across seeds
if common_seeds:
    task_ids = sorted(rows(base_by_seed[common_seeds[0]]).keys())
    print()
    print("Per-task CSR delta (harness - baseline, mean over seeds):")
    for tid in task_ids:
        deltas = []
        for s in common_seeds:
            br = rows(base_by_seed[s])
            hr = rows(harness_by_seed[s])
            if tid in br and tid in hr:
                deltas.append(hr[tid] - br[tid])
        if deltas:
            m = sum(deltas) / len(deltas)
            if abs(m) > 1e-4:
                print(f"  task{tid}: {m:+.3f}")

summary = {
    "seeds": common_seeds,
    "baseline": {
        "csr_mean": bm,
        "csr_std": bs,
        "tsr_mean": btm,
        "tsr_std": bts,
        "per_seed_csr": {str(s): agg(base_by_seed[s]).get("macro_goal_success_rate", 0) for s in common_seeds},
    },
    "harness": {
        "prefix": harness_prefix,
        "csr_mean": hm,
        "csr_std": hs,
        "tsr_mean": tm,
        "tsr_std": ts,
        "per_seed_csr": {str(s): agg(harness_by_seed[s]).get("macro_goal_success_rate", 0) for s in common_seeds},
    },
    "delta_csr_mean": hm - bm,
    "paired_delta_csr_mean": dm,
    "paired_delta_csr_std": ds,
}
out_path = out_base / "main_multiseed_summary.json"
out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print()
print(f"[OK] wrote {out_path}")
PY
